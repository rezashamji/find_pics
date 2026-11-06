import Foundation
import CoreImage
import FindPicsCore
import HuggingFace
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// THE NIGHT PASS (M78). A small vision model looks at each photo ONCE, while the phone charges, and writes one
/// line of what it sees; that line becomes 384 numbers a search can read in milliseconds.
///
/// WHY THIS EXISTS, in Reza's words (11-05): "if things run on time they arent on app thats ok" - heavy work at
/// night is fine, waiting at search time is not. Today the photo judge opens ~600 photos while he waits, which is
/// why "food photos" took 14.6 minutes. The cluster measured (RESULTS 59-69) that facts written once give the same
/// quality with ~65 judge calls instead of ~1,200.
///
/// What it is NOT: this does not replace the judge. The facts scorer is as PRECISE as the judge (.975 vs .979) but
/// finds only ~79% of its matches, and it is weakest at exactly the things Reza cares about - counting (.468),
/// how a person looks (.736), colour (.755). So the judge stays for the hard tail; the facts remove the easy 95%.
actor FactsWriter {
    /// The writer's prompt and sampling, fixed by M77 and measured on the device at 0.18 s vision+prefill and
    /// 94.1 tokens/s. The shipped student writes ~74 tokens, so ~0.97 s per photo.
    static let prompt = "Search facts for this photo, one line."
    static let maxTokens = 48      // K = 48: RESULTS 70 measured no quality cost vs uncapped, ~47.5 tokens instead of 74 (M87(4))
    /// The 448 px rendition the index already makes - NOT the judge's 896 px. The writer runs at index time where
    /// the cheaper read is what the whole pass can afford.
    static let side = 448.0

    private let writerID: String
    private var writer: Judge?
    private var encoder: EmbedderModelContainer?
    private var tokenizer: FindPicsCore.BertTokenizer?

    /// THE TRAINED WRITER (cluster M84, verified 11-05: 1,782,040,976 B, sha256 771d8304...). The base
    /// mlx-community 2B answers the prompt with a short caption (median 22 tokens measured in M77); this one
    /// writes the full field line it was trained for (~74 tokens), which is what every cluster number assumes.
    init(writerID: String = "rezashamji/findpics-writer-2b") { self.writerID = writerID }

    /// Loads the writer and the text encoder. Both stay resident for the pass; the pass is long and the loads are
    /// the expensive part (the 4B judge takes ~2 s to load, measured).
    /// The text encoder alone (33M params): everything the comparison needs, without the 1.7 GB writer. Measured
    /// 11-05: loading both the writer and the photo judge left 1.07 GB free with 4.60 GB resident, which is the
    /// territory the app was killed in last month.
    func loadEncoderOnly() async throws {
        try await loadEncoder()
    }

    func load() async throws {
        if writer == nil {
            let w = Judge(modelID: writerID)
            try await w.load { _ in }
            writer = w
        }
        try await loadEncoder()
    }

    private func loadEncoder() async throws {
        if encoder == nil {
            guard let dir = Bundle.module.url(forResource: "towervec_encoder_fp16", withExtension: nil,
                                              subdirectory: "Models/facts") else {
                throw NSError(domain: "FactsWriter", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "towervec_encoder_fp16 not in the bundle"])
            }
            // Same loader the judge uses (#huggingFaceTokenizerLoader = Tokenizers.AutoTokenizer); the weights are
            // already in the bundle, so nothing is downloaded.
            encoder = try await EmbedderModelFactory.shared.loadContainer(
                from: dir, using: #huggingFaceTokenizerLoader())
            tokenizer = try FindPicsCore.BertTokenizer(vocabURL: dir.appendingPathComponent("vocab.txt"))
        }
    }

    func unload() async {
        if let w = writer { await w.unload() }
        writer = nil; encoder = nil; tokenizer = nil
    }

    /// One photo: the written line (stopped at its first repeat) and its 384 numbers, or nil when the photo cannot
    /// be read. `place` and `taken` go in front of the line exactly as the cluster's training data had them.
    /// Read seconds and model seconds, kept apart. Without this the pass reports one number and every explanation
    /// of it is a guess - which is how I called 2.4 s "the read is fatal", then 0.59 s "the read is free", then
    /// 1.5 s "the trained writer is just slower". One of those was right and I could not tell which.
    private(set) var readSecs = 0.0, modelSecs = 0.0, reads = 0

    func write(id: String, place: String?, taken: Date?) async throws -> (line: String, vector: [Float])? {
        guard let w = writer, let enc = encoder, let tok = tokenizer else { return nil }
        // .localOnly is the M16 read path the cluster meant (Index.swift's own pass uses it, 16.5 photos/s): it
        // takes the local rendition PhotoKit keeps for every photo and NEVER downloads an iCloud original. I had
        // .indexForeground here, which may download - so part of what I measured as "the writer" was the network.
        let r0 = Date()
        guard let img = await PhotoLibrary.ciImage(id, side: FactsWriter.side, purpose: .localOnly)
        else { return nil }
        readSecs += Date().timeIntervalSince(r0); reads += 1
        let m0 = Date()
        let raw = try await w.generate(image: img, prompt: FactsWriter.prompt, maxTokens: FactsWriter.maxTokens)
        // The 2B loops on some photos (it starts listing the same field again). FactsLine is the cluster's
        // pre-registered stop rule, v2, chosen on 70,580 public photos for the lower loop rate (.054 vs .075).
        let line = FactsLine.applyRepeatStop(raw).line
        modelSecs += Date().timeIntervalSince(m0)
        let text = factsPrefix(place: place, taken: taken) + line
        let ids = tok.factsInput(text)
        let v = try await FactsWriter.encode(ids, in: enc)
        return (line, v)
    }

    /// #writerparity (M91) only: the shipped writing path for an image that is NOT in his library - same prompt,
    /// same K, same sampling, same repeat-stop as `write`, so what the comparison measures is the MODEL on this
    /// phone against the cluster's MLX line, not a second code path that happens to look similar.
    func parityLine(image: CIImage) async throws -> String? {
        guard let w = writer else { return nil }
        let m0 = Date()
        let raw = try await w.generate(image: image, prompt: FactsWriter.prompt, maxTokens: FactsWriter.maxTokens)
        modelSecs += Date().timeIntervalSince(m0)
        return FactsLine.applyRepeatStop(raw).line
    }

    /// #factsparity (M85) only: the ids and the 384 numbers for a text the fixture supplies, by the SAME two
    /// paths the real pass uses - a facts line goes through factsInput (254 word pieces), a question through
    /// questionInput (64). The check is worthless if it encodes the text any other way than the product does,
    /// which is why this reuses `encode` rather than restating it.
    func parityVector(_ text: String, question: Bool) async throws -> (ids: [Int], v: [Float])? {
        guard let enc = encoder, let tok = tokenizer else { return nil }
        let ids = question ? tok.questionInput(text) : tok.factsInput(text)
        return (ids, try await FactsWriter.encode(ids, in: enc))
    }

    /// A search question as 384 numbers, the same encoder and pooling the lines use.
    func questionVector(_ text: String) async throws -> [Float] {
        guard let enc = encoder, let tok = tokenizer else { return [] }
        return try await FactsWriter.encode(tok.questionInput(text), in: enc)
    }

    /// Mean-pooled, unnormalised 384-d, with tokenTypeIds = zeros. The README is explicit that omitting the zeros
    /// drops cosine to .72 - a silent wrongness that would have looked like a bad model rather than a bad call.
    private static func encode(_ ids: [Int], in container: EmbedderModelContainer) async throws -> [Float] {
        try await container.perform { ctx in
            let input = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
            let types = MLXArray.zeros(like: input)
            let mask = MLXArray.ones(like: input)
            let out = ctx.model(input, positionIds: nil, tokenTypeIds: types, attentionMask: mask)
            guard let h = out.hiddenStates else { return [] }
            let pooled = h.mean(axis: 1)        // mean pool, NO normalisation (the README is explicit)
            pooled.eval()
            return pooled.asArray(Float.self)
        }
    }
}
