// The judge: one photo + a yes/no question -> P(yes). Same scoring as the server (vlm.VLLMJudge.p_yes): next-token
// probability of "yes" vs "no" after "<question> Answer with one word: yes or no.", thinking disabled.
// Model: Qwen3.5 4-bit MLX (4B by default; 9B once memory on the 12 GB phone is measured).
import CoreImage
@preconcurrency import FindPicsCore
import Foundation
import HuggingFace       // #huggingFaceLoadModelContainer expands to HuggingFace.HubClient + Tokenizers.AutoTokenizer
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers

/// Anything that answers "<question>" about one photo with a score in 0...1 (Qwen judge, Apple's model).
protocol PhotoJudge: Actor {
    func pYes(_ image: CIImage, question: String) async throws -> Double
    /// Same answer. `keepImage`: another question about this SAME image object is likely next (Search.swift's exclude /
    /// filter follow-ups), so a judge that can (Judge, RESULTS 44) keeps the photo's image prefix for it.
    func pYes(_ image: CIImage, question: String, keepImage: Bool) async throws -> Double
    func cached(_ key: String) -> Double?
    func remember(_ key: String, _ p: Double)
}

extension PhotoJudge {
    func pYes(_ image: CIImage, question: String, keepImage: Bool) async throws -> Double {
        try await pYes(image, question: question)
    }
}

/// One photo's image prefix kept between pYes calls (RESULTS 44, item a): the image object itself (held, so no new
/// image can get its address), the prompt tokens through <|vision_end|>, their KV cache and the M-RoPE state the
/// continuation needs. Its MLX arrays are only touched inside ModelContainer.perform, which runs one call at a time.
final class ImagePrefix: @unchecked Sendable {
    let image: CIImage, tokens: [Int], cache: [KVCache], state: LMOutput.State?
    init(image: CIImage, tokens: [Int], cache: [KVCache], state: LMOutput.State?) {
        self.image = image; self.tokens = tokens; self.cache = cache; self.state = state
    }
}

struct JudgeAnswer: Sendable { let p: Double; var prefix: ImagePrefix? = nil; var reused = false }

actor Judge: PhotoJudge {
    static let modelID = "mlx-community/Qwen3.5-4B-4bit"
    /// Candidate photo judge measured 10-06 on 402 eye-labeled photos: Qwen3-VL-4B kept 237/261 real with 47/141
    /// wrong at P>=0.9 vs this 4B's 232/65 at 0.5 (16-bit; phone 4-bit and real-library checks pending). Only the PHOTO
    /// judge would switch; the planner stays on modelID (+ planner adapter). Two models need the increased-memory
    /// entitlement (~2.5 + 3 GB).
    static let visionJudgeCandidateID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    let id: String
    init(modelID: String = Judge.modelID) { self.id = modelID }
    private var container: ModelContainer?
    private var yesIDs: [Int] = [], noIDs: [Int] = []
    /// <|vision_end|> (Qwen3-VL): where the image part of the prompt ends. nil = the model has none (no reuse).
    private var visionEndID: Int?
    /// The last photo's image prefix (RESULTS 44): its follow-up questions skip the vision encoder and image prefill.
    private var prefix: ImagePrefix?
    /// KV reuse on (default) / off with the launch argument -noKVReuse (A/B on the device, MAC_INBOX M39).
    static let kvReuse = !ProcessInfo.processInfo.arguments.contains("-noKVReuse")
    /// -checkKVReuse: ALSO run the old one-pass call for every reused / split answer and log both P(yes) to
    /// Documents/memcheck.txt ("KV CHECK"), so the device shows the split changes nothing but rounding.
    static let checkKVReuse = ProcessInfo.processInfo.arguments.contains("-checkKVReuse")
    /// Answers already given (photo + crop + question -> P(yes)): follow-ups ("only the ones outdoors") re-ask nothing.
    private var cache: [String: Double] = [:]
    func cached(_ key: String) -> Double? { cache[key] }
    func remember(_ key: String, _ p: Double) { cache[key] = p }

    func load(progress: @Sendable @escaping (Double) -> Void) async throws {
        if container != nil { return }
        // THE CRASH, MEASURED (M36(B), 10-10 13:17). The model loads fine - active memory sits at 2.88 GB and
        // never moves. What killed the app four photos into a search is MLX's BUFFER CACHE: 0.00 -> 0.98 -> 1.99 GB
        // while free fell 2.74 -> 0.68, until iOS jetsammed us. MLX keeps every freed buffer for reuse and will
        // happily consume all remaining memory on a device that has a hard per-app cap.
        // So: cap it. 256 MB is plenty for reuse between photos and leaves the headroom the vision encoder needs.
        // This is why the 8-bit vision checkpoint would NOT have fixed it - the weights were never the problem.
        MLX.GPU.set(cacheLimit: 256 * 1024 * 1024)
        Judge.logMem("A before load \(id)")
        // retry transient network errors (FindPicsCore.downloadRetryDelay; MAC 10-10: 16 manual restarts on -1005)
        var attempt = 0
        var loaded: ModelContainer? = nil
        while loaded == nil {
            do {
                loaded = try await #huggingFaceLoadModelContainer(
                    configuration: ModelConfiguration(id: id),
                    progressHandler: { p in progress(p.fractionCompleted) })
            } catch {
                attempt += 1
                guard let wait = downloadRetryDelay(attempt: attempt, urlErrorCode: urlErrorCode(error)) else { throw error }
                try await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000))
            }
        }
        let c = loaded!
        Judge.logMem("B weights loaded \(id)")
        let (y, n, ve) = await c.perform { ctx in
            let enc = { (w: String) in ctx.tokenizer.encode(text: w, addSpecialTokens: false).first }
            return (["yes", "Yes", " yes", " Yes"].compactMap(enc), ["no", "No", " no", " No"].compactMap(enc),
                    ctx.tokenizer.convertTokenToId("<|vision_end|>"))
        }
        yesIDs = Array(Set(y)); noIDs = Array(Set(n)); visionEndID = ve; container = c
        Judge.logMem("C container ready \(id)")
        if let d = JudgeAdapter.directory, let a = try? LoRAContainer.from(directory: d) {
            try await c.perform { ctx in try a.fuse(with: ctx.model) }      // distilled judge, permanently
        }
    }

    /// Free this model's weights. The app holds TWO ~2.9 GB models - the planner (Qwen3.5-4B, loaded at launch)
    /// and the photo judge (Qwen3-VL-4B, loaded on the first search) - and they do NOT both fit, even with the
    /// increased-memory entitlement: iOS killed the app the first time Reza typed a real query (10-10 10:08).
    /// Reloading afterwards reads from disk and needs no download.
    /// Are this model's weights resident? The app can hold only ONE of the two ~2.9 GB models at a time
    /// (planner 2.83 GB + judge 2.88 GB = 5.71 GB against ~5.8 GB free), so searches swap them.
    var isLoaded: Bool { container != nil }

    func unload() {
        container = nil; prefix = nil
        yesIDs = []; noIDs = []
        // Dropping the Swift reference is NOT enough: MLX keeps freed buffers in its own cache and the memory
        // never goes back to the system, so the next model load still hits the app's limit and iOS kills it.
        // (MEASURED 10-10: freeing the planner alone did not stop the crash; MLX.GPU.cacheMemory is the reason.)
        MLX.GPU.clearCache()
    }

    /// M36(B): append one memory line to Documents/memcheck.txt. On DISK because the failure is a hard kill by
    /// iOS - anything held only in memory or shown on screen dies with the app, which is why three crashes told
    /// us nothing (10-10). The LAST line in the file is the step that was running when it died.
    static func logMem(_ label: String) {
        let gb = { (b: Int) in String(format: "%.2f", Double(b) / 1_073_741_824) }
        let line = "\(Date().formatted(date: .omitted, time: .standard)) \(label): free "
                 + String(format: "%.2f", availableGB) + " GB | MLX active \(gb(MLX.GPU.activeMemory))"
                 + " cache \(gb(MLX.GPU.cacheMemory)) peak \(gb(MLX.GPU.peakMemory))\n"
        guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let f = dir.appendingPathComponent("memcheck.txt")
        if let h = try? FileHandle(forWritingTo: f) { h.seekToEndOfFile(); h.write(Data(line.utf8)); try? h.close() }
        else { try? line.write(to: f, atomically: true, encoding: .utf8) }
    }

    /// What MLX is actually holding, for the on-screen note - so a memory failure is a number, not a guess.
    static func memoryNote() -> String {
        let gb = { (b: Int) in String(format: "%.2f", Double(b) / 1_073_741_824) }
        return "app may use \(String(format: "%.2f", availableGB)) GB; MLX active \(gb(MLX.GPU.activeMemory))"
             + " GB, cached \(gb(MLX.GPU.cacheMemory)) GB"
    }

    /// Memory this app may still use, in GB.
    static var availableGB: Double { Double(os_proc_available_memory()) / 1_073_741_824 }

    /// P(yes) for one image (photos are resized to <= 896 px on the long side like the server).
    func pYes(_ image: CIImage, question: String) async throws -> Double {
        try await pYes(image, question: question, keepImage: false)
    }

    /// KV REUSE (RESULTS 44, item a). The prompt is [header, <|vision_start|>, ~588 image tokens, <|vision_end|>]
    /// + [question tail, ~30 tokens]; the first part depends only on the photo. With `keepImage` the first question
    /// prefills that part once (vision encoder + image tokens) into a KV cache that is kept; a later question on the
    /// SAME image object appends only its tail to it (Qwen3VL.prepare's warm continuation: positions resume at the
    /// cache offset + the image's M-RoPE delta, carried in the returned state) and trims the cache back after. Used only
    /// when every layer is a plain KVCacheSimple (Qwen3-VL; the hybrid Qwen3.5 layers cannot be trimmed back) and the
    /// new prompt starts with exactly the kept tokens; anything else is the old one-pass call.
    func pYes(_ image: CIImage, question: String, keepImage: Bool) async throws -> Double {
        guard let c = container else { throw NSError(domain: "Judge", code: 1) }
        let yes = yesIDs, no = noIDs, visionEnd = visionEndID
        if let old = prefix, old.image !== image { prefix = nil }     // another photo: free its KV before this prefill
        let held = Judge.kvReuse ? prefix : nil
        let split = Judge.kvReuse && keepImage, check = Judge.checkKVReuse
        let ans: JudgeAnswer = try await c.perform { ctx in
            var input = UserInput(chat: [.user(question + " Answer with one word: yes or no.", images: [.ciImage(image)])],
                                  additionalContext: ["enable_thinking": false])
            input.processing.resize = CGSize(width: 896, height: 896)
            Judge.logMem("D before vision encode of a 896 px photo")
            let lm = try await ctx.processor.prepare(input: input)
            Judge.logMem("E after vision encode")
            func pOf(_ logits: MLXArray) -> Double {
                let last = logits[0, -1].asType(.float32)
                let p = softmax(last, axis: -1)
                let py = yes.map { p[$0].item(Float.self) }.reduce(0, +), pn = no.map { p[$0].item(Float.self) }.reduce(0, +)
                return Double(py / max(py + pn, 1e-9))
            }
            func onePass() throws -> Double {                        // the call as it always was
                let cache = try ctx.model.newCache(parameters: nil)
                switch try ctx.model.prepare(lm, cache: cache, state: nil, prefill: .init()) {
                case .logits(let out): return pOf(out.logits)
                case .tokens(let toks): return pOf(ctx.model(toks, cache: cache, state: nil).logits)
                }
            }
            /// One question's tail on top of a kept image prefix; the cache is trimmed back to the prefix afterwards.
            func ask(_ pre: ImagePrefix, _ tail: ArraySlice<Int>) throws -> Double {
                let n = pre.tokens.count
                defer { for layer in pre.cache { _ = layer.trim(layer.offset - n) } }
                let t = LMInput(tokens: MLXArray(tail.map { Int32($0) }).expandedDimensions(axis: 0))
                guard case .logits(let out) = try ctx.model.prepare(t, cache: pre.cache, state: pre.state, prefill: .init())
                else { throw NSError(domain: "Judge", code: 2) }
                return pOf(out.logits)
            }
            func logCheck(_ what: String, _ p: Double) throws {
                let p1 = try onePass()
                Judge.logMem("KV CHECK \(what): split \(String(format: "%.5f", p)) one-pass \(String(format: "%.5f", p1)) "
                             + "diff \(String(format: "%.5f", abs(p - p1)))")
            }
            guard held != nil || split, let ve = visionEnd else { return JudgeAnswer(p: try onePass()) }
            let toks = lm.text.tokens.asArray(Int.self)
            if let pre = held, let tail = tailAfterPrefix(toks, prefix: pre.tokens) {          // a follow-up: reuse
                let p = try ask(pre, tail)
                if check { try logCheck("reused", p) }
                return JudgeAnswer(p: p, prefix: pre, reused: true)
            }
            guard split, lm.image != nil, let n = imagePrefixLength(toks, visionEnd: ve) else { return JudgeAnswer(p: try onePass()) }
            let cache = try ctx.model.newCache(parameters: nil)
            guard cache.allSatisfy({ type(of: $0) == KVCacheSimple.self }) else { return JudgeAnswer(p: try onePass()) }
            let head = LMInput(text: .init(tokens: lm.text.tokens[0..., ..<n], mask: lm.text.mask?[0..., ..<n]), image: lm.image)
            guard case .logits(let out) = try ctx.model.prepare(head, cache: cache, state: nil, prefill: .init())
            else { return JudgeAnswer(p: try onePass()) }
            eval(cache)          // the image prefill; its logits are never read, so no lm_head runs over the ~590 tokens
            let pre = ImagePrefix(image: image, tokens: Array(toks[..<n]), cache: cache, state: out.state)
            let p = try ask(pre, toks[n...])
            if check { try logCheck("first", p) }
            return JudgeAnswer(p: p, prefix: pre)
        }
        // keep the prefix only while more questions about this image may come; otherwise free its ~0.1 GB now
        prefix = keepImage && Judge.kvReuse ? ans.prefix : nil
        return ans.p
    }

    /// Text in, text out (the planner). `adapter`: the distilled planner LoRA, loaded on top of the same base model.
    private var plannerAdapter: LoRAContainer?? = .none      // .none = not loaded yet; .some(nil) = not bundled

    func text(_ prompt: String, maxTokens: Int = 1024) async throws -> String {
        guard let c = container else { throw NSError(domain: "Judge", code: 1) }
        if plannerAdapter == nil {
            plannerAdapter = .some(try? PlannerAdapter.container())
            // SAY WHETHER THE TRAINED PLANNER IS ACTUALLY IN USE. Until 10-12 22:11 the adapter was not in the repo
            // at all and this path returned nil in silence, so every device measurement we ever made was the BASE
            // model while we believed otherwise. A missing file must never again be indistinguishable from a
            // working one.
            Judge.logMem("PLANNER adapter " + ((plannerAdapter ?? nil) != nil
                ? "loaded from \(PlannerAdapter.directory?.lastPathComponent ?? "?")" : "NOT FOUND - using the base model"))
        }
        let adapter = plannerAdapter ?? nil
        return try await c.perform { ctx in
            // the distilled planner: adapter on for this call only, so judge calls keep seeing the base model
            if let a = adapter { try a.load(into: ctx.model) }
            defer { if let a = adapter { a.unload(from: ctx.model) } }
            let input = UserInput(chat: [.user(prompt)], additionalContext: ["enable_thinking": false])
            let lm = try await ctx.processor.prepare(input: input)
            var out = ""
            let params = GenerateParameters(maxTokens: maxTokens, temperature: 0)
            for await piece in try MLXLMCommon.generate(input: lm, parameters: params, context: ctx) {
                if case .chunk(let s) = piece { out += s }
            }
            return out
        }
    }
}

/// Photo judge = mean P(yes) of two models the phone loads anyway (Qwen3-VL-4B + the Qwen3.5-4B planner model), as a
/// cascade: only photos the first model does not clearly reject (P >= 0.4) get the second opinion. Port of
/// vlm.EnsembleJudge; eye labels 10-06: 231/261 real kept with 34/141 wrong vs 208/33 for Qwen3.5-4B alone.
actor EnsembleJudge: PhotoJudge {
    let first: any PhotoJudge, second: any PhotoJudge
    let gate: Double
    private var cache: [String: Double] = [:]
    init(first: any PhotoJudge, second: any PhotoJudge, gate: Double = 0.4) { self.first = first; self.second = second; self.gate = gate }
    func cached(_ key: String) -> Double? { cache[key] }
    func remember(_ key: String, _ p: Double) { cache[key] = p }
    func pYes(_ image: CIImage, question: String) async throws -> Double {
        try await pYes(image, question: question, keepImage: false)
    }
    func pYes(_ image: CIImage, question: String, keepImage: Bool) async throws -> Double {
        let a = try await first.pYes(image, question: question, keepImage: keepImage)
        if a < gate { return a / 2 }
        let b = try await second.pYes(image, question: question, keepImage: keepImage)
        return (a + b) / 2
    }
}


/// The NSURLError code inside an error (URLError, an NSURLErrorDomain NSError, or one wrapped as the underlying error).
func urlErrorCode(_ e: Error) -> Int? {
    if let u = e as? URLError { return u.code.rawValue }
    let ns = e as NSError
    if ns.domain == NSURLErrorDomain { return ns.code }
    if let inner = ns.userInfo[NSUnderlyingErrorKey] as? Error { return urlErrorCode(inner) }
    return nil
}
