// The judge: one photo + a yes/no question -> P(yes). Same scoring as the server (vlm.VLLMJudge.p_yes): next-token
// probability of "yes" vs "no" after "<question> Answer with one word: yes or no.", thinking disabled.
// Model: Qwen3.5 4-bit MLX (4B by default; 9B once memory on the 12 GB phone is measured).
import CoreImage
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
    func cached(_ key: String) -> Double?
    func remember(_ key: String, _ p: Double)
}

actor Judge: PhotoJudge {
    static let modelID = "mlx-community/Qwen3.5-4B-4bit"
    private var container: ModelContainer?
    private var yesIDs: [Int] = [], noIDs: [Int] = []
    /// Answers already given (photo + crop + question -> P(yes)): follow-ups ("only the ones outdoors") re-ask nothing.
    private var cache: [String: Double] = [:]
    func cached(_ key: String) -> Double? { cache[key] }
    func remember(_ key: String, _ p: Double) { cache[key] = p }

    func load(progress: @Sendable @escaping (Double) -> Void) async throws {
        if container != nil { return }
        let c = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(id: Judge.modelID),
            progressHandler: { p in progress(p.fractionCompleted) })
        let (y, n) = await c.perform { ctx in
            let enc = { (w: String) in ctx.tokenizer.encode(text: w, addSpecialTokens: false).first }
            return (["yes", "Yes", " yes", " Yes"].compactMap(enc), ["no", "No", " no", " No"].compactMap(enc))
        }
        yesIDs = Array(Set(y)); noIDs = Array(Set(n)); container = c
    }

    /// P(yes) for one image (photos are resized to <= 896 px on the long side like the server).
    func pYes(_ image: CIImage, question: String) async throws -> Double {
        guard let c = container else { throw NSError(domain: "Judge", code: 1) }
        let yes = yesIDs, no = noIDs
        return try await c.perform { ctx in
            var input = UserInput(chat: [.user(question + " Answer with one word: yes or no.", images: [.ciImage(image)])],
                                  additionalContext: ["enable_thinking": false])
            input.processing.resize = CGSize(width: 896, height: 896)
            let lm = try await ctx.processor.prepare(input: input)
            let cache = try ctx.model.newCache(parameters: nil)
            let logits: MLXArray
            switch try ctx.model.prepare(lm, cache: cache, state: nil, prefill: .init()) {
            case .logits(let out): logits = out.logits
            case .tokens(let toks): logits = ctx.model(toks, cache: cache, state: nil).logits
            }
            let last = logits[0, -1].asType(.float32)
            let p = softmax(last, axis: -1)
            let py = yes.map { p[$0].item(Float.self) }.reduce(0, +), pn = no.map { p[$0].item(Float.self) }.reduce(0, +)
            return Double(py / max(py + pn, 1e-9))
        }
    }

    /// Text in, text out (the planner). `adapter`: the distilled planner LoRA, loaded on top of the same base model.
    private var plannerAdapter: LoRAContainer?? = .none      // .none = not loaded yet; .some(nil) = not bundled

    func text(_ prompt: String, maxTokens: Int = 1024) async throws -> String {
        guard let c = container else { throw NSError(domain: "Judge", code: 1) }
        if plannerAdapter == nil { plannerAdapter = .some(try? PlannerAdapter.container()) }
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
