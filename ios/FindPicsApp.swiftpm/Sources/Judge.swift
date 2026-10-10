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
    func cached(_ key: String) -> Double?
    func remember(_ key: String, _ p: Double)
}

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
        let (y, n) = await c.perform { ctx in
            let enc = { (w: String) in ctx.tokenizer.encode(text: w, addSpecialTokens: false).first }
            return (["yes", "Yes", " yes", " Yes"].compactMap(enc), ["no", "No", " no", " No"].compactMap(enc))
        }
        yesIDs = Array(Set(y)); noIDs = Array(Set(n)); container = c
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
        container = nil
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
        guard let c = container else { throw NSError(domain: "Judge", code: 1) }
        let yes = yesIDs, no = noIDs
        return try await c.perform { ctx in
            var input = UserInput(chat: [.user(question + " Answer with one word: yes or no.", images: [.ciImage(image)])],
                                  additionalContext: ["enable_thinking": false])
            input.processing.resize = CGSize(width: 896, height: 896)
            Judge.logMem("D before vision encode of a 896 px photo")
            let lm = try await ctx.processor.prepare(input: input)
            Judge.logMem("E after vision encode")
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
        let a = try await first.pYes(image, question: question)
        if a < gate { return a / 2 }
        let b = try await second.pYes(image, question: question)
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
