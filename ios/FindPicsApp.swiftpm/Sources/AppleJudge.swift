// Apple's on-device model (Foundation Models, iOS 27 image input) as the judge: no download, no app memory, but NO
// probabilities (the API returns content and token counts only; checked in Apple's docs 10-06). Two ways to get a
// number out of it, both to be measured on the phone against the Qwen judge and Reza's labels:
//   .yesNo  : a constrained "yes"/"no" answer        -> 0.98 / 0.02 (keep / drop; ranking ties)
//   .rating : a constrained integer 1...10           -> (r - 1) / 9 (a coarse score for ranking and the A-vs-B split)
// Cluster simulation of the same two modes on the server judge: FP_JUDGE_MODE=hard|rating (scripts/run_demo_mode.sh).
// UNTESTED: written from Apple's documentation before the first Xcode build; expect compile fixes.
#if canImport(FoundationModels)
import CoreImage
import Foundation
import FoundationModels

@available(iOS 27.0, *)
actor AppleJudge: PhotoJudge {
    private var cache: [String: Double] = [:]
    func cached(_ key: String) -> Double? { cache[key] }
    func remember(_ key: String, _ p: Double) { cache[key] = p }
    enum Mode { case yesNo, rating, rating100 }

    @Generable struct YesNo {
        @Guide(description: "yes or no", .anyOf(["yes", "no"])) var answer: String
    }

    @Generable struct Rating100 {
        @Guide(description: "0 = clearly no, 100 = clearly yes", .range(0...100)) var score: Int
    }

    @Generable struct Rating {
        @Guide(description: "1 = clearly no, 10 = clearly yes", .range(1...10)) var score: Int
    }

    let mode: Mode
    init(mode: Mode) { self.mode = mode }
    /// Photos Apple's safety filter refused to look at since the last takeRefused(). Reza's phone 10-07: ONE refusal
    /// ("May contain unsafe content") ended the whole "dog" search at photo 75; now a refusal is a 'no' that is counted
    /// and reported, and the search goes on.
    private var refused = 0
    func takeRefused() -> Int { defer { refused = 0 }; return refused }

    /// nil when Apple Intelligence is off or the device has no on-device model: the app falls back to the Qwen judge.
    static var unavailableReason: String? {
        switch SystemLanguageModel.default.availability {
        case .available: return nil
        case .unavailable(let why): return "\(why)"
        }
    }

    private static let ctx = CIContext()

    /// Same contract as Judge.pYes. One fresh session per photo: images use context tokens and must not accumulate.
    func pYes(_ image: CIImage, question: String) async throws -> Double {
        do { return try await answer(image, question: question) }
        catch let e as LanguageModelSession.GenerationError {
            if case .guardrailViolation = e { refused += 1; return 0 }
            throw e
        }
        catch {
            // MEASURED 10-07 on the iPhone 18 Pro: a safety refusal does NOT arrive as
            // LanguageModelSession.GenerationError.guardrailViolation. It arrives as LanguageModelError, which the
            // branch above cannot match, so the "dog" search still died at photo 75 of 16,965 with
            // "May contain unsafe content" even after the GenerationError catch was added. LanguageModelError has no
            // public case to switch on here, so the refusal is recognised by its message; anything else still throws,
            // so a real fault (context window, model unloaded) is not silently swallowed 17,000 times.
            if "\(error)".localizedCaseInsensitiveContains("unsafe content") { refused += 1; return 0 }
            throw error
        }
    }

    private func answer(_ image: CIImage, question: String) async throws -> Double {
        guard let cg = AppleJudge.ctx.createCGImage(image, from: image.extent) else { return 0.5 }
        let session = LanguageModelSession()
        let options = GenerationOptions(samplingMode: .greedy)
        switch mode {
        case .yesNo:
            let r = try await session.respond(generating: YesNo.self, options: options) {
                "Look at the photo and answer the question with yes or no. Question: \(question)"
                Attachment(cg)
            }
            return r.content.answer == "yes" ? 0.98 : 0.02
        case .rating:
            let r = try await session.respond(generating: Rating.self, options: options) {
                "Look at the photo. Question: \(question) Rate from 1 (clearly no) to 10 (clearly yes)."
                Attachment(cg)
            }
            return min(max(Double(r.content.score - 1) / 9, 0.02), 0.98)
        case .rating100:
            let r = try await session.respond(generating: Rating100.self, options: options) {
                "Look at the photo. Question: \(question) Rate from 0 (clearly no) to 100 (clearly yes)."
                Attachment(cg)
            }
            return min(max(Double(r.content.score) / 100, 0.01), 0.99)
        }
    }
}

/// Apple's on-device model as the PLANNER (text -> plan JSON). The planner prompt is ~1,840 tokens (Qwen tokenizer,
/// measured 10-06): fits a 4,096-token context with room for the answer; one fresh session per call.
@available(iOS 27.0, *)
enum AppleText {
    static func text(_ prompt: String) async throws -> String {
        let session = LanguageModelSession()
        // builder form: the documented respond(to:) takes a Prompt, and a String variable is not one
        let r = try await session.respond(options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 1024)) { prompt }
        return r.content
    }
}
#endif
