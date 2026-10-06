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
    enum Mode { case yesNo, rating }

    @Generable struct YesNo {
        @Guide(description: "yes or no", .anyOf(["yes", "no"])) var answer: String
    }

    @Generable struct Rating {
        @Guide(description: "1 = clearly no, 10 = clearly yes", .range(1...10)) var score: Int
    }

    let mode: Mode
    init(mode: Mode) { self.mode = mode }

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
        }
    }
}

/// Apple's on-device model as the PLANNER (text -> plan JSON). The planner prompt is ~1,840 tokens (Qwen tokenizer,
/// measured 10-06): fits a 4,096-token context with room for the answer; one fresh session per call.
@available(iOS 27.0, *)
enum AppleText {
    static func text(_ prompt: String) async throws -> String {
        let session = LanguageModelSession()
        let r = try await session.respond(to: prompt, options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 1024))
        return r.content
    }
}
#endif
