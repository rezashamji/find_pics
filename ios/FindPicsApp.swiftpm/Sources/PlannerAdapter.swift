// The distilled planner: a LoRA adapter (~58 MB, Sources/Models/planner_adapter/: adapter_config.json +
// adapters.safetensors, made by scripts/peft_to_mlx_adapter.py) on top of the SAME 4-bit Qwen3.5-4B the judge uses,
// so the phone holds one base model. 30 scripted conversations: base 4B 20/30, distilled + code rules 30/30 (16-bit
// base; the 4-bit check is eval/planners_q4merged.log). Loaded only around planner calls; judge calls see the base model.
// UNTESTED on a device (mlx-swift-lm LoRAContainer API read from its sources, 10-06).
import Foundation
import MLXLMCommon

enum PlannerAdapter {
    /// nil when the adapter files were not bundled: the planner then runs on the base model.
    static let directory: URL? = {
        guard let u = Bundle.module.url(forResource: "planner_adapter", withExtension: nil, subdirectory: "Models"),
              FileManager.default.fileExists(atPath: u.appending(component: "adapters.safetensors").path) else { return nil }
        return u
    }()

    static func container() throws -> LoRAContainer? {
        guard let d = directory else { return nil }
        return try LoRAContainer.from(directory: d)
    }
}


/// The distilled JUDGE: a LoRA (Sources/Models/judge_adapter/, from scripts/peft_to_mlx_adapter.py on
/// models/judge_4b_lora) trained to give the 9B's P(yes). Fused into the base once at load (judge calls are the
/// common case); the planner adapter is then loaded on top only around planner calls. Ship only if the tests
/// (eval/eval_judge_distill.py, everyday16_jd) show it beats the base 4B.
enum JudgeAdapter {
    static let directory: URL? = {
        guard let u = Bundle.module.url(forResource: "judge_adapter", withExtension: nil, subdirectory: "Models"),
              FileManager.default.fileExists(atPath: u.appending(component: "adapters.safetensors").path) else { return nil }
        return u
    }()
}
