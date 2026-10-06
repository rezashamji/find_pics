// Words -> grounded plan, on the phone. Same flow as converse.plan_turn: prompt (byte-identical to the server's),
// model answer, JSON, retry with the problem fed back, then the code-enforced grounding (FindPicsCore.ground).
import FindPicsCore
import Foundation

struct Planner {
    /// Text model behind the plan: the Qwen judge's text() by default, or Apple's on-device model (AppleText.text).
    let generate: (String) async throws -> String
    init(judge: Judge) { generate = { try await judge.text($0) } }
    init(generate: @escaping (String) async throws -> String) { self.generate = generate }
    var owner = "me"
    var people: [String] = []

    func plan(_ message: String, history: [String] = [], current: Plan? = nil, today: Day) async throws -> Plan {
        let prompt = PlannerPrompt.build(message: message, history: history, current: current, owner: owner, people: people, today: today)
        let said = (history + [message]).joined(separator: " \n ")
        let names = people + [owner]
        var last: String? = nil, fallback: Plan? = nil
        for _ in 0..<3 {
            let p = last == nil ? prompt : prompt + "\n(Previous output was invalid: \(last!). Return valid JSON only.)\nJSON:"
            let out = try await generate(p)
            do {
                guard let start = out.firstIndex(of: "{"), let end = out.lastIndex(of: "}") else { throw PlanError.noJSON }
                var plan = try JSONDecoder().decode(Plan.self, from: Data(out[start...end].utf8))
                for i in plan.albums.indices where plan.albums[i].anchor != nil && (plan.albums[i].anchor!.judgeQuestion ?? "").isEmpty {
                    let lk = plan.albums[i].anchor!.looks.filter { !$0.isEmpty }
                    plan.albums[i].anchor = lk.isEmpty ? nil : Step(looks: plan.albums[i].anchor!.looks, judgeQuestion: "Does this photo show \(lk[0])?")
                }
                if plan.albums.isEmpty { throw PlanError.zeroAlbums }
                if let problem = unanswerable(plan, names: names, said: said) { fallback = plan; last = problem; continue }
                return keepPartialUndo(ground(plan, message: message, history: history, today: today, owner: owner, people: people),
                                       current: current, message: message)
            } catch {
                last = String(describing: error).prefix(300).description
            }
        }
        if let f = fallback {
            return keepPartialUndo(ground(dropUnanswerable(f, said: said, names: names), message: message, history: history, today: today,
                                          owner: owner, people: people), current: current, message: message)
        }
        throw PlanError.failed(last ?? "")
    }

    enum PlanError: Error { case noJSON, zeroAlbums, failed(String) }
}
