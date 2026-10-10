// Usage: GroundReplay <cases.json> <out.json>
// cases: [{conv, message, history, owner, people, today, prompt, current, raw, grounded}] from eval/eval_planner_vl.py.
// For each turn: Swift prompt == Python prompt? Swift (dropUnanswerable if needed -> ground -> keepPartialUndo) on the
// raw plan == Python grounded plan (planJSON: all fields but notes)? Writes the Swift plans for Python-side scoring.
import FindPicsCore
import Foundation

struct Case: Decodable {
    let conv: Int; let message: String; let history: [String]; let owner: String; let people: [String]
    let today: String; let prompt: String; let current: Plan?; let raw: Plan; let grounded: Plan
}
let a = CommandLine.arguments
let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: a[1])))
var out: [String] = [], badPlan = 0, badPrompt = 0
for c in cases {
    let day = Day(iso: c.today)!
    let said = (c.history + [c.message]).joined(separator: " \n ")
    let names = c.people + [c.owner]
    let pr = PlannerPrompt.build(message: c.message, history: c.history, current: c.current, owner: c.owner,
                                 people: c.people, today: day)
    if pr != c.prompt { badPrompt += 1; print("PROMPT MISMATCH conv \(c.conv): \(c.message)") }
    var p = c.raw
    if unanswerable(p, names: names, said: said) != nil { p = dropUnanswerable(p, said: said, names: names) }
    let g = keepPartialUndo(ground(p, message: c.message, history: c.history, today: day, owner: c.owner, people: c.people),
                            current: c.current, message: c.message)
    let sj = planJSON(g), pj = planJSON(c.grounded)
    if sj != pj { badPlan += 1; print("PLAN MISMATCH conv \(c.conv): \(c.message)\n  swift:  \(sj)\n  python: \(pj)") }
    out.append(sj)
}
try JSONSerialization.data(withJSONObject: out).write(to: URL(fileURLWithPath: a[2]))
print("REPLAY \(a[1]): prompt mismatches \(badPrompt) of \(cases.count); grounded-plan mismatches \(badPlan) of \(cases.count)")
