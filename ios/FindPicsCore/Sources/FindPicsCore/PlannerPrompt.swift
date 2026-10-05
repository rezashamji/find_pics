// The planner prompt, byte-identical to converse.build_prompt (the distilled phone planner was trained on these exact
// prompts). Template text: Resources/planner_template.txt (exported from converse.TEMPLATE).
import Foundation

func jsonString(_ s: String) -> String {          // pydantic / serde_json style: UTF-8 kept, only required escapes
    var o = "\""
    for u in s.unicodeScalars {
        switch u {
        case "\"": o += "\\\""
        case "\\": o += "\\\\"
        case "\n": o += "\\n"
        case "\r": o += "\\r"
        case "\t": o += "\\t"
        case "\u{08}": o += "\\b"
        case "\u{0C}": o += "\\f"
        default:
            if u.value < 0x20 { o += String(format: "\\u%04x", u.value) } else { o.unicodeScalars.append(u) }
        }
    }
    return o + "\""
}
func jsonOpt(_ s: String?) -> String { s.map(jsonString) ?? "null" }
func jsonList(_ xs: [String]) -> String { "[" + xs.map(jsonString).joined(separator: ",") + "]" }
func jsonStep(_ s: Step?) -> String {
    guard let s = s else { return "null" }
    return "{\"looks\":\(jsonList(s.looks)),\"judge_question\":\(jsonOpt(s.judgeQuestion))}"
}

/// Plan as pydantic's model_dump_json(exclude={"notes"}): field order as declared, compact.
public func planJSON(_ p: Plan) -> String {
    let albums = p.albums.map { a -> String in
        let fields: [String] = [
            "\"name\":" + jsonString(a.name), "\"person\":" + jsonOpt(a.person), "\"looks\":" + jsonList(a.looks),
            "\"avoid\":" + jsonList(a.avoid), "\"judge_question\":" + jsonOpt(a.judgeQuestion),
            "\"date_from\":" + jsonOpt(a.dateFrom), "\"date_to\":" + jsonOpt(a.dateTo),
            "\"time_phrase\":" + jsonOpt(a.timePhrase), "\"place\":" + jsonOpt(a.place),
            "\"time_of_day\":" + jsonOpt(a.timeOfDay), "\"media\":" + jsonString(a.media), "\"want\":" + jsonString(a.want),
            "\"max_items\":" + (a.maxItems.map { String($0) } ?? "null"), "\"anchor\":" + jsonStep(a.anchor),
            "\"window\":" + jsonOpt(a.window), "\"exclude_question\":" + jsonOpt(a.excludeQuestion),
            "\"filter_question\":" + jsonOpt(a.filterQuestion), "\"with_people\":" + jsonList(a.withPeople),
            "\"until\":" + jsonStep(a.until),
        ]
        return "{" + fields.joined(separator: ",") + "}"
    }
    return "{\"albums\":[\(albums.joined(separator: ","))],\"unknown_people\":\(jsonList(p.unknownPeople))}"
}

public enum PlannerPrompt {
    static let template: String = {
        let url = Bundle.module.url(forResource: "planner_template", withExtension: "txt")!
        return try! String(contentsOf: url, encoding: .utf8)
    }()

    /// Python str.format on the template: {name} -> value, {{ -> {, }} -> }.
    static func format(_ t: String, _ values: [String: String]) -> String {
        var out = "", i = t.startIndex
        while i < t.endIndex {
            let c = t[i], n = t.index(after: i)
            if c == "{", n < t.endIndex, t[n] == "{" { out += "{"; i = t.index(after: n); continue }
            if c == "}", n < t.endIndex, t[n] == "}" { out += "}"; i = t.index(after: n); continue }
            if c == "{", let close = t[n...].firstIndex(of: "}") {
                out += values[String(t[n..<close])] ?? ""; i = t.index(after: close); continue
            }
            out.append(c); i = n
        }
        return out
    }

    public static func build(message: String, history: [String], current: Plan?, owner: String = "me",
                             people: [String] = [], today: Day) -> String {
        let msg = message.trimmingCharacters(in: .whitespacesAndNewlines)
        let conv: String
        if let cur = current {
            let rules: [String] = [
                "Return the WHOLE updated plan. A follow-up EDITS the existing album(s); keep everything it does not change.\n",
                "- \"only ...\" narrows them: dates/place/media if it is about those, otherwise filter_question.\n",
                "- \"also ...\" widens them (e.g. \"also videos\" -> media any; \"also 2019\" -> widen the dates).\n",
                "- \"drop/remove/without ...\" -> exclude_question.\n",
                "- Undoing PART of an earlier change edits that field and keeps the rest: exclude_question \"sandwich or ",
                "burger?\" + \"actually keep the sandwiches\" -> exclude_question \"burger?\".\n",
                "- \"he/she/her/him/it/them\" refers to the subject of the current album(s), never a new person.\n",
                "- Add a new album ONLY if the message clearly asks for a separate, additional group.\n",
                "- Change only the album(s) the message is about; leave the others exactly as they are.",
            ]
            let earlier: String = history.map { "- " + $0 }.joined(separator: "\n")
            let parts: [String] = ["\nThis is a follow-up. Earlier messages:\n", earlier, "\nCurrent plan:\n", planJSON(cur), "\n",
                                   rules.joined(), "\nNew message: ", msg]
            conv = parts.joined()
        } else {
            conv = "\nRequest: \(msg)"
        }
        let days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return format(template, ["today": today.description, "weekday": days[today.weekday], "owner": owner,
                                 "people": people.isEmpty ? "unknown" : people.joined(separator: ", "), "conversation": conv])
    }
}
