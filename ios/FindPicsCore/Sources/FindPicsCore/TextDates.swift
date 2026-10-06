// Dates and clock times from the person's words, computed in code (never by the model): port of
// converse.resolve_time_of_day and converse.resolve_relative. Checked against the Python on fuzz-corpus phrases.
import Foundation

// MARK: - small helpers

/// A calendar day (proleptic Gregorian), independent of time zones and Foundation calendars.
public struct Day: Comparable, Hashable, CustomStringConvertible, Sendable {
    public let y: Int, m: Int, d: Int
    public init(_ y: Int, _ m: Int, _ d: Int) { self.y = y; self.m = m; self.d = d }
    public init?(iso: String) {
        let p = iso.split(separator: "-").compactMap { Int($0) }
        guard p.count == 3 else { return nil }
        self.init(p[0], p[1], p[2])
    }
    /// days since 1970-01-01 (Howard Hinnant's algorithm)
    var serial: Int {
        let yy = m <= 2 ? y - 1 : y
        let era = (yy >= 0 ? yy : yy - 399) / 400
        let yoe = yy - era * 400
        let doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy
        return era * 146097 + doe - 719468
    }
    init(serial z0: Int) {
        let z = z0 + 719468
        let era = (z >= 0 ? z : z - 146096) / 146097
        let doe = z - era * 146097
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100)
        let mp = (5 * doy + 2) / 153
        let d = doy - (153 * mp + 2) / 5 + 1
        let m = mp < 10 ? mp + 3 : mp - 9
        self.init(yoe + era * 400 + (m <= 2 ? 1 : 0), m, d)
    }
    public func adding(_ days: Int) -> Day { Day(serial: serial + days) }
    /// Monday = 0 ... Sunday = 6 (Python's date.weekday())
    public var weekday: Int { pymod(serial + 3, 7) }
    public var description: String { String(format: "%04d-%02d-%02d", y, m, d) }
    public static func < (a: Day, b: Day) -> Bool { a.serial < b.serial }
    static func daysIn(_ y: Int, _ m: Int) -> Int { Day(y + (m == 12 ? 1 : 0), m % 12 + 1, 1).serial - Day(y, m, 1).serial }
}

/// Python's % (result has the sign of the divisor)
func pymod(_ a: Int, _ n: Int) -> Int { ((a % n) + n) % n }

/// Python divmod for positive divisor
func pydivmod(_ a: Int, _ n: Int) -> (Int, Int) { let r = pymod(a, n); return ((a - r) / n, r) }

/// A regex match: group(i) as String? (nil when the group did not take part), like Python's re.Match.
struct Match {
    let groups: [String?]
    func group(_ i: Int) -> String? { i < groups.count ? groups[i] : nil }
}

func regex(_ pattern: String) -> NSRegularExpression {
    // compiled once per pattern string
    if let r = regexCache[pattern] { return r }
    let r = try! NSRegularExpression(pattern: pattern, options: [])
    regexCache[pattern] = r
    return r
}
nonisolated(unsafe) var regexCache = [String: NSRegularExpression]()

func search(_ pattern: String, _ s: String) -> Match? {
    let ns = s as NSString
    guard let m = regex(pattern).firstMatch(in: s, range: NSRange(location: 0, length: ns.length)) else { return nil }
    return Match(groups: (0..<m.numberOfRanges).map { m.range(at: $0).location == NSNotFound ? nil : ns.substring(with: m.range(at: $0)) })
}

func fullmatch(_ pattern: String, _ s: String) -> Match? { search("^(?:" + pattern + ")$", s) }

func sub(_ pattern: String, _ template: String, _ s: String) -> String {
    let ns = s as NSString
    return regex(pattern).stringByReplacingMatches(in: s, range: NSRange(location: 0, length: ns.length), withTemplate: template)
}

/// planner._norm: lowercase, every run of non-[a-z0-9] -> one space, trimmed
public func normText(_ x: String) -> String {
    sub("[^a-z0-9]+", " ", x.lowercased()).trimmingCharacters(in: .whitespaces)
}

func hm(_ h: Int, _ m: Int = 0) -> String { String(format: "%02d:%02d", pymod(h, 24), m) }

// MARK: - time of day

/// "between 8 and 11pm" -> 20:00-23:00, "after 10pm" -> 22:00-04:00, "at night" -> 20:00-04:00. nil if no clock time.
public func resolveTimeOfDay(_ text: String) -> String? {
    let t = text.lowercased()
    let num = #"(\d{1,2})(?::(\d\d))?\s*(am|pm|a\.m\.|p\.m\.)?"#
    func to24(_ h: Int, _ ap: String?) -> Int { h % 12 + ((ap ?? "").hasPrefix("p") ? 12 : 0) }
    if let m = search(#"\b(?:between|from)\s+"# + num + #"\s*(?:and|to|-|till|until|til)\s*"# + num, t),
       m.group(3) != nil || m.group(6) != nil {
        let h1 = Int(m.group(1)!)!, m1 = Int(m.group(2) ?? "0")!, ap1 = m.group(3)
        let h2 = Int(m.group(4)!)!, m2 = Int(m.group(5) ?? "0")!, ap2 = m.group(6)
        let e = to24(h2, ap2 ?? ap1)
        let s0: Int
        if ap1 != nil { s0 = to24(h1, ap1) } else {
            let cands = [to24(h1, "am"), to24(h1, "pm")]
            func key(_ x: Int) -> Int { let k = pymod(e - x, 24); return k == 0 ? 24 : k }
            s0 = key(cands[1]) < key(cands[0]) ? cands[1] : cands[0]      // Python min keeps the first on ties
        }
        return "\(hm(s0, m1))-\(hm(e, m2))"
    }
    if let m = search(#"\b(after|past|later than)\s+"# + num, t), let ap = m.group(4) {
        let h = Int(m.group(2)!)! % 12 + (ap.hasPrefix("p") ? 12 : 0)
        return "\(hm(h, Int(m.group(3) ?? "0")!))-\(h >= 12 || h < 4 ? hm(4) : hm(12))"
    }
    if let m = search(#"\b(before|earlier than)\s+"# + num, t), let ap = m.group(4) {
        let h = Int(m.group(2)!)! % 12 + (ap.hasPrefix("p") ? 12 : 0)
        return "\(h > 4 ? hm(4) : hm(0))-\(hm(h, Int(m.group(3) ?? "0")!))"
    }
    if let m = search(#"\b(?:at|around|about|near)\s+"# + num, t), let ap = m.group(3) {
        let h = Int(m.group(1)!)! % 12 + (ap.hasPrefix("p") ? 12 : 0), mm = Int(m.group(2) ?? "0")!
        return "\(hm(h - 1, mm))-\(hm(h + 1, mm))"
    }
    if search(#"\bmidnight\b"#, t) != nil { return "23:00-01:00" }
    if search(#"\b(noon|midday|lunch ?time)\b"#, t) != nil { return "11:30-14:00" }
    if search(#"\b(late at night|late night|middle of the night)\b"#, t) != nil { return "23:00-04:00" }
    if search(#"\b(at night|night ?time|nighttime|in the night)\b"#, t) != nil { return "20:00-04:00" }
    if search(#"\b(in the (early )?morning|this morning|morning|mornings)\b"#, t) != nil,
       search(#"\bmorning (after|of)\b"#, t) == nil { return "05:00-12:00" }
    if search(#"\b(in the afternoon|this afternoon|afternoon|afternoons)\b"#, t) != nil { return "12:00-17:00" }
    if search(#"\b(in the evening|this evening|evening|evenings)\b"#, t) != nil { return "17:00-21:00" }
    return nil
}

// MARK: - relative dates

let monthNum: [String: Int] = {
    var d = [String: Int]()
    let names = "january february march april may june july august september october november december jan feb mar apr may jun jul aug sep oct nov dec".split(separator: " ")
    for (i, m) in names.enumerated() { d[String(m)] = i % 12 + 1 }
    d["sept"] = 9
    return d
}()
let monthNames = monthNum.keys.sorted { $0.count != $1.count ? $0.count > $1.count : $0 < $1 }
let holidays: [(String, (Int, Int))] = [
    ("christmas eve", (12, 24)), ("christmas|xmas", (12, 25)), ("new year s eve|nye", (12, 31)),
    ("new year s day|new year s|new years|new year", (1, 1)), ("(?:fourth|4th) of july|july (?:4th|fourth|4)", (7, 4)),
    ("halloween", (10, 31)), ("valentine s day|valentines day|valentine s", (2, 14)), ("thanksgiving", (11, 0))]
let dayAbbr: [String: Int] = {
    var d = [String: Int]()
    for (i, n) in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].enumerated() { d[n] = i }
    for (k, v) in ["mon": 0, "tue": 1, "tues": 1, "wed": 2, "weds": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6] { d[k] = v }
    return d
}()
let numWords: [String: Int] = {
    var d = [String: Int]()
    for (i, w) in "zero one two three four five six seven eight nine ten eleven twelve".split(separator: " ").enumerated() { d[String(w)] = i }
    return d
}()
// Python dicts iterate in insertion order; the alternation order matters only for equal-prefix words, so keep it
let numAlt = "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"

/// (date_from, date_to exclusive) for simple relative phrases; nil = not one of these forms (planner's dates are kept).
public func resolveRelative(_ phrase: String, today: Day) -> (String, String)? {
    var t = sub(#"^(?:(?:from|in|on|during|taken|over|within|for|of|since|at)\s+)*(?:the\s+)?"#, "", normText(phrase))
    t = sub(#"\s+(?:only|too)$"#, "", t)
    func iso(_ a: Day, _ b: Day) -> (String, String) { (a.description, b.description) }
    func monthStart(_ y: Int, _ m: Int) -> Day { Day(y, m, 1) }
    func nextMonth(_ y: Int, _ m: Int) -> Day { Day(y + (m == 12 ? 1 : 0), m % 12 + 1, 1) }
    if ["last", "latest", "most recent", "recent", "newest", "last one"].contains(t) { return nil }
    let mn = monthNames.joined(separator: "|")
    if let mo = fullmatch("(?:last |this past |past |this )?(" + mn + ")(?: (\\d{4}))?", t) {
        let mi = monthNum[mo.group(1)!]!
        var y = mo.group(2).map { Int($0)! } ?? (mi <= today.m ? today.y : today.y - 1)
        if mo.group(2) == nil && t.hasPrefix("last ") && mi == today.m { y -= 1 }
        return iso(monthStart(y, mi), nextMonth(y, mi))
    }
    if let r = fullmatch("(?:between )?(" + mn + ")(?: (\\d{4}))? (?:to|till|til|until|through|thru|and|-) (" + mn +
                         ")(?: (\\d{4}))?(?: (?:of )?(last|this) year)?", t) {
        let m1 = monthNum[r.group(1)!]!, m2 = monthNum[r.group(3)!]!
        var y2 = Int(r.group(4) ?? r.group(2) ?? "0")!
        if y2 == 0 { y2 = r.group(5) == "last" ? today.y - 1 : (r.group(5) == "this" ? today.y : today.y - (m2 > today.m ? 1 : 0)) }
        let y1 = r.group(2).map { Int($0)! } ?? (m1 <= m2 ? y2 : y2 - 1)
        return iso(monthStart(y1, m1), nextMonth(y2, m2))
    }
    if let wk = fullmatch("week (?:of|around) (.+)", t), let r0 = resolveRelative(wk.group(1)!, today: today) {
        let d0 = Day(iso: r0.0)!; let m0 = d0.adding(-d0.weekday)
        return iso(m0, m0.adding(7))
    }
    for (names, (hmo, hd)) in holidays {
        if fullmatch("(?:last |this past |this )?(?:" + names + ")(?: (?:morning|day|eve|night|party|dinner|celebration|" +
                     "fireworks|holidays?))*", t) != nil {
            let y = (hmo, hd) <= (today.m, today.d) ? today.y : today.y - 1
            if hmo == 11 && hd == 0 {
                var d0 = Day(y, 11, 1); var th = d0.adding(pymod(3 - d0.weekday, 7) + 21)
                if th > today { d0 = Day(y - 1, 11, 1); th = d0.adding(pymod(3 - d0.weekday, 7) + 21) }
                return iso(th, th.adding(1))
            }
            return iso(Day(y, hmo, hd), Day(y, hmo, hd).adding(1))
        }
    }
    if let ago = fullmatch("(\\d+|" + numAlt + "|a) (day|week|month|year)s? ago", t) {
        let g1 = ago.group(1)!
        let n = Int(g1) ?? numWords[g1] ?? 1
        switch ago.group(2)! {
        case "day": return iso(today.adding(-n), today.adding(-n + 1))
        case "week": let m0 = today.adding(-(today.weekday + 7 * n)); return iso(m0, m0.adding(7))
        case "month":
            let (y, m_) = pydivmod(today.y * 12 + today.m - 1 - n, 12)
            return iso(Day(y, m_ + 1, 1), Day(y + (m_ == 11 ? 1 : 0), (m_ + 1) % 12 + 1, 1))
        default: return iso(Day(today.y - n, 1, 1), Day(today.y - n + 1, 1, 1))
        }
    }
    if let se = fullmatch("this (summer|spring|fall|autumn)", t) {
        let m1 = ["spring": 3, "summer": 6, "fall": 9, "autumn": 9][se.group(1)!]!
        let y = (m1, 1) <= (today.m, today.d) ? today.y : today.y - 1
        return iso(Day(y, m1, 1), Day(y, m1 + 3, 1))
    }
    if ["last quarter", "previous quarter", "this quarter"].contains(t) {
        let q0 = Day(today.y, 3 * ((today.m - 1) / 3) + 1, 1)
        if t == "this quarter" { return iso(q0, today.adding(1)) }
        let (y, m_) = q0.m > 3 ? (q0.y, q0.m - 3) : (q0.y - 1, 10)
        return iso(Day(y, m_, 1), q0)
    }
    if ["today", "this morning", "this afternoon", "this evening", "tonight", "earlier today"].contains(t) {
        return iso(today, today.adding(1))
    }
    if ["yesterday", "last night", "yesterday morning", "yesterday afternoon", "yesterday evening"].contains(t) {
        return iso(today.adding(-1), today)
    }
    if ["last weekend", "past weekend"].contains(t) {
        let back = pymod(today.weekday - 5, 7); var sat = today.adding(-(back == 0 ? 7 : back))
        if sat.adding(1) >= today { sat = sat.adding(-7) }
        return iso(sat, sat.adding(2))
    }
    if ["weekend before last", "the weekend before last"].contains(t) {
        let (a, b) = resolveRelative("last weekend", today: today)!
        return iso(Day(iso: a)!.adding(-7), Day(iso: b)!.adding(-7))
    }
    if t == "this weekend" {
        let sat = today.weekday < 5 ? today.adding(pymod(5 - today.weekday, 7)) : today.adding(-(today.weekday - 5))
        return iso(sat, sat.adding(2))
    }
    let dayAlt = dayAbbr.keys.sorted { $0.count != $1.count ? $0.count > $1.count : $0 < $1 }.joined(separator: "|")
    if let m = fullmatch("(?:last |past |this past |on )?(" + dayAlt + ")(?: (?:morning|afternoon|evening|night))?", t) {
        let k = pymod(today.weekday - dayAbbr[m.group(1)!]!, 7), back = k == 0 ? 7 : k
        return iso(today.adding(-back), today.adding(-back + 1))
    }
    let mon = today.adding(-today.weekday)
    if t == "this week" { return iso(mon, today.adding(1)) }
    if t == "week before last" { return iso(mon.adding(-14), mon.adding(-7)) }
    if ["last week", "past week"].contains(t) { return iso(mon.adding(-7), today.adding(1)) }
    let first = Day(today.y, today.m, 1)
    if t == "this month" { return iso(first, today.adding(1)) }
    if t == "last month" { let p = first.adding(-1); return iso(Day(p.y, p.m, 1), first) }
    if t == "this year" { return iso(Day(today.y, 1, 1), today.adding(1)) }
    if t == "last year" { return iso(Day(today.y - 1, 1, 1), Day(today.y, 1, 1)) }
    let m = fullmatch("(?:last|past|previous) (\\d+|" + numAlt + "|a|one) ?(day|week|month|year)s?", t)
        ?? fullmatch("(?:last|past|previous) ()(day|week|month|year)", t)
    if let m = m, let unit = m.group(2), !(m.group(1) == "" && ["week", "month", "year"].contains(unit)) {
        let g1 = m.group(1) ?? ""
        let n = Int(g1) ?? numWords[g1] ?? 1
        let start: Day
        if unit == "day" { start = today.adding(-n) } else if unit == "week" { start = today.adding(-7 * n) } else {
            let k = n * (unit == "year" ? 12 : 1)
            let (y, mo0) = pydivmod(today.y * 12 + today.m - 1 - k, 12); let mo = mo0 + 1
            start = Day(y, mo, min(today.d, Day.daysIn(y, mo)))
        }
        return iso(start, today.adding(1))
    }
    return nil
}
