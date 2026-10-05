// Moments in time around an anchor photo ("the week I went to X", "the day after the wedding", "before the first
// photo of Y"): port of agent.events / agent.window_rows. Times are seconds since 1970 (UTC); nil = no date.
import Foundation

public let eventGapHours = 3.0

/// Event id per item: sort by time, a new event after a gap > 3 h. Undated items sort last and join the event before.
public func events(_ taken: [Double?], gapH: Double = eventGapHours) -> [Int] {
    let order = taken.indices.sorted { a, b in
        switch (taken[a], taken[b]) {
        case let (x?, y?): return x != y ? x < y : a < b
        case (_?, nil): return true
        case (nil, _?): return false
        default: return a < b
        }
    }
    var ev = [Int](repeating: 0, count: taken.count), cur = 0
    for (k, i) in order.enumerated() {
        if k == 0 { cur = 1 } else if let a = taken[order[k - 1]], let b = taken[i], (b - a) / 3600 > gapH { cur += 1 }
        ev[i] = cur
    }
    var next = (ev.max() ?? 0) + 1                      // undated items are not part of any event
    for i in taken.indices where taken[i] == nil { ev[i] = next; next += 1 }
    return ev
}

func utcDay(_ t: Double) -> Day { Day(serial: Int((t / 86400).rounded(.down))) }

/// ISO 8601 (year, week) of a day.
func isoWeek(_ d: Day) -> (Int, Int) {
    let wd = d.weekday                                   // Monday = 0
    let thursday = d.adding(3 - wd)                       // the week's Thursday decides its year
    let week = (thursday.serial - Day(thursday.y, 1, 1).serial) / 7 + 1
    return (thursday.y, week)
}

public func windowRows(taken: [Double?], place: [String], anchors: [Int], window: String?) -> [Int] {
    let all = Array(taken.indices)
    guard let w = window, !anchors.isEmpty else { return all }
    let anchors = w == "same_place" ? anchors : anchors.filter { taken[$0] != nil }
    if anchors.isEmpty { return [] }                    // undated anchors define no time window
    func keep(_ f: (Int) -> Bool) -> [Int] { all.filter(f) }
    switch w {
    case "same_day":
        let days = Set(anchors.compactMap { taken[$0].map(utcDay) })
        return keep { taken[$0].map { days.contains(utcDay($0)) } ?? false }
    case "same_week":
        let keys = Set(anchors.compactMap { taken[$0].map { "\(isoWeek(utcDay($0)))" } })
        return keep { taken[$0].map { keys.contains("\(isoWeek(utcDay($0)))") } ?? false }
    case "same_month", "same_year":
        func key(_ t: Double) -> String { let d = utcDay(t); return w == "same_month" ? "\(d.y)-\(d.m)" : "\(d.y)" }
        let keys = Set(anchors.compactMap { taken[$0].map(key) })
        return keep { taken[$0].map { keys.contains(key($0)) } ?? false }
    case "same_event":
        let ev = events(taken); let es = Set(anchors.map { ev[$0] })
        return keep { es.contains(ev[$0]) }
    case "before", "after":
        let ev = events(taken)
        var firsts = [Int: Double]()
        for r in anchors { if let t = taken[r] { firsts[ev[r]] = min(firsts[ev[r]] ?? t, t) } }
        return keep { i in
            guard let first = firsts[ev[i]], let t = taken[i] else { return false }
            return w == "before" ? t < first : t > first
        }
    case "since", "until":
        guard let first = anchors.compactMap({ taken[$0] }).min() else { return [] }
        return keep { i in taken[i].map { w == "since" ? $0 > first : $0 < first } ?? false }
    case "same_place":
        let ps = Set(anchors.map { place[$0] }).subtracting([""])
        return keep { ps.contains(place[$0]) }
    default:
        if let m = fullmatch(#"days_(before|after):(\d+)"#, w) {
            let n = Int(m.group(2)!)!, before = m.group(1) == "before"
            let days = Set(anchors.compactMap { taken[$0].map { utcDay($0).serial } })
            return keep { i in
                guard let t = taken[i] else { return false }
                let d = utcDay(t).serial
                return days.contains { a in before ? (d >= a - n && d <= a - 1) : (d >= a + 1 && d <= a + n) }
            }
        }
        if let m = fullmatch(#"minutes_(before|after):(\d+)"#, w) {
            let n = Double(max(Int(m.group(2)!)!, 30)) * 60, before = m.group(1) == "before"
            let ats = anchors.compactMap { taken[$0] }
            return keep { i in
                guard let t = taken[i] else { return false }
                return ats.contains { a in before ? (t >= a - n && t < a) : (t > a && t <= a + n) }
            }
        }
        return all
    }
}
