// Fast stage of a search: which photos are in scope (media / dates / clock / place) and how well each matches the
// words, from image vectors computed once at indexing. Port of engine.scope_mask / time_of_day_mask / look_scores.
import Foundation

public struct LibraryItem {
    public var id: String
    public var media: String            // "photo" | "video"
    public var taken: Double?            // seconds since 1970, UTC
    public var localMinutes: Int?        // wall-clock minutes after midnight where it was taken; nil = unknown
    public var place: String?            // place name text (offline-geocoded GPS)
    public var camera: String            // "front" | "back" (EXIF lens model) | "" unknown
    public init(id: String, media: String, taken: Double?, localMinutes: Int?, place: String?, camera: String = "") {
        self.id = id; self.media = media; self.taken = taken; self.localMinutes = localMinutes; self.place = place
        self.camera = camera
    }
}

func utcSeconds(_ iso: String) -> Double? { Day(iso: iso).map { Double($0.serial) * 86400 } }

/// '20:00-23:00' or wrapping '22:00-04:00'; unknown clocks are OUT (never guessed).
public func inTimeOfDay(_ minutes: Int?, _ range: String) -> Bool {
    guard let mins = minutes else { return false }
    let p = range.split(separator: "-").map { s -> Int in let c = s.split(separator: ":"); return Int(c[0])! * 60 + Int(c[1])! }
    let (a, b) = (p[0], p[1])
    return a < b ? (mins >= a && mins < b) : (mins >= a || mins < b)
}

public func scopeMask(_ items: [LibraryItem], _ album: Album) -> [Bool] {
    let from = album.dateFrom.flatMap(utcSeconds), to = album.dateTo.flatMap(utcSeconds)
    let words = album.place.map { normText($0).split(separator: " ").map(String.init).filter { $0.count > 1 } } ?? []
    // selfies: only when the library carries camera tags at all (else a no-op); back-camera photos are out
    let frontOnly = album.camera == "front" && items.contains { $0.camera == "front" }
    return items.map { it in
        if frontOnly, it.camera == "back" { return false }
        if album.media == "photo" || album.media == "video", it.media != album.media { return false }
        if let f = from { guard let t = it.taken, t >= f else { return false } }
        if let e = to { guard let t = it.taken, t < e else { return false } }
        if let tod = album.timeOfDay, !inTimeOfDay(it.localMinutes, tod) { return false }
        if !words.isEmpty {
            let txt = " " + normText(it.place ?? "") + " "
            for w in words where !txt.contains(" \(w) ") { return false }
        }
        return true
    }
}

/// Per item: best over its units (a video's frames) of mean(look similarity) - mean(avoid similarity).
/// `units`: L2-normalized image vectors, `unitItem`: which item each unit belongs to.
public func lookScores(units: [[Float]], unitItem: [Int], nItems: Int, looks: [[Float]], avoid: [[Float]]) -> [Float] {
    if looks.isEmpty && avoid.isEmpty { return [Float](repeating: 0, count: nItems) }
    func dot(_ a: [Float], _ b: [Float]) -> Float { var s: Float = 0; for k in 0..<a.count { s += a[k] * b[k] }; return s }
    var out = [Float](repeating: -.infinity, count: nItems)
    for (u, v) in units.enumerated() {
        var s: Float = 0
        if !looks.isEmpty { s += looks.map { dot(v, $0) }.reduce(0, +) / Float(looks.count) }
        if !avoid.isEmpty { s -= avoid.map { dot(v, $0) }.reduce(0, +) / Float(avoid.count) }
        out[unitItem[u]] = max(out[unitItem[u]], s)
    }
    return out
}

/// "This specific dog / thing / place" from example photos (port of the converse subject path's vector ranking):
/// every library unit vector and every reference vector is averaged with its k nearest library vectors (neighbour
/// smoothing, DBA; library rows count themselves), then each unit scores the mean similarity to the smoothed
/// references, and each item keeps its best unit. Places with 20k everyday distractors: R-precision 0.680 -> 0.750.
/// `units`, `refs`: L2-normalized vectors.
public func subjectScores(units: [[Float]], unitItem: [Int], nItems: Int, refs: [[Float]], k: Int = 2) -> [Float] {
    func dot(_ a: [Float], _ b: [Float]) -> Float { var s: Float = 0; for i in 0..<a.count { s += a[i] * b[i] }; return s }
    func normalized(_ v: [Float]) -> [Float] { let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot() + 1e-9; return v.map { $0 / n } }
    func nearest(_ q: [Float], _ m: Int) -> [Int] {
        let s = units.map { dot(q, $0) }
        return Array(s.indices.sorted { s[$0] > s[$1] }.prefix(m))
    }
    func smooth(_ q: [Float], selfInLibrary: Bool) -> [Float] {
        let nn = nearest(q, min(k + (selfInLibrary ? 1 : 0), units.count))
        var m = selfInLibrary ? [Float](repeating: 0, count: q.count) : q
        for j in nn { for d in 0..<m.count { m[d] += units[j][d] } }
        return normalized(m)
    }
    let X = units.map { smooth($0, selfInLibrary: true) }
    let V = refs.map { smooth($0, selfInLibrary: false) }
    var best = [Float](repeating: -.infinity, count: nItems)
    for (u, x) in X.enumerated() {
        let s = V.map { dot(x, $0) }.reduce(0, +) / Float(max(V.count, 1))
        best[unitItem[u]] = max(best[unitItem[u]], s)
    }
    return best
}
