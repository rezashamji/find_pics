// Fast stage of a search: which photos are in scope (media / dates / clock / place) and how well each matches the
// words, from image vectors computed once at indexing. Port of engine.scope_mask / time_of_day_mask / look_scores.
import Foundation

public struct LibraryItem: Sendable {
    public var id: String
    public var media: String            // "photo" | "video"
    public var taken: Double?            // seconds since 1970, UTC
    public var localMinutes: Int?        // wall-clock minutes after midnight where it was taken; nil = unknown
    public var place: String?            // place name text (offline-geocoded GPS)
    public var camera: String            // "front" | "back" (EXIF lens model) | "" unknown
    public var isScreenshot: Bool        // PhotoKit's screenshot flag: never a selfie
    public init(id: String, media: String, taken: Double?, localMinutes: Int?, place: String?, camera: String = "",
                isScreenshot: Bool = false) {
        self.id = id; self.media = media; self.taken = taken; self.localMinutes = localMinutes; self.place = place
        self.camera = camera; self.isScreenshot = isScreenshot
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
        if frontOnly, it.camera == "back" || it.isScreenshot { return false }
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
/// `units`: L2-normalized image vectors (in memory, or the index store's mapped rows: read 1,024 rows at a time, so RAM
/// stays bounded for any library size), `unitItem`: which item each unit belongs to.
public func lookScores<V: EmbeddingRows>(units: V, unitItem: [Int], nItems: Int, looks: [[Float]], avoid: [[Float]]) -> [Float] {
    if looks.isEmpty && avoid.isEmpty { return [Float](repeating: 0, count: nItems) }
    var out = [Float](repeating: -.infinity, count: nItems)
    let d = units.dim, nl = looks.count, na = avoid.count, nq = nl + na
    guard units.count > 0, d > 0 else { return out }
    let Q = MatrixMath.flat(looks + avoid, d: d)
    let block = 1024
    var S = [Float](repeating: 0, count: block * nq)
    units.forEachBlock(size: block) { s, n, rows in
        Q.withUnsafeBufferPointer { q in
            S.withUnsafeMutableBufferPointer { sp in MatrixMath.gemmNT(rows.baseAddress!, m: n, q.baseAddress!, n: nq, d: d, sp.baseAddress!) }
        }
        for r in 0..<n {
            var sc: Float = 0
            if nl > 0 { var t: Float = 0; for l in 0..<nl { t += S[r * nq + l] }; sc += t / Float(nl) }
            if na > 0 { var t: Float = 0; for a in 0..<na { t += S[r * nq + nl + a] }; sc -= t / Float(na) }
            let it = unitItem[s + r]
            out[it] = max(out[it], sc)
        }
    }
    return out
}

/// subjectScores for a library of any size: the same neighbour smoothing (k nearest library units by cosine, a unit
/// counting itself; references smoothed with their k nearest), computed in blocks (query block x library block with
/// MatrixMath.gemmNT) without holding the library or its smoothed copy in RAM. Same scores as subjectScores up to float
/// rounding and the order of exact ties. Cost is still n^2 dot products (library x library).
public func subjectScoresBlocked<V: EmbeddingRows>(units: V, unitItem: [Int], nItems: Int, refs: [[Float]], k: Int = 2,
                                                  queryBlock: Int = 1024, libraryBlock: Int = 2048) -> [Float] {
    let n = units.count, d = units.dim
    guard n > 0, d > 0, !refs.isEmpty else { return [Float](repeating: -.infinity, count: nItems) }
    /// For each of the m query rows (m x d): the `want` library units with the highest cosine, best first.
    func topK(_ q: UnsafeBufferPointer<Float>, _ m: Int, _ want: Int) -> [[Int]] {
        var best = [[(Int, Float)]](repeating: [], count: m)
        var S = [Float](repeating: 0, count: m * libraryBlock)
        units.forEachBlock(size: libraryBlock) { s, nb, lib in
            S.withUnsafeMutableBufferPointer { sp in MatrixMath.gemmNT(q.baseAddress!, m: m, lib.baseAddress!, n: nb, d: d, sp.baseAddress!) }
            for r in 0..<m {
                var b = best[r]
                for j in 0..<nb {
                    let v = S[r * nb + j]
                    if b.count < want { b.append((s + j, v)); b.sort { $0.1 > $1.1 } }
                    else if v > b[want - 1].1 { b[want - 1] = (s + j, v); b.sort { $0.1 > $1.1 } }
                }
                best[r] = b
            }
        }
        return best.map { $0.map { $0.0 } }
    }
    func normalized(_ v: [Float]) -> [Float] { let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot() + 1e-9; return v.map { $0 / n } }
    func addRows(_ m: inout [Float], _ nn: [Int]) { for j in nn { units.withRows(j, 1) { r in for i in 0..<d { m[i] += r[i] } } } }
    // smoothed references: ref + its k nearest library units
    let R = MatrixMath.flat(refs, d: d)
    let refNN = R.withUnsafeBufferPointer { topK($0, refs.count, min(k, n)) }
    let V: [[Float]] = refs.enumerated().map { r, v in var m = v; addRows(&m, refNN[r]); return normalized(m) }
    var best = [Float](repeating: -.infinity, count: nItems)
    units.forEachBlock(size: queryBlock) { s, m, q in
        let nn = topK(q, m, min(k + 1, n))
        for r in 0..<m {
            var x = [Float](repeating: 0, count: d)
            addRows(&x, nn[r])
            x = normalized(x)
            var sc: Float = 0
            for v in V { var t: Float = 0; for i in 0..<d { t += x[i] * v[i] }; sc += t }
            let it = unitItem[s + r]
            best[it] = max(best[it], sc / Float(V.count))
        }
    }
    return best
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

/// "only from Paris" written as filter question "Is the location Paris?": a judge cannot see which city a photo is from;
/// GPS place names can. A capitalized name in the filter that matches this library's place names becomes the place
/// filter (port of converse.filter_to_place).
public func filterToPlace(_ items: [LibraryItem], _ a: Album) -> Album {
    guard let fq = a.filterQuestion, !fq.isEmpty else { return a }
    let names = searchAll(#"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)"#, fq).compactMap { $0.group(1) }
    for n in names.reversed() {
        if ["Is", "Does", "Are", "Was", "Do", "The"].contains(String(n.split(separator: " ")[0])) { continue }
        var probe = Album(name: "p"); probe.place = n
        if scopeMask(items, probe).contains(true) { var b = a; b.place = n; b.filterQuestion = nil; return b }
    }
    return a
}

/// A 'place' that matches no item's place name in THIS library ("beach", "gym") is a kind of scene: turn it into a
/// visual condition instead of silently returning nothing (port of converse.place_or_look).
public func placeOrLook(_ items: [LibraryItem], _ a: Album) -> Album {
    guard let pl = a.place, !pl.isEmpty else { return a }
    var probe = Album(name: "p"); probe.place = pl
    if scopeMask(items, probe).contains(true) { return a }
    let q = truthy(a.person) ? "Is the person in the red box at a \(pl)?" : "Was this photo taken at a \(pl)?"
    var b = a
    if let jq = a.judgeQuestion, !jq.isEmpty {
        var t = jq; while let c = t.last, c == "?" || c == " " { t.removeLast() }
        b.judgeQuestion = "\(t), and \(q.prefix(1).lowercased())\(q.dropFirst())"
    } else { b.judgeQuestion = q }
    b.place = nil; b.looks = a.looks + [pl]
    return b
}
