// Who is in a photo, from face vectors (whatever face model produced them): port of people.py. The cuts belong to the
// face model (FaceProfile): defaults are FaceProfile.shipped's; callers pass the profile's values explicitly.
// face_sims, other_identities, item_person_scores (incl. "closer to another frequent person -> not this person"),
// expand_refs, face_groups ("the faces I see most often", for naming people without Apple's People tags).
import Foundation

@inline(__always) func dot(_ a: [Float], _ b: [Float]) -> Float {
    var s: Float = 0; for k in 0..<a.count { s += a[k] * b[k] }; return s
}

/// Best cosine of every face to any reference; exact self-matches (> 0.999) are ignored. `faces`: any rows (in memory
/// or the index store's mapped file), read in blocks.
public func faceSims<V: EmbeddingRows>(_ faces: V, refs: [[Float]]) -> [Float] {
    if faces.count == 0 || refs.isEmpty { return [Float](repeating: 0, count: faces.count) }
    return blockedMax(faces, refs)
}

/// Per row of `faces`: max over `refs` of the cosine, a cosine > 0.999 counting as -1 (exact self-match).
func blockedMax<V: EmbeddingRows>(_ faces: V, _ refs: [[Float]], block: Int = 1024) -> [Float] {
    let d = faces.dim, nr = refs.count
    let R = MatrixMath.flat(refs, d: d)
    var out = [Float](repeating: -.infinity, count: faces.count)
    var S = [Float](repeating: 0, count: block * nr)
    faces.forEachBlock(size: block) { s, n, rows in
        R.withUnsafeBufferPointer { r in
            S.withUnsafeMutableBufferPointer { sp in
                MatrixMath.gemmNT(rows.baseAddress!, m: n, r.baseAddress!, n: nr, d: d, sp.baseAddress!)
            }
        }
        for i in 0..<n {
            var m = -Float.infinity
            for j in 0..<nr { let v = S[i * nr + j]; m = max(m, v > 0.999 ? -1 : v) }
            out[s + i] = m
        }
    }
    return out
}

/// Faces of OTHER frequent people: groups whose faces match the person's references below `accept` on average.
public func otherIdentities(groups: [[[Float]]], refs: [[Float]], accept: Float = FaceProfile.shipped.other) -> [[Float]] {
    if groups.isEmpty || refs.isEmpty { return [] }
    var out = [[Float]]()
    for g in groups {
        let m = g.map { f in refs.map { dot(f, $0) }.max()! }.reduce(0, +) / Float(g.count)
        if m < accept { out += g }
    }
    return out
}

/// (item score, best face index per item or -1). A face counts only if it beats every other frequent person's faces.
public func itemPersonScores<V: EmbeddingRows>(faces: V, faceItem: [Int], nItems: Int, refs: [[Float]], others: [[Float]] = [])
    -> (scores: [Float], best: [Int]) {
    var s = faceSims(faces, refs: refs)
    if !others.isEmpty && faces.count > 0 {
        let o = blockedMax(faces, others)
        for i in 0..<faces.count where o[i] >= s[i] { s[i] = -1 }
    }
    var score = [Float](repeating: -1, count: nItems), best = [Int](repeating: -1, count: nItems)
    for (i, it) in faceItem.enumerated() where best[it] == -1 || s[i] > score[it] { score[it] = s[i]; best[it] = i }
    return (score, best)
}

/// Add faces with similarity >= accept as new references (query-time clustering).
public func expandRefs<V: EmbeddingRows>(_ faces: V, refs r0: [[Float]], accept: Float, rounds: Int = 2, maxNew: Int = 2000) -> [[Float]] {
    var refs = r0
    for _ in 0..<rounds {
        let s = faceSims(faces, refs: refs)
        var new = s.indices.filter { s[$0] >= accept }
        if new.count <= refs.count { break }
        new.sort { s[$0] != s[$1] ? s[$0] > s[$1] : $0 < $1 }      // numpy argsort(-s) is stable for ties
        refs = new.prefix(maxNew).map { faces[$0] }
    }
    return refs
}

public struct FaceGroup: Equatable, Sendable { public let faces: [Int]; public let items: [Int]; public let rep: Int }

/// The most frequent people without names: greedy grouping (take the face with the most live neighbours at cosine
/// >= accept, make it + its neighbours a group, remove them). Faces below minPx / minDet are skipped.
/// The greedy step compares every face with every other (F^2): above `cap` faces (Reza's library: ~200k) it runs on a
/// bounded sample instead (faceSampleRows: the newest cap/2 faces + a fixed hash sample of the rest; the people who
/// appear OFTEN are all in it), and every other face then joins the first group whose seed face it matches at
/// >= accept (O(F x groups)). eval/face_groups_sampled.py: top-10 agreement with grouping everything.
/// Neighbours are kept as bitsets (n^2 / 8 bytes: 50 MB at n = 20k), similarities come from MatrixMath.gemmNT.
public func faceGroups<V: EmbeddingRows>(faces: V, faceItem: [Int], facePx: [Float], det: [Float], top: Int = 12,
                                         accept: Float = FaceProfile.shipped.group, minPx: Float = 40, minDet: Float = 0.7,
                                         cap: Int = .max, faceTaken: [Double?]? = nil, newestFraction: Double = faceSampleNewest,
                                         seed: UInt64 = 0) -> [FaceGroup] {
    let rows = (0..<faces.count).filter { facePx[$0] >= minPx && det[$0] >= minDet }
    let sampled = rows.count > cap
    let samp = sampled ? faceSampleRows(rows, taken: faceTaken ?? [], cap: cap, newestFraction: newestFraction, seed: seed) : rows
    // on a sample the greedy step makes 3x `top` groups: a short video gives one person many faces on 1-2 photos,
    // and such groups would otherwise take slots that, ranked by photos after the assignment, belong to others
    let greedyTop = sampled ? 3 * top : top
    let n = samp.count, d = faces.dim
    guard n > 0, d > 0 else { return [] }
    // normalized sample rows, contiguous
    var E = [Float](repeating: 0, count: n * d)
    E.withUnsafeMutableBufferPointer { e in
        for (k, r) in samp.enumerated() {
            faces.withRows(r, 1) { v in
                var ss: Float = 0
                for j in 0..<d { ss += v[j] * v[j] }
                let nn = sqrt(ss) + 1e-8
                for j in 0..<d { e[k * d + j] = v[j] / nn }
            }
        }
    }
    // neighbour bitsets: bit j of row i <=> cos(i, j) >= accept (self included)
    let words = (n + 63) / 64
    var nb = [UInt64](repeating: 0, count: n * words)
    let blk = 256
    var S = [Float](repeating: 0, count: blk * n)
    E.withUnsafeBufferPointer { e in
        S.withUnsafeMutableBufferPointer { sp in
            nb.withUnsafeMutableBufferPointer { bits in
                var s0 = 0
                while s0 < n {
                    let m = min(blk, n - s0)
                    MatrixMath.gemmNT(e.baseAddress! + s0 * d, m: m, e.baseAddress!, n: n, d: d, sp.baseAddress!)
                    for r in 0..<m {
                        let row = (s0 + r) * words, base = r * n
                        for j in 0..<n where sp[base + j] >= accept { bits[row + j >> 6] |= 1 << UInt64(j & 63) }
                    }
                    s0 += m
                }
            }
        }
    }
    var alive = [UInt64](repeating: 0, count: words)
    for j in 0..<n { alive[j >> 6] |= 1 << UInt64(j & 63) }
    var isAlive = [Bool](repeating: true, count: n), nAlive = n
    var groups = [FaceGroup](), seeds = [Int]()
    while groups.count < greedyTop && nAlive > 0 {
        // the alive face with the most alive neighbours (first one on ties), counted afresh each round
        var c = -1, dmax = -1
        nb.withUnsafeBufferPointer { bits in
            for i in 0..<n where isAlive[i] {
                var cnt = 0
                let row = i * words
                for w in 0..<words { cnt += (bits[row + w] & alive[w]).nonzeroBitCount }
                if cnt > dmax { dmax = cnt; c = i }
            }
        }
        if dmax < 2 { break }
        var mem = [Int]()
        for w in 0..<words {
            var x = nb[c * words + w] & alive[w]
            while x != 0 { let b = x.trailingZeroBitCount; mem.append(w * 64 + b); x &= x - 1 }
        }
        for m in mem { alive[m >> 6] &= ~(1 << UInt64(m & 63)); isAlive[m] = false }
        if isAlive[c] { isAlive[c] = false; alive[c >> 6] &= ~(1 << UInt64(c & 63)) }
        nAlive = isAlive.reduce(0) { $0 + ($1 ? 1 : 0) }
        var mean = [Float](repeating: 0, count: d)
        for m in mem { for k in 0..<d { mean[k] += E[m * d + k] } }
        mean = mean.map { $0 / Float(mem.count) }
        var bi = mem[0], bs = -Float.infinity
        for m in mem {
            var v: Float = 0
            for k in 0..<d { v += E[m * d + k] * mean[k] }
            if v > bs { bs = v; bi = m }
        }
        let fr = mem.map { samp[$0] }
        groups.append(FaceGroup(faces: fr, items: Array(Set(fr.map { faceItem[$0] })).sorted(), rep: samp[bi]))
        seeds.append(c)
    }
    // the faces left out of the sample join the first group whose seed they match
    if sampled && !groups.isEmpty {
        let inSample = Set(samp)
        let rest = rows.filter { !inSample.contains($0) }
        let Sd = seeds.flatMap { Array(E[($0 * d)..<(($0 + 1) * d)]) }
        var extra = [[Int]](repeating: [], count: groups.count)
        let G = seeds.count
        var T = [Float](repeating: 0, count: 1024 * G)
        let view = faces.subset(rest)
        view.forEachBlock(size: 1024) { s0, m, blockRows in
            var Nb = [Float](repeating: 0, count: m * d)
            for r in 0..<m {
                var ss: Float = 0
                for j in 0..<d { ss += blockRows[r * d + j] * blockRows[r * d + j] }
                let nn = sqrt(ss) + 1e-8
                for j in 0..<d { Nb[r * d + j] = blockRows[r * d + j] / nn }
            }
            Nb.withUnsafeBufferPointer { a in Sd.withUnsafeBufferPointer { b in T.withUnsafeMutableBufferPointer { t in
                MatrixMath.gemmNT(a.baseAddress!, m: m, b.baseAddress!, n: G, d: d, t.baseAddress!)
            } } }
            for r in 0..<m { if let g = (0..<G).first(where: { T[r * G + $0] >= accept }) { extra[g].append(rest[s0 + r]) } }
        }
        for g in groups.indices where !extra[g].isEmpty {
            let fr = (groups[g].faces + extra[g]).sorted()
            groups[g] = FaceGroup(faces: fr, items: Array(Set(fr.map { faceItem[$0] })).sorted(), rep: groups[g].rep)
        }
    }
    return Array(groups.enumerated().sorted { $0.element.items.count != $1.element.items.count ? $0.element.items.count > $1.element.items.count : $0.offset < $1.offset }.map { $0.element }.prefix(top))
}

/// Which faces the bounded grouping looks at: the newest cap x newestFraction (by photo date, undated last, then face
/// order) and the rest by a fixed hash rank (splitmix64 of the face index + seed), cap in all, in face order.
public func faceSampleRows(_ rows: [Int], taken: [Double?], cap: Int, newestFraction: Double = faceSampleNewest,
                           seed: UInt64 = 0) -> [Int] {
    guard rows.count > cap else { return rows }
    func t(_ r: Int) -> Double? { r < taken.count ? taken[r] : nil }
    let newest = rows.sorted { a, b in
        switch (t(a), t(b)) {
        case (nil, nil): return a < b
        case (nil, _): return false
        case (_, nil): return true
        case let (x?, y?): return x != y ? x > y : a < b
        }
    }.prefix(Int(Double(cap) * newestFraction))
    var chosen = Set(newest)
    let rest = rows.filter { !chosen.contains($0) }
        .map { (splitmix64(UInt64($0) &+ seed), $0) }.sorted { $0.0 != $1.0 ? $0.0 < $1.0 : $0.1 < $1.1 }
    for (_, r) in rest.prefix(cap - newest.count) { chosen.insert(r) }
    return chosen.sorted()
}

func splitmix64(_ x0: UInt64) -> UInt64 {
    var x = x0 &+ 0x9E37_79B9_7F4A_7C15
    x = (x ^ (x >> 30)) &* 0xBF58_476D_1CE4_E5B9
    x = (x ^ (x >> 27)) &* 0x94D0_49BB_1331_11EB
    return x ^ (x >> 31)
}

/// Share of the face sample taken from the newest photos (the rest: a fixed hash sample of all faces).
public let faceSampleNewest = 0.0
