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
public func faceGroups<V: EmbeddingRows>(faces: V, faceItem: [Int], facePx: [Float], det: [Float], top: Int = 12,
                       accept: Float = FaceProfile.shipped.group, minPx: Float = 40, minDet: Float = 0.7) -> [FaceGroup] {
    let rows = (0..<faces.count).filter { facePx[$0] >= minPx && det[$0] >= minDet }
    let E = rows.map { r -> [Float] in let v = faces[r]; let n = sqrt(dot(v, v)) + 1e-8; return v.map { $0 / n } }
    let nb: [[Int]] = E.indices.map { i in E.indices.filter { dot(E[i], E[$0]) >= accept } }
    var alive = [Bool](repeating: true, count: rows.count), groups = [FaceGroup]()
    while groups.count < top && alive.contains(true) {
        var c = -1, dmax = -1
        for i in rows.indices { let d = alive[i] ? nb[i].filter { alive[$0] }.count : -1; if d > dmax { dmax = d; c = i } }
        if dmax < 2 { break }
        let mem = nb[c].filter { alive[$0] }
        for m in mem { alive[m] = false }; alive[c] = false
        var mean = [Float](repeating: 0, count: E[0].count)
        for m in mem { for k in 0..<mean.count { mean[k] += E[m][k] } }
        mean = mean.map { $0 / Float(mem.count) }
        var bi = mem[0], bs = -Float.infinity
        for m in mem { let v = dot(E[m], mean); if v > bs { bs = v; bi = m } }
        let fr = mem.map { rows[$0] }
        groups.append(FaceGroup(faces: fr, items: Array(Set(fr.map { faceItem[$0] })).sorted(), rep: rows[bi]))
    }
    return groups.enumerated().sorted { $0.element.items.count != $1.element.items.count ? $0.element.items.count > $1.element.items.count : $0.offset < $1.offset }.map { $0.element }
}
