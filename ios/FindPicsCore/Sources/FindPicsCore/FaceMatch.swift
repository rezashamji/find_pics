// Who is in a photo, from face vectors (whatever face model produced them): port of people.py.
// face_sims, other_identities, item_person_scores (incl. "closer to another frequent person -> not this person"),
// expand_refs, face_groups ("the faces I see most often", for naming people without Apple's People tags).
import Foundation

@inline(__always) func dot(_ a: [Float], _ b: [Float]) -> Float {
    var s: Float = 0; for k in 0..<a.count { s += a[k] * b[k] }; return s
}

/// Best cosine of every face to any reference; exact self-matches (> 0.999) are ignored.
public func faceSims(_ faces: [[Float]], refs: [[Float]]) -> [Float] {
    if faces.isEmpty || refs.isEmpty { return [Float](repeating: 0, count: faces.count) }
    return faces.map { f in refs.map { r in let s = dot(f, r); return s > 0.999 ? -1 : s }.max()! }
}

/// Faces of OTHER frequent people: groups whose faces match the person's references below `accept` on average.
public func otherIdentities(groups: [[[Float]]], refs: [[Float]], accept: Float = 0.40) -> [[Float]] {
    if groups.isEmpty || refs.isEmpty { return [] }
    var out = [[Float]]()
    for g in groups {
        let m = g.map { f in refs.map { dot(f, $0) }.max()! }.reduce(0, +) / Float(g.count)
        if m < accept { out += g }
    }
    return out
}

/// (item score, best face index per item or -1). A face counts only if it beats every other frequent person's faces.
public func itemPersonScores(faces: [[Float]], faceItem: [Int], nItems: Int, refs: [[Float]], others: [[Float]] = [])
    -> (scores: [Float], best: [Int]) {
    var s = faceSims(faces, refs: refs)
    if !others.isEmpty {
        for (i, f) in faces.enumerated() {
            let o = others.map { r -> Float in let v = dot(f, r); return v > 0.999 ? -1 : v }.max()!
            if o >= s[i] { s[i] = -1 }
        }
    }
    var score = [Float](repeating: -1, count: nItems), best = [Int](repeating: -1, count: nItems)
    for (i, it) in faceItem.enumerated() where best[it] == -1 || s[i] > score[it] { score[it] = s[i]; best[it] = i }
    return (score, best)
}

/// Add faces with similarity >= accept as new references (query-time clustering).
public func expandRefs(_ faces: [[Float]], refs r0: [[Float]], accept: Float, rounds: Int = 2, maxNew: Int = 2000) -> [[Float]] {
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
public func faceGroups(faces: [[Float]], faceItem: [Int], facePx: [Float], det: [Float], top: Int = 12,
                       accept: Float = 0.55, minPx: Float = 40, minDet: Float = 0.7) -> [FaceGroup] {
    let rows = faces.indices.filter { facePx[$0] >= minPx && det[$0] >= minDet }
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
