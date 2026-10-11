// [F8] "Is this you?" without trusting a whole face group blindly. A face group is an automatic cluster; if it holds
// someone else, confirming it used to teach the app that this other person is "me", and every person search inherited
// it. Now the person sees every face of the group (least alike first, so an intruder is near the top), excludes the
// wrong ones, and only the kept faces become references. Excluded faces are kept as NEGATIVES for that person:
// expandRefs never re-adds them and a face at least as close to one of them never counts as the person
// (FaceMatch.swift). The saved pick can be opened again later (People screen) to remove faces or reset it.
import Foundation

/// What a saved person is made of: reference faces (`refs`, with where each one is) and the faces the person said are
/// NOT them (`negatives`, with where each one is). `refs[k]` pairs with `sources[k]`; same for the negatives.
public struct PersonPick: Equatable, Sendable {
    public var refs: [[Float]]
    public var sources: [FaceSource]
    public var negatives: [[Float]]
    public var negativeSources: [FaceSource]
    public init(refs: [[Float]], sources: [FaceSource], negatives: [[Float]] = [], negativeSources: [FaceSource] = []) {
        self.refs = refs; self.sources = sources; self.negatives = negatives; self.negativeSources = negativeSources
    }
}

/// Two face vectors are the same face (same photo, same model): cosine >= 0.9999 (locateRefs' rule).
func sameFace(_ a: [Float], _ b: [Float]) -> Bool {
    guard a.count == b.count, !a.isEmpty else { return false }
    return dot(unit(a), unit(b)) >= 0.9999
}

/// Earlier negatives + new ones, each face once, minus any face that is now a reference (the person kept it this time).
/// Negatives and their sources stay paired; if either input's sources do not pair with its vectors, the result has
/// no sources (the vectors still work as negatives; only a face-model change cannot re-derive them).
public func mergeNegatives(_ a: [[Float]], _ aSrc: [FaceSource], _ b: [[Float]], _ bSrc: [FaceSource], refs: [[Float]])
    -> (negatives: [[Float]], sources: [FaceSource]) {
    let paired = a.count == aSrc.count && b.count == bSrc.count
    var out = [[Float]](), src = [FaceSource]()
    let all = a + b, allSrc = paired ? aSrc + bSrc : []
    for (k, v) in all.enumerated() {
        if refs.contains(where: { sameFace($0, v) }) || out.contains(where: { sameFace($0, v) }) { continue }
        out.append(v); if paired { src.append(allSrc[k]) }
    }
    return (out, src)
}

/// Confirm a face group minus the faces the person excluded (`excluded`: positions in `vectors`). Kept faces become
/// the references; excluded ones join the earlier negatives. nil when nothing is kept or vectors / sources do not pair.
public func pickFromGroup(vectors: [[Float]], sources: [FaceSource], excluded: Set<Int>, earlier: PersonPick? = nil) -> PersonPick? {
    guard vectors.count == sources.count else { return nil }
    let keep = vectors.indices.filter { !excluded.contains($0) }
    guard !keep.isEmpty else { return nil }
    let drop = vectors.indices.filter { excluded.contains($0) }
    let refs = keep.map { vectors[$0] }
    let n = mergeNegatives(earlier?.negatives ?? [], earlier?.negativeSources ?? [], drop.map { vectors[$0] }, drop.map { sources[$0] },
                           refs: refs)
    return PersonPick(refs: refs, sources: keep.map { sources[$0] }, negatives: n.negatives, negativeSources: n.sources)
}

/// The People screen: remove some of a saved person's reference faces (`remove`: positions in `pick.refs`). Removed
/// faces become negatives. nil when nothing would be left (reset instead) or refs / sources do not pair up (an old
/// pick whose faces cannot be told apart: only "pick again" or reset).
public func removeFromPick(_ pick: PersonPick, remove: Set<Int>) -> PersonPick? {
    guard pick.refs.count == pick.sources.count else { return nil }
    return pickFromGroup(vectors: pick.refs, sources: pick.sources, excluded: remove, earlier: pick)
}

/// A new pick from "Add a photo of you": the earlier negatives still hold, except faces now picked.
public func pickFromPhotos(refs: [[Float]], sources: [FaceSource], earlier: PersonPick? = nil) -> PersonPick? {
    guard !refs.isEmpty else { return nil }
    let n = mergeNegatives(earlier?.negatives ?? [], earlier?.negativeSources ?? [], [], [], refs: refs)
    return PersonPick(refs: refs, sources: sources, negatives: n.negatives, negativeSources: n.sources)
}

/// Faces of a group the person already said are not them: shown as excluded when the group is reviewed again.
public func alreadyExcluded(vectors: [[Float]], negatives: [[Float]]) -> Set<Int> {
    guard !negatives.isEmpty else { return [] }
    return Set(vectors.indices.filter { i in negatives.contains { sameFace($0, vectors[i]) } })
}

/// Review order: least like the rest of the group first (cosine to the mean of the group's unit vectors, ascending;
/// ties by position). An intruder in a cluster is the face least like its mean, so it is near the top of the grid
/// instead of somewhere in hundreds of faces.
public func leastAlikeFirst(_ vectors: [[Float]]) -> [Int] {
    guard let d = vectors.first?.count, d > 0 else { return Array(vectors.indices) }
    let u = vectors.map(unit)
    var mean = [Float](repeating: 0, count: d)
    for v in u where v.count == d { for k in 0..<d { mean[k] += v[k] } }
    let m = unit(mean)
    let s = u.map { $0.count == d ? dot($0, m) : -2 }
    return vectors.indices.sorted { s[$0] != s[$1] ? s[$0] < s[$1] : $0 < $1 }
}

/// The square around a face to show it in a grid, as fractions of the image (0...1): the box's longer side times
/// (1 + 2 x pad), centred on the box, moved (not shrunk) to stay inside the image; shrunk only when the image is
/// smaller than the square. `box`: x1 y1 x2 y2 in the analysed image's pixels (imageW x imageH).
public func faceCropRect(box: [Double], imageW: Double, imageH: Double, pad: Double = 0.3) -> Rect {
    guard box.count == 4, imageW > 0, imageH > 0 else { return Rect(0, 0, 1, 1) }
    let cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2
    let side = min(max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * pad), imageW, imageH)
    let x1 = min(max(0, cx - side / 2), imageW - side), y1 = min(max(0, cy - side / 2), imageH - side)
    return Rect(x1 / imageW, y1 / imageH, (x1 + side) / imageW, (y1 + side) / imageH)
}

/// How big to read the photo so the face crop is about `want` px: the image's long side, clamped to lo...hi.
public func faceCropReadSide(box: [Double], imageW: Double, imageH: Double, want: Double = 180, pad: Double = 0.3,
                             lo: Double = 300, hi: Double = 1600) -> Double {
    let r = faceCropRect(box: box, imageW: imageW, imageH: imageH, pad: pad)
    let frac = max(r.x2 - r.x1, 1e-6) * imageW / max(imageW, imageH)       // crop width / image long side
    return min(hi, max(lo, (want / frac).rounded(.up)))
}

/// The picker's top line: picking your own photos is the most reliable way (the group is a guess).
public func pickPhotosPrompt(_ person: String) -> String {
    isOwnerWord(person) ? "Most accurate: pick 1-3 photos of yourself" : "Most accurate: pick 1-3 photos of \(person)"
}

/// The confirm button under a reviewed group: says how many faces it will use.
public func confirmGroupLabel(_ person: String, kept: Int, total: Int) -> String {
    let who = isOwnerWord(person) ? "me" : person
    if kept == 0 { return "Every face is excluded" }
    return kept == total ? "Yes, these \(kept) faces are \(who)" : "Yes, these \(kept) of \(total) faces are \(who)"
}
