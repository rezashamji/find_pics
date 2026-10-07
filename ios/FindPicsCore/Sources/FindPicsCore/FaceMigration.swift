// Changing the face model (FaceProfile.shipped) without mixing vectors of two models: the index re-embeds the faces it
// already found (same boxes), and a saved person ("me", "Mom") is re-derived from THE SAME faces, located by where they
// are (photo, video frame, face box), or asked again. These are the platform-free parts; the app does the reading.
import Foundation

/// Where a saved person's reference face is: the photo / video (`id`), the video frame (`t`), the face box in the
/// analysed image (x1, y1, x2, y2 pixels, top-left origin) and that image's size. `side`: long side of the read the
/// reference vector came from (FaceUpgrade: below faceReadSide it is re-derived once the photo is re-read; nil unknown).
public struct FaceSource: Codable, Equatable, Sendable {
    public let id: String
    public let t: Double?
    public let box: [Double]
    public let imageW: Double, imageH: Double
    public let side: Double?
    public init(id: String, t: Double?, box: [Double], imageW: Double, imageH: Double, side: Double? = nil) {
        self.id = id; self.t = t; self.box = box; self.imageW = imageW; self.imageH = imageH; self.side = side
    }
    public func withSide(_ s: Double?) -> FaceSource { FaceSource(id: id, t: t, box: box, imageW: imageW, imageH: imageH, side: s) }
}

/// For each reference, the library face it was copied from (naming a face group copies the index's vectors), or nil.
/// Exact equality first (compared after rounding to Float16, the index store's precision: a reference saved from the
/// old Float32 index still finds its face in the Float16 store); then cosine >= 0.9999 for the rest (a face
/// re-detected on the same photo by the same model). `faces` is read row by row (never copied whole).
public func locateRefs<V: EmbeddingRows>(_ refs: [[Float]], in faces: V, maxSearched: Int = 50) -> [Int?] {
    func key(_ p: UnsafeBufferPointer<Float>) -> [UInt16] { p.map { halfBits($0) } }
    var want = [[UInt16]: [Int]]()
    for (k, r) in refs.enumerated() { r.withUnsafeBufferPointer { want[key($0), default: []].append(k) } }
    var out = [Int?](repeating: nil, count: refs.count)
    if !want.isEmpty {
        for i in 0..<faces.count {
            faces.withRows(i, 1) { f in
                if let ks = want[key(f)] { for k in ks where out[k] == nil { out[k] = i } }
            }
        }
    }
    var searched = 0
    for k in out.indices where out[k] == nil && searched < maxSearched {
        searched += 1
        let r = refs[k], nr = sqrt(dot(r, r)) + 1e-8
        var best = -1, bs: Float = 0.9999
        for i in 0..<faces.count {
            faces.withRows(i, 1) { f in
                guard f.count == r.count else { return }
                var ff: Float = 0, fr: Float = 0
                for j in 0..<r.count { ff += f[j] * f[j]; fr += f[j] * r[j] }
                let s = fr / (nr * (sqrt(ff) + 1e-8))
                if s >= bs { bs = s; best = i }
            }
        }
        if best >= 0 { out[k] = best }
    }
    return out
}

/// Intersection over union of two boxes (x1, y1, x2, y2).
public func boxIoU(_ a: [Double], _ b: [Double]) -> Double {
    let w = max(0, min(a[2], b[2]) - max(a[0], b[0])), h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return u > 0 ? w * h / u : 0
}

/// The candidate face that is the source face: boxes compared in image-relative coordinates (the photo may have been
/// read at another size), best IoU >= 0.5; nil if none.
public func matchSourceFace(_ s: FaceSource, candidates: [(box: [Double], imageW: Double, imageH: Double)]) -> Int? {
    func rel(_ b: [Double], _ w: Double, _ h: Double) -> [Double] { [b[0] / w, b[1] / h, b[2] / w, b[3] / h] }
    let want = rel(s.box, s.imageW, s.imageH)
    var best: Int? = nil, bi = 0.5
    for (k, c) in candidates.enumerated() {
        let v = boxIoU(want, rel(c.box, c.imageW, c.imageH))
        if v >= bi { bi = v; best = k }
    }
    return best
}

/// A saved person survives a face-model change only if most of their reference faces were found again (re-derived
/// from those); otherwise the app asks "Is this you?" again rather than guess.
public func keepAfterRederive(found: Int, total: Int) -> Bool { found >= 1 && 2 * found >= total }
