// "This specific dog / thing / place" from 1-3 example photos (port of the server's subject path, converse.py):
// 1. rank: image vectors with neighbour smoothing (FindPicsCore.subjectScores is the tested reference; here the same
//    math with Accelerate so a 30k-photo library takes seconds, not minutes);
// 2. veto: the judge sees [example | candidate] side by side and is asked whether it is the SAME individual; a photo
//    stays unless the judge is fairly sure it is not (P >= 0.2 keeps it; server measurement: kept 140/147 true photos).
// Identity of a pet / object is NOT certified: no completeness bound is shown, best matches first.
// UNTESTED on a device (written on Linux); expect compile fixes at the first Xcode build.
import Accelerate
import CoreImage
import FindPicsCore
import Foundation

extension SearchEngine {
    static let subjectQuestion = "The left panel shows %@, one specific %@. Compare individual features: colour pattern and "
        + "markings, shape, scars, distinctive details. Is the %@ in the right panel the SAME individual %@, not just a "
        + "similar-looking one? If you are not sure, answer no."
    static let subjectKeep = 0.2
    static let subjectCandidates = 300

    func runSubject(examples: [CIImage], name: String, kind: String, album: Album,
                    update: @escaping @Sendable (AlbumResult) -> Void) async throws -> AlbumResult {
        var res = AlbumResult(name: album.name)
        let entries = await index.entries
        let ids = Array(entries.keys).sorted()
        let items = await index.libraryItems(order: ids)
        let mask = scopeMask(items, album)
        let u = await index.units(ids: ids)
        let refs = try examples.map { try embedder.vector(of: $0) }
        let score = SearchEngine.smoothedSubjectScores(units: u.vectors, unitItem: u.unitItem, nItems: ids.count, refs: refs)
        let order = ids.indices.filter { mask[$0] }.sorted { score[$0] > score[$1] }.prefix(SearchEngine.subjectCandidates)
        res.inScope = order.count
        let q = String(format: SearchEngine.subjectQuestion, name, kind, kind, kind)
        guard let ref = examples.first else { res.done = true; return res }
        var kept: [String] = []
        for k in order {
            let id = ids[k]
            let key = id + "|subject|" + q
            var p = await judge.cached(key)
            if p == nil, let img = await PhotoLibrary.ciImage(id, side: 1280) {
                let v = try await judge.pYes(SearchEngine.sideBySide(ref, img), question: q)
                await judge.remember(key, v); p = v
            }
            if (p ?? 0) >= SearchEngine.subjectKeep { kept.append(id) }
            res.judged += 1
            if res.judged % 20 == 0 { res.found = kept; update(res) }
        }
        res.found = kept
        res.note = "Best matches first. A pet's or object's identity is checked side by side but not guaranteed: "
            + "look through the list."
        res.done = true
        return res
    }

    /// [reference | candidate] on white, reference <= 640 px tall, candidate <= 1024 x 640 (engine.side_by_side).
    static func sideBySide(_ ref: CIImage, _ im: CIImage, H: CGFloat = 640) -> CIImage {
        func fit(_ x: CIImage, _ w: CGFloat, _ h: CGFloat) -> CIImage {
            let s = min(1, min(w / x.extent.width, h / x.extent.height))
            let y = x.transformed(by: CGAffineTransform(scaleX: s, y: s))
            return y.transformed(by: CGAffineTransform(translationX: -y.extent.minX, y: -y.extent.minY))
        }
        let r = fit(ref, H, H), i = fit(im, H * 1.6, H)
        let W = r.extent.width + 24 + i.extent.width, Ht = max(r.extent.height, i.extent.height)
        let bg = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: W, height: Ht))
        // top-aligned like PIL's paste at (x, 0): CoreImage's origin is bottom-left
        let rr = r.transformed(by: CGAffineTransform(translationX: 0, y: Ht - r.extent.height))
        let ii = i.transformed(by: CGAffineTransform(translationX: r.extent.width + 24, y: Ht - i.extent.height))
        return ii.composited(over: rr.composited(over: bg))
    }

    /// Same as FindPicsCore.subjectScores (neighbour smoothing k = 2, mean over references, best unit per item), with
    /// the library x library similarities from Accelerate in blocks of 1,024 rows.
    static func smoothedSubjectScores(units: [[Float]], unitItem: [Int], nItems: Int, refs: [[Float]], k: Int = 2) -> [Float] {
        let n = units.count, d = units.first?.count ?? 0
        guard n > 0, d > 0, !refs.isEmpty else { return [Float](repeating: -.infinity, count: nItems) }
        let X = units.flatMap { $0 }
        func topK(_ rows: [Float], _ m: Int, _ want: Int) -> [[Int]] {     // rows: m x d -> indices of best `want` units
            var S = [Float](repeating: 0, count: m * n)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(m), Int32(n), Int32(d), 1, rows, Int32(d), X, Int32(d), 0, &S, Int32(n))
            return (0..<m).map { r in
                let row = S[(r * n)..<((r + 1) * n)]
                return Array(row.indices.sorted { S[$0] > S[$1] }.prefix(want)).map { $0 - r * n }
            }
        }
        func normalized(_ v: inout [Float]) { var s: Float = 0; vDSP_svesq(v, 1, &s, vDSP_Length(v.count)); let inv = 1 / (s.squareRoot() + 1e-9); vDSP_vsmul(v, 1, [inv], &v, 1, vDSP_Length(v.count)) }
        var smoothed = [[Float]](); smoothed.reserveCapacity(n)
        for s in stride(from: 0, to: n, by: 1024) {
            let m = min(1024, n - s)
            for (r, nn) in topK(Array(X[(s * d)..<((s + m) * d)]), m, min(k + 1, n)).enumerated() {
                _ = r
                var v = [Float](repeating: 0, count: d)
                for j in nn { vDSP_vadd(v, 1, units[j], 1, &v, 1, vDSP_Length(d)) }
                normalized(&v); smoothed.append(v)
            }
        }
        let V: [[Float]] = topK(refs.flatMap { $0 }, refs.count, min(k, n)).enumerated().map { r, nn in
            var v = refs[r]
            for j in nn { vDSP_vadd(v, 1, units[j], 1, &v, 1, vDSP_Length(d)) }
            normalized(&v); return v
        }
        var best = [Float](repeating: -.infinity, count: nItems)
        for (u, x) in smoothed.enumerated() {
            var s: Float = 0
            for v in V { var t: Float = 0; vDSP_dotpr(x, 1, v, 1, &t, vDSP_Length(d)); s += t }
            best[unitItem[u]] = max(best[unitItem[u]], s / Float(V.count))
        }
        return best
    }
}
