// Fast mode's ORDER = cosine + a learned head, blended by z-scores (RESULTS 48; eval/learned_head/head_ship.py).
//
// Fast mode asks the slow judge (~2-4 s a photo) about photos in rank order, so a better order finds the same matches
// with fewer judge calls. The order used to be the cosine between each photo's stored PE-Core-B-16 vector and the look
// text (lookScores). The cluster trained a small network ("head") on 2,412 real planner questions: input = the stored
// photo vector and the judge question's text vector, output = a logit of "the judge would say yes". Alone it is worse
// than cosine on concepts it never saw; ADDED to cosine (each turned into a z-score over the search's photos, equal
// weights fixed before testing) it is better on every test: with the shipped stop rule and this file's head, fast mode
// reaches the phone cosine order's first-answer matches with 27% (audit4) / 42% (16 libraries) fewer judge calls
// (RESULTS 48, current plan). It is a RANKING only: it never decides a match.
//
// Cost on the phone: the question is fixed for a search, so the head's first layer folds once per search into one
// H1 x D matrix (W_img + W_prod * question, column-scaled) and a constant; per photo that is one 1024 x 1024 and one
// 256 x 1024 matrix-vector product (~2.6 MFLOP), batched into matrix products. A head file with the |img - txt| block
// (B = 4) needs one more 1024 x 1024 product per photo, which cannot be folded.
//
// The matrix products go through a GemmNT closure: MatrixMath.gemmNT by default (Accelerate's cblas_sgemm on Apple, a
// plain loop on Linux, where the golden test runs); the app may pass an MLX / BNNS / Metal one instead.
import Foundation

/// C (m x n, row-major) = A (m x d) * B (n x d)^T, all Float32 row-major (the contract of MatrixMath.gemmNT).
public typealias GemmNT = @Sendable (_ A: UnsafePointer<Float>, _ m: Int, _ B: UnsafePointer<Float>, _ n: Int, _ d: Int,
                           _ C: UnsafeMutablePointer<Float>) -> Void

public let defaultGemmNT: GemmNT = { a, m, b, n, d, c in MatrixMath.gemmNT(a, m: m, b, n: n, d: d, c) }

/// The head's logit is clipped to +-this before its z-score: logit(1 - 1e-6), as the cluster's blend clipped P.
public let headLogitClip: Float = 13.815510557964274

public enum HeadFileError: Error, Equatable {
    case badMagic, badVersion(UInt32), badShape(String), truncated(expected: Int, got: Int)
}

/// The learned head, read from an ".fphead" file (layout: eval/learned_head/head_ship.py; little-endian):
/// "FPHEAD01" | u32 version=1, D, B, H1, H2, metaLen | metaLen bytes JSON | zero pad to 16 |
/// W1 f16 [H1][B*D] (blocks: img, txt, img*txt[, |img - txt|]; standardisation folded in) | b1 f32 [H1] |
/// W2 f16 [H2][H1] | b2 f32 [H2] | w3 f32 [H2] | b3 f32.
public struct HeadRanker: Sendable {
    public let dim: Int, blocks: Int, hidden1: Int, hidden2: Int
    /// Provenance JSON from the file (run, seed, features, how the blend is computed).
    public let meta: String
    let w1: [Float], b1: [Float], w2: [Float], b2: [Float], w3: [Float], b3: Float

    public init(contentsOf url: URL) throws { try self.init(data: Data(contentsOf: url, options: .mappedIfSafe)) }

    public init(data: Data) throws {
        guard data.count >= 32, data.prefix(8) == Data("FPHEAD01".utf8) else { throw HeadFileError.badMagic }
        func u32(_ o: Int) -> UInt32 { data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt32.self)) } }
        guard u32(8) == 1 else { throw HeadFileError.badVersion(u32(8)) }
        let D = Int(u32(12)), B = Int(u32(16)), H1 = Int(u32(20)), H2 = Int(u32(24)), metaLen = Int(u32(28))
        guard D > 0, H1 > 0, H2 > 0, B == 3 || B == 4 else { throw HeadFileError.badShape("D \(D) B \(B) H1 \(H1) H2 \(H2)") }
        var o = 32 + metaLen
        o += (16 - o % 16) % 16
        let need = o + H1 * B * D * 2 + H1 * 4 + H2 * H1 * 2 + H2 * 4 + H2 * 4 + 4
        guard data.count == need else { throw HeadFileError.truncated(expected: need, got: data.count) }
        meta = String(decoding: data[(data.startIndex + 32)..<(data.startIndex + 32 + metaLen)], as: UTF8.self)
        func halfs(_ n: Int) -> [Float] {
            var bits = [UInt16](repeating: 0, count: n), out = [Float](repeating: 0, count: n)
            data.withUnsafeBytes { raw in bits.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[o..<(o + 2 * n)])) } }
            bits.withUnsafeBytes { src in out.withUnsafeMutableBufferPointer { MatrixMath.halfToFloat(src.baseAddress!, $0.baseAddress!, n) } }
            o += 2 * n; return out
        }
        func floats(_ n: Int) -> [Float] {
            var out = [Float](repeating: 0, count: n)
            data.withUnsafeBytes { raw in out.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[o..<(o + 4 * n)])) } }
            o += 4 * n; return out
        }
        w1 = halfs(H1 * B * D); b1 = floats(H1); w2 = halfs(H2 * H1); b2 = floats(H2); w3 = floats(H2); b3 = floats(1)[0]
        dim = D; blocks = B; hidden1 = H1; hidden2 = H2
    }

    /// Per search: fold the question's text vector (L2-normalised, same text tower as the photo vectors) into the
    /// first layer. Cost ~2 x H1 x D multiply-adds, once.
    public func fold(question t: [Float]) -> FoldedHead {
        precondition(t.count == dim, "question vector has \(t.count) values, head expects \(dim)")
        let D = dim, W = blocks * dim
        var m = [Float](repeating: 0, count: hidden1 * D), c = [Float](repeating: 0, count: hidden1)
        var wAbs = [Float]()
        for r in 0..<hidden1 {
            let row = r * W
            var s: Float = 0
            for j in 0..<D {
                m[r * D + j] = w1[row + j] + w1[row + 2 * D + j] * t[j]
                s += w1[row + D + j] * t[j]
            }
            c[r] = s + b1[r]
            if blocks == 4 { wAbs += w1[(row + 3 * D)..<(row + 4 * D)] }
        }
        return FoldedHead(dim: D, hidden1: hidden1, hidden2: hidden2, m: m, c: c, wAbs: blocks == 4 ? wAbs : nil, text: t,
                          w2: w2, b2: b2, w3: w3, b3: b3)
    }
}

/// A head folded for one search's question: score photo vectors with it.
public struct FoldedHead: Sendable {
    public let dim: Int, hidden1: Int, hidden2: Int
    let m: [Float], c: [Float], wAbs: [Float]?, text: [Float], w2: [Float], b2: [Float], w3: [Float], b3: Float

    /// Logits of n photo vectors (rows: n x dim, row-major), written to `out[0..<n]`.
    public func logits(_ rows: UnsafePointer<Float>, count n: Int, into out: UnsafeMutablePointer<Float>,
                       gemm: GemmNT = defaultGemmNT) {
        guard n > 0 else { return }
        var h = [Float](repeating: 0, count: n * hidden1)
        var g = [Float](repeating: 0, count: n * hidden2)
        m.withUnsafeBufferPointer { mp in h.withUnsafeMutableBufferPointer { gemm(rows, n, mp.baseAddress!, hidden1, dim, $0.baseAddress!) } }
        if let wa = wAbs {                                   // B = 4: |img - txt| cannot be folded, one more product
            var a = [Float](repeating: 0, count: n * dim), h2 = [Float](repeating: 0, count: n * hidden1)
            for i in 0..<n { for j in 0..<dim { a[i * dim + j] = Swift.abs(rows[i * dim + j] - text[j]) } }
            a.withUnsafeBufferPointer { ap in wa.withUnsafeBufferPointer { wp in h2.withUnsafeMutableBufferPointer {
                gemm(ap.baseAddress!, n, wp.baseAddress!, hidden1, dim, $0.baseAddress!) } } }
            for k in 0..<(n * hidden1) { h[k] += h2[k] }
        }
        for i in 0..<n { for j in 0..<hidden1 { h[i * hidden1 + j] = Swift.max(h[i * hidden1 + j] + c[j], 0) } }
        h.withUnsafeBufferPointer { hp in w2.withUnsafeBufferPointer { wp in g.withUnsafeMutableBufferPointer {
            gemm(hp.baseAddress!, n, wp.baseAddress!, hidden2, hidden1, $0.baseAddress!) } } }
        for i in 0..<n {
            var s: Float = 0
            for k in 0..<hidden2 { s += Swift.max(g[i * hidden2 + k] + b2[k], 0) * w3[k] }
            out[i] = s + b3
        }
    }

    /// Per item: the best logit over its units (a video's frames), like lookScores; -inf for an item with no unit.
    /// `units` are the stored L2-normalised photo vectors (the index store's mapped rows are fine: read `block` rows
    /// at a time).
    public func itemLogits<V: EmbeddingRows>(units: V, unitItem: [Int], nItems: Int, block: Int = 512,
                                             gemm: GemmNT = defaultGemmNT) -> [Float] {
        var out = [Float](repeating: -.infinity, count: nItems)
        guard units.count > 0 else { return out }
        precondition(units.dim == dim, "photo vectors have \(units.dim) values, head expects \(dim)")
        var lg = [Float](repeating: 0, count: block)
        units.forEachBlock(size: block) { s, n, rows in
            lg.withUnsafeMutableBufferPointer { logits(rows.baseAddress!, count: n, into: $0.baseAddress!, gemm: gemm) }
            for r in 0..<n { let it = unitItem[s + r]; out[it] = Swift.max(out[it], lg[r]) }
        }
        return out
    }
}

/// (x - mean) / std over the finite values (population std; std 0 -> 1); non-finite values come back as -inf.
public func zScores(_ x: [Float]) -> [Float] {
    var n = 0, sum = 0.0
    for v in x where v.isFinite { n += 1; sum += Double(v) }
    guard n > 0 else { return x.map { _ in -.infinity } }
    let mean = sum / Double(n)
    var ss = 0.0
    for v in x where v.isFinite { let d = Double(v) - mean; ss += d * d }
    let sd = (ss / Double(n)).squareRoot()
    let div = sd > 0 ? sd : 1
    return x.map { $0.isFinite ? Float((Double($0) - mean) / div) : -.infinity }
}

/// Fast mode's order key: z(cosine) + z(clipped head logit), both over the same search's in-scope photos (one library,
/// as RESULTS 48's phone replay). `cosine` = lookScores (keep using those raw scores for VectorFirst's thresholds);
/// `headLogits` = FoldedHead.itemLogits for the same items. A photo without a vector (cosine -inf) stays last.
public func blendScores(cosine: [Float], headLogits: [Float]) -> [Float] {
    precondition(cosine.count == headLogits.count)
    let h: [Float] = zip(cosine, headLogits).map { c, l in
        guard c.isFinite else { return -.infinity }
        return l.isNaN ? -headLogitClip : Swift.min(Swift.max(l, -headLogitClip), headLogitClip)
    }
    let zc = zScores(cosine), zh = zScores(h)
    return zip(zc, zh).map { $0.isFinite && $1.isFinite ? $0 + $1 : -.infinity }
}
