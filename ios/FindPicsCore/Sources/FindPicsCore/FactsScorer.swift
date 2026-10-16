// RESULTS 68 (3): the millisecond scorers of the facts cascade in pure Swift - the towervec HEAD (facts vector f + question
// vector q from the towervec text encoder, PE-Core-B-16 image vector v + text vector t) and vecB (v, t only; photos without a
// facts line, policy V). Port of eval/facts_scorer/scorers.py Head; weights exported by eval/cascade_port/heads.py
// (.fphead). Checked against PyTorch fp32 on CPU (Tests: tiny random heads always; the real heads on two dev libraries with
// FP_HEADS_DIR).
//   x = [LN_f(f), LN_q(q), LN_f(f) * LN_q(q)] (towervec only) + [32 v, 32 t, (32 v) * (32 t)]
//   P = sigmoid(W3 gelu(W2 gelu(W1 x + b1) + b2) + b3), gelu = exact (erf), LN eps 1e-5
// Per search the first layer FOLDS (it is linear in the photo side once q, t are fixed): FoldedScorerHead holds
//   A = [W_f + W_fq diag(q~) | W_v + W_vt diag(32 t)] (512 x (384 + 1024)), c = W_q q~ + W_t 32 t + b1
// so one photo costs 512 x 1,408 + 512 x 128 + 128 multiply-adds (towervec; vecB 512 x 1,024 + ...).
import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

public enum FactsScorerError: Error { case badFile(String) }

public struct ScorerHead: Sendable {
    public let kind: String            // "towervec" | "vec"
    public let dTxt: Int, dImg: Int, hid: Int, hid2: Int
    public let imgScale: Float, eps: Float
    let lnF: [Float], lnFb: [Float], lnQ: [Float], lnQb: [Float]
    let w1: [Float], b1: [Float], w2: [Float], b2: [Float], w3: [Float], b3: Float
    var dIn: Int { (dTxt > 0 ? 3 * dTxt : 0) + 3 * dImg }

    public init(data: Data) throws {
        let bytes = [UInt8](data)
        guard bytes.count > 12, String(bytes: bytes[0..<8], encoding: .ascii) == "FPHEAD01" else { throw FactsScorerError.badFile("magic") }
        let hl = Int(UInt32(bytes[8]) | UInt32(bytes[9]) << 8 | UInt32(bytes[10]) << 16 | UInt32(bytes[11]) << 24)
        guard let hdr = try JSONSerialization.jsonObject(with: Data(bytes[12..<(12 + hl)])) as? [String: Any],
              let tensors = hdr["tensors"] as? [[Any]] else { throw FactsScorerError.badFile("header") }
        var off = 12 + hl
        var t = [String: [Float]]()
        for e in tensors {
            guard let name = e[0] as? String, let shape = e[1] as? [Int] else { throw FactsScorerError.badFile("tensor") }
            let n = shape.reduce(1, *)
            guard off + 4 * n <= bytes.count else { throw FactsScorerError.badFile("short") }
            var a = [Float](repeating: 0, count: n)
            a.withUnsafeMutableBytes { dst in bytes.withUnsafeBytes { src in
                dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[off..<(off + 4 * n)])) } }
            #if _endian(big)
            a = a.map { Float(bitPattern: $0.bitPattern.byteSwapped) }
            #endif
            t[name] = a; off += 4 * n
        }
        kind = hdr["kind"] as? String ?? ""
        dTxt = hdr["d_txt"] as? Int ?? 0; dImg = hdr["d_img"] as? Int ?? 0
        hid = hdr["hid"] as? Int ?? 0; hid2 = hdr["hid2"] as? Int ?? 0
        imgScale = Float(hdr["img_scale"] as? Double ?? 32); eps = Float(hdr["ln_eps"] as? Double ?? 1e-5)
        lnF = t["ln_f.weight"] ?? []; lnFb = t["ln_f.bias"] ?? []; lnQ = t["ln_q.weight"] ?? []; lnQb = t["ln_q.bias"] ?? []
        w1 = t["net.0.weight"] ?? []; b1 = t["net.0.bias"] ?? []; w2 = t["net.2.weight"] ?? []; b2 = t["net.2.bias"] ?? []
        w3 = t["net.4.weight"] ?? []; b3 = t["net.4.bias"]?.first ?? 0
        guard w1.count == hid * dIn, w2.count == hid2 * hid, w3.count == hid2 else { throw FactsScorerError.badFile("shapes") }
    }

    public init(contentsOf url: URL) throws { try self.init(data: Data(contentsOf: url)) }

    static func gelu(_ x: Float) -> Float { 0.5 * x * (1 + Float(erf(Double(x) / 2.0.squareRoot()))) }
    static func sigmoid(_ x: Float) -> Float { 1 / (1 + Float(exp(-Double(x)))) }

    func layerNorm(_ x: [Float], _ w: [Float], _ b: [Float]) -> [Float] {
        let n = Float(x.count)
        let mu = x.reduce(0, +) / n
        let va = x.reduce(0) { $0 + ($1 - mu) * ($1 - mu) } / n
        let r = 1 / (va + eps).squareRoot()
        return x.indices.map { (x[$0] - mu) * r * w[$0] + b[$0] }
    }

    func tail(_ h1: [Float]) -> Float {      // gelu -> Linear 128 -> gelu -> Linear 1 (logit)
        let a = h1.map(Self.gelu)
        var out = b3
        for j in 0..<hid2 {
            var s = b2[j]
            let row = j * hid
            for k in 0..<hid { s += w2[row + k] * a[k] }
            out += w3[j] * Self.gelu(s)
        }
        return out
    }

    /// reference forward, unfolded (one photo, one question): logit
    public func logit(f: [Float]?, q: [Float]?, v: [Float], t: [Float]) -> Float {
        var x = [Float]()
        x.reserveCapacity(dIn)
        if dTxt > 0, let f = f, let q = q {
            let fn = layerNorm(f, lnF, lnFb), qn = layerNorm(q, lnQ, lnQb)
            x += fn; x += qn; x += zip(fn, qn).map { $0 * $1 }
        }
        let vs = v.map { $0 * imgScale }, ts = t.map { $0 * imgScale }
        x += vs; x += ts; x += zip(vs, ts).map { $0 * $1 }
        var h1 = b1
        for i in 0..<hid {
            var s: Float = 0
            let row = i * dIn
            for k in 0..<dIn { s += w1[row + k] * x[k] }
            h1[i] += s
        }
        return tail(h1)
    }

    public func p(f: [Float]?, q: [Float]?, v: [Float], t: [Float]) -> Float { Self.sigmoid(logit(f: f, q: q, v: v, t: t)) }

    /// the per-search folded first layer for one question (q = towervec question vector, nil for vec; t = PE text vector)
    public func fold(q: [Float]?, t: [Float]) -> FoldedScorerHead {
        let dPhoto = (dTxt > 0 ? dTxt : 0) + dImg
        var A = [Float](repeating: 0, count: hid * dPhoto)
        var c = b1
        let qn = (dTxt > 0 && q != nil) ? layerNorm(q!, lnQ, lnQb) : []
        let ts = t.map { $0 * imgScale }
        let oImg = dTxt > 0 ? 3 * dTxt : 0
        for i in 0..<hid {
            let row = i * dIn, arow = i * dPhoto
            if dTxt > 0 {
                for k in 0..<dTxt {
                    A[arow + k] = w1[row + k] + w1[row + 2 * dTxt + k] * qn[k]
                    c[i] += w1[row + dTxt + k] * qn[k]
                }
            }
            let ao = dTxt > 0 ? dTxt : 0
            for k in 0..<dImg {
                // x_v = 32 v enters as itself and times 32 t; fold the image scale into A
                A[arow + ao + k] = (w1[row + oImg + k] + w1[row + oImg + 2 * dImg + k] * ts[k]) * imgScale
                c[i] += w1[row + oImg + dImg + k] * ts[k]
            }
        }
        return FoldedScorerHead(head: self, A: A, c: c, dPhoto: dPhoto)
    }
}

/// One question's folded head: P for many photos at the cost of one matrix product.
public struct FoldedScorerHead: Sendable {
    let head: ScorerHead
    let A: [Float], c: [Float], dPhoto: Int

    /// photo side input: [LN_f(f) (towervec only), v]
    func photoInput(f: [Float]?, v: [Float]) -> [Float] {
        head.dTxt > 0 ? head.layerNorm(f!, head.lnF, head.lnFb) + v : v
    }

    public func logit(f: [Float]?, v: [Float]) -> Float {
        let x = photoInput(f: f, v: v)
        var h1 = c
        for i in 0..<head.hid {
            var s: Float = 0
            let row = i * dPhoto
            for k in 0..<dPhoto { s += A[row + k] * x[k] }
            h1[i] += s
        }
        return head.tail(h1)
    }

    public func p(f: [Float]?, v: [Float]) -> Float { ScorerHead.sigmoid(logit(f: f, v: v)) }

    /// many photos at once: X = n photo inputs, row-major n x dPhoto -> n P. On Apple platforms the first layer is one
    /// vDSP_mmul (AMX); elsewhere plain loops.
    public func p(batch fs: [[Float]]?, vs: [[Float]]) -> [Float] {
        let n = vs.count
        if n == 0 { return [] }
        var X = [Float](); X.reserveCapacity(n * dPhoto)
        for i in 0..<n { X += photoInput(f: fs?[i], v: vs[i]) }
        var H = [Float](repeating: 0, count: n * head.hid)     // n x hid = X (n x d) * A^T (d x hid)
        #if canImport(Accelerate)
        var At = [Float](repeating: 0, count: dPhoto * head.hid)
        for i in 0..<head.hid { for k in 0..<dPhoto { At[k * head.hid + i] = A[i * dPhoto + k] } }
        vDSP_mmul(X, 1, At, 1, &H, 1, vDSP_Length(n), vDSP_Length(head.hid), vDSP_Length(dPhoto))
        #else
        for r in 0..<n {
            let xo = r * dPhoto
            for i in 0..<head.hid {
                var s: Float = 0
                let ao = i * dPhoto
                for k in 0..<dPhoto { s += A[ao + k] * X[xo + k] }
                H[r * head.hid + i] = s
            }
        }
        #endif
        return (0..<n).map { r in
            ScorerHead.sigmoid(head.tail((0..<head.hid).map { H[r * head.hid + $0] + c[$0] }))
        }
    }
}
