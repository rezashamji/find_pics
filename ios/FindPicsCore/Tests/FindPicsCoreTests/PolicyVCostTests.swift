// What the judge-free answer COSTS over a whole library. The screen says "at once", so the cost of saying it has to be
// a measured number, not an adjective. This is a Mac/cluster measurement of the same code path the phone runs
// (Accelerate vDSP on Apple silicon); the phone's own number comes from the MILLI log line, and it will be slower.
// Gated on FP_POLICYV_COST so the ordinary suite stays quick.
import Foundation
import XCTest
@testable import FindPicsCore

/// n rows of the same vector, generated on demand: times the maths without 608 MB of [[Float]].
struct RepeatRows: EmbeddingRows {
    let count: Int, dim: Int, row: [Float]
    func withRows<R>(_ start: Int, _ n: Int, _ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        var buf = [Float](); buf.reserveCapacity(n * dim)
        for _ in 0..<n { buf += row }
        return try buf.withUnsafeBufferPointer(body)
    }
}

final class PolicyVCostTests: XCTestCase {
    func testWholeLibraryCost() throws {
        guard ProcessInfo.processInfo.environment["FP_POLICYV_COST"] != nil else { return }
        let (tw, vb) = try PolicyVTests.heads()
        var r = PolicyVTests.LCG(s: 11)
        let t = (0..<vb.dImg).map { _ in r.f() }
        let fv = vb.fold(q: nil, t: t), ft = tw.fold(q: (0..<tw.dTxt).map { _ in r.f() }, t: t)
        let f = (0..<tw.dTxt).map { _ in r.f() }
        let n = 148_428                                   // the scope of Reza's "food photos" search
        let rows = RepeatRows(count: n, dim: vb.dImg, row: (0..<vb.dImg).map { _ in r.f() })
        let photoRow = Array(0..<n)
        for (name, described) in [("vecB only (today: ~1% described, so ~all of it)", 0.0), ("every photo described", 1.0)] {
            let t0 = Date()
            _ = policyVScores(rows: rows, photoRow: photoRow, photos: n, towervec: ft,
                              factsVector: { Double($0 % 100) < described * 100 ? f : nil }, vec: fv)
            let s = Date().timeIntervalSince(t0)
            print(String(format: "POLICYV COST %@: %d photos in %.2f s (%.1f us each)", name, n, s, s / Double(n) * 1e6))
        }
    }
}

/// WHERE THE 7.6 s GOES (M95(b)): the cluster's arithmetic says the matrix product is a fraction of a second, so
/// 51 us a photo has to be mostly something else. This times the three parts of p(batch:) separately rather than
/// guessing - the last three times I explained a cost on this project by reasoning I was wrong.
extension PolicyVCostTests {
    func testWhereTheJudgeFreePassSpendsItsTime() throws {
        guard ProcessInfo.processInfo.environment["FP_POLICYV_COST"] != nil else { return }
        let (_, vb) = try PolicyVTests.heads()
        var r = PolicyVTests.LCG(s: 5)
        let t = (0..<vb.dImg).map { _ in r.f() }
        let fv = vb.fold(q: nil, t: t)
        let n = 16_384, d = vb.dImg, hid = vb.hid
        let vs = (0..<n).map { _ in (0..<d).map { _ in r.f() } }

        var t0 = Date()
        var X = [Float](); X.reserveCapacity(n * d)
        for i in 0..<n { X += fv.photoInput(f: nil, v: vs[i]) }
        let buildX = Date().timeIntervalSince(t0)

        t0 = Date()
        var H = [Float](repeating: 0, count: n * hid)
        var At = [Float](repeating: 0, count: d * hid)
        for i in 0..<hid { for k in 0..<d { At[k * hid + i] = fv.A[i * d + k] } }
        let transpose = Date().timeIntervalSince(t0)

        t0 = Date()
        MatrixMath.gemmNT(X, m: n, fv.A, n: hid, d: d, &H)
        let mmul = Date().timeIntervalSince(t0)

        t0 = Date()
        var out = [Float](repeating: 0, count: n)
        for row in 0..<n { out[row] = ScorerHead.sigmoid(vb.tail((0..<hid).map { H[row * hid + $0] + fv.c[$0] })) }
        let tail = Date().timeIntervalSince(t0)

        let per = { (s: Double) in String(format: "%.1f us/photo", s / Double(n) * 1e6) }
        print("POLICYV SPLIT (reference per-photo tail) \(n) photos: buildX \(per(buildX)), At transpose \(per(transpose)), "
              + "mmul \(per(mmul)), tail \(per(tail)); total \(per(buildX + transpose + mmul + tail))")
        XCTAssertEqual(out.count, n)
    }
}

/// The tail WAS 98% of the pass, so it gets split too, for the same reason: it has two candidate costs - 640 erf calls
/// a photo (gelu on 512 hidden units, then on 128) and 128 x 512 = 65,536 SCALAR multiply-adds a photo - and the fix
/// is different for each.
extension PolicyVCostTests {
    func testWhereTheTailSpendsItsTime() throws {
        guard ProcessInfo.processInfo.environment["FP_POLICYV_COST"] != nil else { return }
        let (_, vb) = try PolicyVTests.heads()
        var r = PolicyVTests.LCG(s: 9)
        let n = 16_384, hid = vb.hid, hid2 = vb.hid2
        let H = (0..<n).map { _ in (0..<hid).map { _ in r.f() } }

        var t0 = Date()
        var sink: Float = 0
        for row in 0..<n { let a = H[row].map(ScorerHead.gelu); sink += a[0] }
        let gelu1 = Date().timeIntervalSince(t0)

        let a = H[0].map(ScorerHead.gelu)
        t0 = Date()
        for _ in 0..<n {
            for j in 0..<hid2 {
                var s: Float = 0
                let row = j * hid
                for k in 0..<hid { s += vb.w2[row + k] * a[k] }
                sink += s
            }
        }
        let layer2 = Date().timeIntervalSince(t0)

        t0 = Date()
        for _ in 0..<n { for j in 0..<hid2 { sink += ScorerHead.gelu(Float(j)) } }
        let gelu2 = Date().timeIntervalSince(t0)

        let per = { (s: Double) in String(format: "%.1f us/photo", s / Double(n) * 1e6) }
        print("TAIL SPLIT (reference per-photo tail) \(n) photos: gelu on \(hid) hidden \(per(gelu1)), layer-2 \(hid2)x\(hid) scalar MACs "
              + "\(per(layer2)), gelu on \(hid2) \(per(gelu2)) [sink \(sink)]")
    }
}
