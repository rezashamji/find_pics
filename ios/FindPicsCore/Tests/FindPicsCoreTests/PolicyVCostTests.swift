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
