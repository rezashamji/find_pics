// Whole-library scale: the bounded "Who is X?" grouping (faceGroups with cap) and subject search on a candidate set
// (subjectScoresCandidates), against golden files from their Python references (eval/face_groups_sampled.py,
// eval/dba_candidates.py), plus the exact-order fallback matrix product.
import Foundation
@testable import FindPicsCore
import XCTest

final class ScaleTests: XCTestCase {
    func fixture(_ name: String) throws -> [String: Any] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures/" + name, withExtension: "json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }
    func floats(_ x: Any?) -> [[Float]] { (x as! [[Double]]).map { $0.map { Float($0) } } }

    /// Off Apple the blocked product is bit-identical to the plain dot loop (FaceMatch.dot), so Linux results equal the
    /// old [[Float]] code exactly.
    func testGemmFallbackIsThePlainLoop() {
        var r = LCG(s: 9)
        for (m, n, d) in [(1, 1, 1), (5, 3, 7), (9, 11, 64), (4, 2, 3)] {
            let A = (0..<(m * d)).map { _ in r.next() }, B = (0..<(n * d)).map { _ in r.next() }
            var C = [Float](repeating: 0, count: m * n)
            A.withUnsafeBufferPointer { a in B.withUnsafeBufferPointer { b in C.withUnsafeMutableBufferPointer { c in
                MatrixMath.gemmNT(a.baseAddress!, m: m, b.baseAddress!, n: n, d: d, c.baseAddress!)
            } } }
            for i in 0..<m { for j in 0..<n {
                let want = dot(Array(A[(i * d)..<((i + 1) * d)]), Array(B[(j * d)..<((j + 1) * d)]))
                #if canImport(Accelerate)
                XCTAssertEqual(C[i * n + j], want, accuracy: 1e-5)
                #else
                XCTAssertEqual(C[i * n + j].bitPattern, want.bitPattern)
                #endif
            } }
        }
    }

    func testFaceGroupsCapMatchesPython() throws {
        let f = try fixture("face_groups_cap")
        let emb = floats(f["emb"])
        let names = f["item"] as! [String]
        let num = Dictionary(uniqueKeysWithValues: Array(Set(names)).sorted().enumerated().map { ($1, $0) })
        let item = names.map { num[$0]! }
        let px = (f["px"] as! [Double]).map { Float($0) }, det = (f["det"] as! [Double]).map { Float($0) }
        let taken = (f["taken"] as! [Any]).map { $0 as? Double }
        let accept = Float(f["accept"] as! Double)
        let want = f["groups"] as! [String: [[String: Any]]]
        func check(_ got: [FaceGroup], _ key: String) {
            let w = want[key]!
            XCTAssertEqual(got.map(\.faces), w.map { $0["faces"] as! [Int] }, key)
            XCTAssertEqual(got.map(\.rep), w.map { $0["rep"] as! Int }, key)
        }
        check(faceGroups(faces: emb, faceItem: item, facePx: px, det: det, accept: accept), "full")
        check(faceGroups(faces: emb, faceItem: item, facePx: px, det: det, accept: accept, cap: 300, faceTaken: taken,
                         newestFraction: 0.5), "cap300_newest0.5")
        check(faceGroups(faces: emb, faceItem: item, facePx: px, det: det, accept: accept, cap: 300, faceTaken: taken,
                         newestFraction: 0), "cap300_newest0")
        let rows = (0..<emb.count).filter { px[$0] >= 40 && det[$0] >= 0.7 }
        XCTAssertGreaterThan(rows.count, 300)                         // the cap bites: assignment exercised
        XCTAssertEqual(faceSampleRows(rows, taken: taken, cap: 300, newestFraction: 0.5), f["sample300"] as! [Int])
        // under the cap the bounded grouping IS the full one, also through faceGroupsPreferChecked
        let all = [Bool](repeating: true, count: emb.count)
        XCTAssertEqual(faceGroupsPreferChecked(faces: emb, faceItem: item, facePx: px, det: det, checked: all, accept: accept),
                       faceGroups(faces: emb, faceItem: item, facePx: px, det: det, accept: accept))
    }

    func testSubjectCandidatesMatchPython() throws {
        let f = try fixture("subject_candidates")
        let units = floats(f["units"]), refs = floats(f["refs"])
        let n = units.count, ui = Array(0..<n)
        func close(_ a: [Float], _ b: [Double], _ what: String) {
            XCTAssertEqual(a.count, b.count)
            let worst = zip(a, b).map { abs(Double($0) - $1) }.max()!
            XCTAssertLessThan(worst, 2e-5, what)
            let ra = a.indices.sorted { a[$0] > a[$1] }.prefix(50), rb = b.indices.sorted { b[$0] > b[$1] }.prefix(50)
            XCTAssertGreaterThanOrEqual(Set(ra).intersection(rb).count, 49, what)
        }
        close(subjectScoresBlocked(units: units, unitItem: ui, nItems: n, refs: refs), f["exact"] as! [Double], "exact")
        close(subjectScores(units: units, unitItem: ui, nItems: n, refs: refs), f["exact"] as! [Double], "reference")
        for K in [200, 600] {
            close(subjectScoresCandidates(units: units, unitItem: ui, nItems: n, refs: refs, candidates: K), f["cand\(K)"] as! [Double], "K\(K)")
        }
        // all in scope = no scope; a library within K = exact
        XCTAssertEqual(subjectScoresCandidates(units: units, unitItem: ui, nItems: n, refs: refs, candidates: 200, inScope: ui.map { _ in true }),
                       subjectScoresCandidates(units: units, unitItem: ui, nItems: n, refs: refs, candidates: 200))
        XCTAssertEqual(subjectScoresCandidates(units: units, unitItem: ui, nItems: n, refs: refs, candidates: n),
                       subjectScoresBlocked(units: units, unitItem: ui, nItems: n, refs: refs))
        // a scope: its best photos are among the smoothed ones
        let scope = ui.map { $0 % 3 == 0 }
        let s = subjectScoresCandidates(units: units, unitItem: ui, nItems: n, refs: refs, candidates: 100, inScope: scope)
        let q = subjectScoresCandidates(units: units, unitItem: ui, nItems: n, refs: refs, candidates: 0)   // q only
        let smoothed = ui.filter { s[$0] != q[$0] }
        XCTAssertEqual(smoothed.count, smoothed.filter { scope[$0] }.count)
        XCTAssertGreaterThan(smoothed.count, 90)
    }
}
