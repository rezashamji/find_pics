import XCTest
@testable import FindPicsCore

final class BurstsTests: XCTestCase {
    /// Same cases as tests/test_bursts.py.
    func testSameMomentOnly() {
        let e: [[Float]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        func norm(_ v: [Float]) -> [Float] { let s = sqrt(v.map { $0 * $0 }.reduce(0, +)); return v.map { $0 / s } }
        let v = [e[0], norm([0.99, 0.1, 0]), e[0], e[2], e[0]]
        let t0 = 1_682_935_200.0   // 2023-05-01 10:00 UTC
        let taken: [Double?] = [t0, t0 + 120, t0 + 3 * 3600, t0 + 60, nil]
        let g = burstIds(vectors: v, taken: taken)
        XCTAssertEqual(g[0], g[1])          // same scene, 2 min apart
        XCTAssertNotEqual(g[2], g[0])       // same scene 3 h later
        XCTAssertNotEqual(g[3], g[0])       // different scene, same minute
        XCTAssertFalse(g[0..<4].contains(g[4]))   // undated: never stacked
        XCTAssertEqual(g[0], 0)
    }
}
