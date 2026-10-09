import Foundation
import XCTest
@testable import FindPicsCore

struct PairCase: Decodable {
    let name: String
    let pA: [String: Double]
    let pB: [String: Double]
    let eventOf: [String: String]?
    let expected: [String: Int?]?
}

final class PairingTests: XCTestCase {
    /// Same answers as the Python engine (_split_pair) on public eras (Pratt / Hill / Rogen), synthetic sets, and the
    /// tied-event regression cases (RESULTS 39; eval/pairing_fixtures.py regenerates the file).
    func testMatchesPythonEngine() throws {
        let url = Bundle.module.url(forResource: "pairing", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([PairCase].self, from: Data(contentsOf: url))
        XCTAssertEqual(cases.count, 10)
        for c in cases {
            let got = splitPair(pA: c.pA, pB: c.pB, eventOf: c.eventOf)
            if c.expected == nil { XCTAssertNil(got, c.name); continue }
            XCTAssertNotNil(got, c.name)
            XCTAssertEqual(got?.count, c.expected?.count, c.name)
            for (id, want) in c.expected! { XCTAssertEqual(got?[id] ?? nil, want, "\(c.name) \(id)") }
        }
    }

    /// One long event's photos share one median (a tie). The quartile start alone collapsed onto it; the two-means start
    /// must win and put (nearly) every upper-group photo in album 0 and every lower-group photo in album 1.
    func testTiedEventDoesNotCollapseTheSplit() throws {
        let url = Bundle.module.url(forResource: "pairing", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([PairCase].self, from: Data(contentsOf: url))
        let c = try XCTUnwrap(cases.first { $0.name == "tied_spike_quartile_collapse" })
        let got = try XCTUnwrap(splitPair(pA: c.pA, pB: c.pB, eventOf: c.eventOf))
        let ids = c.pA.keys.sorted()                      // s000..s391: 0..<150 lower group, 180..<392 upper group
        let lower = ids[0..<150], upper = ids[180...]
        XCTAssertGreaterThanOrEqual(upper.filter { got[$0] == .some(0) }.count, upper.count - 5)
        XCTAssertGreaterThanOrEqual(lower.filter { got[$0] == .some(1) }.count, lower.count - 5)
        XCTAssertTrue(upper.allSatisfy { got[$0] != .some(1) } && lower.allSatisfy { got[$0] != .some(0) })
    }

    func testTwoMeansIsTheExactBestSplit() {
        let m = twoMeans([0, 0.1, 0.2, 5, 5.1, 9])
        XCTAssertEqual(m[0], 0.1, accuracy: 1e-12); XCTAssertEqual(m[1], (5 + 5.1 + 9) / 3, accuracy: 1e-12)
        XCTAssertEqual(twoMeans([0, 0, 0]), [0, 0])
    }
}
