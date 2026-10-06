import Foundation
import XCTest
@testable import FindPicsCore

struct MarginFixture: Decodable { let pA: [String: Double]; let pB: [String: Double]; let expected: [String: Int?] }

final class MarginTests: XCTestCase {
    /// Same assignments as engine.make_exclusive's rank-margin branch (Python golden values).
    func testRankMarginMatchesPython() throws {
        let url = Bundle.module.url(forResource: "margin", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(MarginFixture.self, from: Data(contentsOf: url))
        let got = rankMarginPair(pA: f.pA, pB: f.pB)
        // JSON nulls ("not clearly either") are dropped when decoding [String: Int?]: a missing key means nil
        let ids = Set(f.pA.keys).union(f.pB.keys)
        XCTAssertEqual(got.count, ids.count)
        for id in ids { XCTAssertEqual(got[id] ?? nil, f.expected[id] ?? nil, id) }
    }
}
