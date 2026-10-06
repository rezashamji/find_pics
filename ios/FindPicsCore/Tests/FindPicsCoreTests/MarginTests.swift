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
        XCTAssertEqual(got.count, f.expected.count)
        for (id, want) in f.expected { XCTAssertEqual(got[id] ?? nil, want, id) }
    }
}
