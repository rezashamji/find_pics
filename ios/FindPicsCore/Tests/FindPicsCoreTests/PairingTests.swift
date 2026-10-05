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
    /// Same answers as the Python engine (_split_pair) on public eras (Pratt / Hill / Rogen) and synthetic sets.
    func testMatchesPythonEngine() throws {
        let url = Bundle.module.url(forResource: "pairing", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([PairCase].self, from: Data(contentsOf: url))
        XCTAssertEqual(cases.count, 7)
        for c in cases {
            let got = splitPair(pA: c.pA, pB: c.pB, eventOf: c.eventOf)
            if c.expected == nil { XCTAssertNil(got, c.name); continue }
            XCTAssertNotNil(got, c.name)
            XCTAssertEqual(got?.count, c.expected?.count, c.name)
            for (id, want) in c.expected! { XCTAssertEqual(got?[id] ?? nil, want, "\(c.name) \(id)") }
        }
    }
}
