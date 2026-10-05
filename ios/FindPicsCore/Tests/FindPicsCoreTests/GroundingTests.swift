import Foundation
import XCTest
@testable import FindPicsCore

struct GroundCase: Decodable {
    let message: String; let history: [String]; let owner: String; let people: [String]; let today: String
    let raw: Plan; let grounded: Plan
}

final class GroundingTests: XCTestCase {
    /// Same grounded plan (all fields but the notes text) as converse.ground on 2,300 9B planner outputs.
    func testMatchesPython() throws {
        let url = Bundle.module.url(forResource: "ground", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([GroundCase].self, from: Data(contentsOf: url))
        var bad = 0
        for c in cases {
            let said = (c.history + [c.message]).joined(separator: " \n ")
            var p = c.raw
            if unanswerable(p, names: c.people + [c.owner], said: said) != nil { p = dropUnanswerable(p, said: said, names: c.people + [c.owner]) }
            let g = ground(p, message: c.message, history: c.history, today: Day(iso: c.today)!, owner: c.owner, people: c.people)
            if planJSON(g) != planJSON(c.grounded) {
                bad += 1
                if bad <= 6 { print("MISMATCH \(c.message.prefix(80))\n  swift:  \(planJSON(g).prefix(600))\n  python: \(planJSON(c.grounded).prefix(600))") }
            }
        }
        print("GROUND mismatches \(bad) of \(cases.count)")
        XCTAssertEqual(bad, 0)
    }
}
