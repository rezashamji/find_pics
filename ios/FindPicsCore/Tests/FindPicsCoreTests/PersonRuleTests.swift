import Foundation
import XCTest
@testable import FindPicsCore

struct LookCase: Decodable { let q: String?; let look: Bool }
struct LookFixture: Decodable { let cases: [LookCase] }

/// Person albums: a LOOK keeps the top half of the person's own photos, a FACT keeps the judge's yes photos
/// (engine.person_keep / is_look_question; RESULTS 49c).
final class PersonRuleTests: XCTestCase {
    /// The same cases tests/test_engine.py checks, incl. every person-album question of the 30-conversation gate plans.
    func testLookVsFactMatchesPythonFixture() throws {
        let url = Bundle.module.url(forResource: "look_questions", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode(LookFixture.self, from: Data(contentsOf: url)).cases
        XCTAssertGreaterThanOrEqual(cases.count, 60)
        for c in cases { XCTAssertEqual(isLookQuestion(c.q), c.look, c.q ?? "nil") }
    }

    /// Same numbers as test_person_fact_keeps_judge_yes_only_and_look_keeps_top_half (Python).
    func testFactKeepsJudgeYesOnlyAndLookKeepsTopHalf() {
        let ps = [0.95, 0.80, 0.60, 0.30, 0.20, 0.10, 0.05, 0.02]
        let p = Dictionary(uniqueKeysWithValues: ps.enumerated().map { ("i\($0.offset)", $0.element) })
        // a FACT: 0.60 / 0.30 are in the person's top half but the judge said no (RESULTS 49b pink helmet)
        XCTAssertEqual(personAlbumFound(pYes: p, question: "Is the person in the red box wearing a pink helmet?", accept: 0.7),
                       ["i0", "i1"])
        // a LOOK: the top half of the person's own photos, highest P first
        XCTAssertEqual(personAlbumFound(pYes: p, question: "Is the person in the red box looking heavier?", accept: 0.7),
                       ["i0", "i1", "i2", "i3"])
        // a person the judge rates low on everything: the look keeps the same photos, a fact keeps none
        let low = p.mapValues { $0 * 0.3 }
        XCTAssertEqual(personAlbumFound(pYes: low, question: "Is this a photo of a person with a heavy build?", accept: 0.7),
                       ["i0", "i1", "i2", "i3"])
        XCTAssertEqual(personAlbumFound(pYes: low, question: "Is this a selfie?", accept: 0.7), [])
        XCTAssertEqual(personAlbumFound(pYes: [:], question: "Is this a selfie?", accept: 0.7), [])
    }
}
