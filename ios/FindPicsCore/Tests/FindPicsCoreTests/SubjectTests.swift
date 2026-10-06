import Foundation
import XCTest
@testable import FindPicsCore

struct SubjectFixture: Decodable { let units: [[Float]]; let unitItem: [Int]; let refs: [[Float]]; let k: Int; let scores: [Float] }

final class SubjectTests: XCTestCase {
    /// Same ranking as converse._dba + mean over references + best unit per item (Python golden values).
    func testSubjectScoresMatchPython() throws {
        let url = Bundle.module.url(forResource: "subject", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(SubjectFixture.self, from: Data(contentsOf: url))
        let s = subjectScores(units: f.units, unitItem: f.unitItem, nItems: f.scores.count, refs: f.refs, k: f.k)
        XCTAssertEqual(s.count, f.scores.count)
        for (a, b) in zip(s, f.scores) { XCTAssertEqual(a, b, accuracy: 1e-4) }
    }
}
