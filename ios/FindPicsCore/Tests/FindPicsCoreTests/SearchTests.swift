import Foundation
import XCTest
@testable import FindPicsCore

struct ItemJ: Decodable { let id: String; let media: String; let taken: Double?; let localMinutes: Int?; let place: String }
struct ScopeCase: Decodable { let album: Album; let expected: [Bool] }
struct SearchFixture: Decodable {
    let items: [ItemJ]; let scope: [ScopeCase]
    let units: [[Float]]; let unitItem: [Int]; let looks: [[Float]]; let avoid: [[Float]]; let scores: [Float]
}

final class SearchTests: XCTestCase {
    func fixture() throws -> SearchFixture {
        let url = Bundle.module.url(forResource: "search", withExtension: "json", subdirectory: "Fixtures")!
        return try JSONDecoder().decode(SearchFixture.self, from: Data(contentsOf: url))
    }

    /// Same in-scope photos as engine.scope_mask (media, dates, local clock incl. midnight wrap, place words).
    func testScopeMatchesPython() throws {
        let f = try fixture()
        let items = f.items.map { FindPicsCore.LibraryItem(id: $0.id, media: $0.media, taken: $0.taken, localMinutes: $0.localMinutes, place: $0.place) }
        for c in f.scope { XCTAssertEqual(scopeMask(items, c.album), c.expected, c.album.name) }
    }

    func testLookScoresMatchPython() throws {
        let f = try fixture()
        let s = lookScores(units: f.units, unitItem: f.unitItem, nItems: f.scores.count, looks: f.looks, avoid: f.avoid)
        for (a, b) in zip(s, f.scores) { XCTAssertEqual(a, b, accuracy: 1e-5) }
    }
}
