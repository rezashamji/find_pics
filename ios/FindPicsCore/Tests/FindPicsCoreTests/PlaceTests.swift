import Foundation
import XCTest
@testable import FindPicsCore

struct PlaceItem: Decodable { let id: String; let media: String; let place: String }
struct PlaceCase: Decodable { let album: Album; let filtered: Album; let placed: Album }
struct PlaceFixture: Decodable { let items: [PlaceItem]; let cases: [PlaceCase] }

final class PlaceTests: XCTestCase {
    /// Same results as converse.filter_to_place / place_or_look on a synthetic library.
    func testPlaceRulesMatchPython() throws {
        let url = Bundle.module.url(forResource: "places", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(PlaceFixture.self, from: Data(contentsOf: url))
        let items = f.items.map { LibraryItem(id: $0.id, media: $0.media, taken: nil, localMinutes: nil, place: $0.place) }
        for c in f.cases {
            XCTAssertEqual(filterToPlace(items, c.album), c.filtered, "filter \(c.album)")
            XCTAssertEqual(placeOrLook(items, c.album), c.placed, "place \(c.album)")
        }
    }
}
