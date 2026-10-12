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
        let items = f.items.map { FindPicsCore.LibraryItem(id: $0.id, media: $0.media, taken: nil, localMinutes: nil, place: $0.place) }
        for c in f.cases {
            XCTAssertEqual(filterToPlace(items, c.album), c.filtered, "filter \(c.album)")
            XCTAssertEqual(placeOrLook(items, c.album), c.placed, "place \(c.album)")
        }
    }
}

final class PlaceKindTests: XCTestCase {
    /// engine.place_mask (tests/test_converse.py::test_place_kind_words_optional_only_when_the_full_name_matches_nothing):
    /// "New York City" vs addresses that say "New York County, New York" (complex bench dev 10-12).
    func testPlaceKindWordsOptionalOnlyWhenTheFullNameMatchesNothing() {
        let texts = ["Brooklyn Bridge, Manhattan, New York County, New York, 10038, United States",
                     "Faneuil Hall, Boston, Suffolk County, Massachusetts, United States", nil,
                     "Long Island City, Queens County, New York, United States"]
        let items = texts.enumerated().map { LibraryItem(id: "i\($0.offset)", media: "photo", taken: nil, localMinutes: nil, place: $0.element) }
        func mask(_ its: [LibraryItem], _ place: String) -> [Bool] { var a = Album(name: "p"); a.place = place; return scopeMask(its, a) }
        XCTAssertEqual(mask(Array(items.prefix(3)), "New York City"), [true, false, false])   // "city" optional: no full match
        XCTAssertEqual(mask(items, "New York City"), [false, false, false, true])              // a full match wins
        XCTAssertEqual(mask(items, "Boston"), [false, true, false, false])
        XCTAssertEqual(mask(items, "Paris"), [false, false, false, false])
        XCTAssertEqual(placeWords(Array(items.prefix(3)), "New York City"), ["new", "york"])
    }

    /// converse.ground (tests/test_converse.py::test_judge_question_naming_the_albums_place_is_dropped)
    func testJudgeQuestionNamingTheAlbumsPlaceIsDropped() {
        var a = Album(name: "NYC"); a.looks = ["a city skyline"]; a.judgeQuestion = "Is this a photo of New York City?"; a.place = "New York City"
        var b = Album(name: "Boston"); b.judgeQuestion = "Is this a photo of Boston?"; b.place = "Boston"
        var c = Album(name: "x"); c.judgeQuestion = "Is this a photo of Mount Fuji?"
        let g = ground(Plan(albums: [a, b, c]), message: "My photos of New York City versus Boston, and Mount Fuji", history: [],
                       today: Day(iso: "2014-06-06")!, owner: "me", people: [])
        XCTAssertEqual(g.albums.map { $0.judgeQuestion }, [nil, nil, "Is this a photo of Mount Fuji?"])
        XCTAssertEqual(g.albums.map { $0.place }, ["New York City", "Boston", nil])
    }
}
