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
        let items = texts.enumerated().map { FindPicsCore.LibraryItem(id: "i\($0.offset)", media: "photo", taken: nil, localMinutes: nil, place: $0.element) }
        func mask(_ its: [FindPicsCore.LibraryItem], _ place: String) -> [Bool] { var a = Album(name: "p"); a.place = place; return scopeMask(its, a) }
        XCTAssertEqual(mask(Array(items.prefix(3)), "New York City"), [true, false, false])   // "city" optional: no full match
        XCTAssertEqual(mask(items, "New York City"), [false, false, false, true])              // a full match wins
        XCTAssertEqual(mask(items, "Boston"), [false, true, false, false])
        XCTAssertEqual(mask(items, "Paris"), [false, false, false, false])
        XCTAssertEqual(placeWords(Array(items.prefix(3)), "New York City"), ["new", "york"])
    }

    /// engine.place_parts / place_mask (tests/test_converse.py::test_place_list_matches_any_of_its_places, RESULTS 51)
    func testPlaceListMatchesAnyOfItsPlaces() {
        let texts = ["Catedral de Burgos, Burgos, Castile and León, Spain", "Plaza de Regla, León, Castile and León, Spain",
                     "Shibuya, Tokyo, Japan", "Namba, Osaka, Japan", nil, "Port of Spain, Trinidad and Tobago"]
        let items = texts.enumerated().map { FindPicsCore.LibraryItem(id: "i\($0.offset)", media: "photo", taken: nil, localMinutes: nil, place: $0.element) }
        func mask(_ place: String) -> [Bool] { var a = Album(name: "p"); a.place = place; return scopeMask(items, a) }
        XCTAssertEqual(placeParts("Burgos and León"), ["Burgos", "León"])
        XCTAssertEqual(placeParts("Tokyo & Osaka"), ["Tokyo", "Osaka"])
        XCTAssertEqual(placeParts("Tokyo or Osaka"), ["Tokyo", "Osaka"])
        XCTAssertEqual(placeParts("New York City"), ["New York City"])
        XCTAssertEqual(mask("Burgos and León"), [true, true, false, false, false, false])
        XCTAssertEqual(mask("Tokyo and Osaka"), [false, false, true, true, false, false])
        XCTAssertEqual(mask("Tokyo or Osaka"), [false, false, true, true, false, false])
        XCTAssertEqual(mask("Trinidad and Tobago"), [false, false, false, false, false, true])
        XCTAssertEqual(mask("Burgos"), [true, false, false, false, false, false])
    }

    /// converse.earlier_year_if_empty (tests/test_converse.py::test_yearless_date_with_no_photo_moves_back_to_a_year_with_photos)
    func testYearlessDateWithNoPhotoMovesBackToAYearWithPhotos() {
        let rows: [(String, String)] = [("2012-06-24", "Cortina d'Ampezzo, Belluno, Italy"), ("2013-06-02", "Paris, France"),
                                        ("2014-06-10", "Paris, France"), ("2012-11-27", "Milan, Italy"), ("2012-12-31", "Milan, Italy")]
        let items = rows.enumerated().map { FindPicsCore.LibraryItem(id: "i\($0.offset)", media: "photo",
            taken: Double(Day(iso: $0.element.0)!.serial) * 86400 + 36000, localMinutes: nil, place: $0.element.1) }
        func album(_ from: String, _ to: String, _ tp: String, _ place: String? = nil) -> Album {
            var a = Album(name: "x"); a.dateFrom = from; a.dateTo = to; a.timePhrase = tp; a.place = place; return a
        }
        let (b, k) = earlierYearIfEmpty(items, album("2014-06-01", "2014-07-01", "June", "Cortina"))
        XCTAssertEqual([b.dateFrom, b.dateTo], ["2012-06-01", "2012-07-01"]); XCTAssertEqual(k, 2)
        let paris = album("2014-06-01", "2014-07-01", "June", "Paris")
        XCTAssertEqual(earlierYearIfEmpty(items, paris).0, paris); XCTAssertEqual(earlierYearIfEmpty(items, paris).1, 0)
        let (n, kn) = earlierYearIfEmpty(items, album("2013-11-27", "2013-11-28", "November 27th"))
        XCTAssertEqual(n.dateFrom, "2012-11-27"); XCTAssertEqual(kn, 1)
        XCTAssertEqual(earlierYearIfEmpty(items, album("2013-12-01", "2014-03-01", "last winter", "Cortina")).1, 0)
        let (e, ke) = earlierYearIfEmpty(items, album("2013-12-31", "2014-01-01", "New Year's Eve"))
        XCTAssertEqual([e.dateFrom, e.dateTo], ["2012-12-31", "2013-01-01"]); XCTAssertEqual(ke, 1)
        for tp in ["this June", "June 2014", "two years ago", "last week", "yesterday"] {
            let a = album("2014-06-01", "2014-07-01", tp, "Cortina")
            XCTAssertEqual(earlierYearIfEmpty(items, a).0, a, tp); XCTAssertEqual(earlierYearIfEmpty(items, a).1, 0, tp)
        }
    }

    /// converse.ground (tests/test_converse.py::test_photos_from_a_moment_keep_every_photo_of_it, RESULTS 51)
    func testPhotosFromAMomentKeepEveryPhotoOfIt() {
        func run(_ msg: String, _ jq: String) -> Album {
            var a = Album(name: "x"); a.looks = ["a selfie"]; a.judgeQuestion = jq; a.window = "same_day"
            a.anchor = Step(looks: ["a pink life vest"], judgeQuestion: "Is this a selfie of a person wearing a pink life vest?")
            return ground(Plan(albums: [a]), message: msg, history: [], today: Day(iso: "2014-01-28")!, owner: "me", people: []).albums[0]
        }
        let a = run("The photos from the day I took that selfie in the pink life vest", "Is this a selfie?")
        XCTAssertNil(a.judgeQuestion); XCTAssertEqual(a.looks, []); XCTAssertNil(a.camera); XCTAssertNotNil(a.anchor)
        XCTAssertEqual(a.window, "same_day")
        XCTAssertNil(run("Pictures from the day we were at Everest Base Camp with the prayer flags.",
                         "Is this a photo taken at Everest Base Camp featuring prayer flags?").judgeQuestion)
        XCTAssertNil(run("photos from the day we hiked in Denali two years ago", "Is this a photo taken during a hike in Denali?").judgeQuestion)
        XCTAssertNotNil(run("photos from the day I saw that red tractor", "Is there a dog in this photo?").judgeQuestion)
        XCTAssertNotNil(run("photos from the trip where I saw the giraffes, only the elephants", "Is there an elephant?").judgeQuestion)
        XCTAssertNotNil(run("the lake photos from the mountain trip", "Is this a photo of a lake?").judgeQuestion)
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
