import Foundation
import XCTest
@testable import FindPicsCore

struct TodCase: Decodable { let text: String; let want: String? }
struct RelCase: Decodable { let phrase: String; let today: String; let want: [String]? }
struct DateFixtures: Decodable { let time_of_day: [TodCase]; let relative: [RelCase] }

final class DatesTests: XCTestCase {
    func fixtures() throws -> DateFixtures {
        let url = Bundle.module.url(forResource: "dates", withExtension: "json", subdirectory: "Fixtures")!
        return try JSONDecoder().decode(DateFixtures.self, from: Data(contentsOf: url))
    }

    /// Same clock range as converse.resolve_time_of_day on every fuzz-corpus request (1,457 texts).
    func testTimeOfDayMatchesPython() throws {
        var bad = 0
        for c in try fixtures().time_of_day where resolveTimeOfDay(c.text) != c.want {
            bad += 1; if bad <= 10 { XCTFail("\(c.text) -> \(resolveTimeOfDay(c.text) ?? "nil"), python \(c.want ?? "nil")") }
        }
        XCTAssertEqual(bad, 0)
    }

    /// Same dates as converse.resolve_relative on 90 phrases x 6 "today"s (incl. a leap day and Thanksgiving).
    func testRelativeDatesMatchPython() throws {
        var bad = 0
        for c in try fixtures().relative {
            let got = resolveRelative(c.phrase, today: Day(iso: c.today)!).map { [$0.0, $0.1] }
            if got != c.want { bad += 1; if bad <= 10 { XCTFail("\(c.phrase) @\(c.today) -> \(String(describing: got)), python \(String(describing: c.want))") } }
        }
        XCTAssertEqual(bad, 0)
    }

    func testDayArithmetic() {
        XCTAssertEqual(Day(2024, 2, 29).adding(1).description, "2024-03-01")
        XCTAssertEqual(Day(2026, 10, 4).weekday, 6)       // Sunday
        XCTAssertEqual(Day(1970, 1, 1).serial, 0)
        XCTAssertEqual(Day.daysIn(2024, 2), 29)
    }
}
