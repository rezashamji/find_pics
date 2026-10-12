import Foundation
import XCTest
@testable import FindPicsCore

struct FollowUpCase: Decodable { let message: String; let current: Plan; let grounded: Plan; let edited: Plan }

final class FollowUpTests: XCTestCase {
    /// Same plan (all fields but the notes text) as converse.follow_up_edits (partial undo, "add X too" as OR, "show me
    /// all" resets the narrowing) on every follow-up of the 2,300 grounding cases + the complex-bench dev failure kinds.
    func testMatchesPython() throws {
        let url = Bundle.module.url(forResource: "followup", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([FollowUpCase].self, from: Data(contentsOf: url))
        var bad = 0, changed = 0
        for c in cases {
            let e = followUpEdits(c.grounded, current: c.current, message: c.message)
            if planJSON(e) != planJSON(c.grounded) { changed += 1 }
            if planJSON(e) != planJSON(c.edited) {
                bad += 1
                if bad <= 6 { print("FOLLOWUP MISMATCH \(c.message.prefix(80))\n  swift:  \(planJSON(e).prefix(600))\n  python: \(planJSON(c.edited).prefix(600))") }
            }
        }
        print("FOLLOWUP mismatches \(bad) of \(cases.count) (changed by the rules: \(changed))")
        XCTAssertEqual(bad, 0)
        XCTAssertGreaterThanOrEqual(changed, 7)
    }

    func testNoFutureDatesAndEmbeddedYearsAgo() {
        func run(_ tp: String, _ df: String, _ dt: String, _ msg: String, _ today: String = "2014-07-08") -> (String?, String?) {
            var a = Album(name: "x"); a.judgeQuestion = "Is there a concert?"; a.timePhrase = tp; a.dateFrom = df; a.dateTo = dt
            let g = ground(Plan(albums: [a]), message: msg, history: [], today: Day(iso: today)!, owner: "me", people: [])
            return (g.albums[0].dateFrom, g.albums[0].dateTo)
        }
        XCTAssertTrue(run("November 27th", "2014-11-27", "2014-11-28", "the photos from the concert on November 27th") == ("2013-11-27", "2013-11-28"))
        XCTAssertTrue(run("the November 27th show", "2014-11-27", "2014-11-28", "photos from the November 27th show") == ("2013-11-27", "2013-11-28"))
        XCTAssertTrue(run("2015", "2015-01-01", "2016-01-01", "photos from 2015") == ("2015-01-01", "2016-01-01"))
        XCTAssertTrue(run("the day we hiked in Denali two years ago", "2010-12-25", "2010-12-26",
                          "photos from the day we hiked in Denali two years ago", "2012-12-25") == ("2010-01-01", "2011-01-01"))
        XCTAssertTrue(run("October three years ago", "2007-10-01", "2007-11-01", "photos from the wedding in October three years ago",
                          "2010-10-08") == ("2007-10-01", "2007-11-01"))
    }
}
