import XCTest
@testable import FindPicsCore

final class ProgressEstimateTests: XCTestCase {
    func testEstimate() {
        XCTAssertNil(progressEstimate(done: 3, total: 100, startDone: 0, elapsed: 600))      // too few items
        XCTAssertNil(progressEstimate(done: 50, total: 100, startDone: 0, elapsed: 30))      // too early
        XCTAssertEqual(progressEstimate(done: 22, total: 39005, startDone: 7, elapsed: 247), // MAC 06:22-06:26
                       "about 7 days left (~4 per minute)")
        XCTAssertEqual(progressEstimate(done: 600, total: 1200, startDone: 0, elapsed: 600), "about 10 min left (~60 per minute)")
        XCTAssertEqual(progressEstimate(done: 10, total: 200, startDone: 0, elapsed: 3600), "about 19 h left (~10 per hour)")
    }
}

final class SearchableSoFarTests: XCTestCase {
    func testMonotonicAcrossLaunches() {
        // MAC 10-08: session ended "3,446 of 38,870", relaunch read "104 of 35,269" on a 187,120 library
        let before = searchableSoFar(libraryCount: 187_120, passTotal: 38_870, passDone: 3_446)
        let after = searchableSoFar(libraryCount: 187_120, passTotal: 35_269, passDone: 104)
        XCTAssertEqual(before, 151_696); XCTAssertEqual(after, 151_955); XCTAssertGreaterThanOrEqual(after, before)
        XCTAssertEqual(searchableSoFar(libraryCount: 10, passTotal: 0, passDone: 0), 10)
        XCTAssertEqual(searchableSoFar(libraryCount: 0, passTotal: 5, passDone: 1), 0)
    }
}
