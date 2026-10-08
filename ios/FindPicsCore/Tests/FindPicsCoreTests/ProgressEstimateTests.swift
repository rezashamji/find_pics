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
