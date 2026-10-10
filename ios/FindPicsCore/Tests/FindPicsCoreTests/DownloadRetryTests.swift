import XCTest
@testable import FindPicsCore

final class DownloadRetryTests: XCTestCase {
    func testRetriesTransientErrorsWithBackoff() {
        XCTAssertEqual(downloadRetryDelay(attempt: 1, urlErrorCode: -1005), 2)          // MAC 10-10: connection lost
        XCTAssertEqual(downloadRetryDelay(attempt: 2, urlErrorCode: -1005), 3)
        XCTAssertEqual(downloadRetryDelay(attempt: 30, urlErrorCode: -1001), 60)        // capped at a minute
        XCTAssertNil(downloadRetryDelay(attempt: 40, urlErrorCode: -1005))              // gives up eventually
        XCTAssertNil(downloadRetryDelay(attempt: 1, urlErrorCode: -1011))               // bad server response: show it
        XCTAssertNil(downloadRetryDelay(attempt: 1, urlErrorCode: nil))                 // not a network error
    }
}
