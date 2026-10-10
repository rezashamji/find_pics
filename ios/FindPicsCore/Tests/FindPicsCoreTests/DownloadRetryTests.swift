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

final class StaleDownloadTempTests: XCTestCase {
    func testOnlyOldDownloadTemps() {
        XCTAssertTrue(isStaleDownloadTemp(name: "CFNetworkDownload_a1B2c3.tmp", ageSeconds: 7_200))
        XCTAssertFalse(isStaleDownloadTemp(name: "CFNetworkDownload_a1B2c3.tmp", ageSeconds: 600))   // maybe active
        XCTAssertFalse(isStaleDownloadTemp(name: "index_store.tmp", ageSeconds: 99_999))           // not ours to touch
        XCTAssertFalse(isStaleDownloadTemp(name: "CFNetworkDownload_x.part", ageSeconds: 99_999))
    }
}
