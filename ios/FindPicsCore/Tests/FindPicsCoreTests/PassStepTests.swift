import XCTest
@testable import FindPicsCore

/// Locks in the two bugs that cost a night (JOURNAL 10-10): "could not run" and "everything left already failed"
/// must never be read as "this pass is finished".
final class PassStepTests: XCTestCase {
    func testWorkRemainingRunsAnotherChunk() {
        XCTAssertEqual(passStep(left: 600, checked: 10_000, total: 79_615), .more)
    }

    /// THE 7-MINUTE DEATH: left == 0 only because every remaining photo failed once this launch.
    func testZeroLeftWithWorkRemainingRetriesInsteadOfFinishing() {
        XCTAssertEqual(passStep(left: 0, checked: 46_000, total: 79_615), .retryAfterFailures)
    }

    /// A chunk that could not run at all (screen off, or downloads not allowed) is not a finished pass.
    func testCouldNotRunRetries() {
        XCTAssertEqual(passStep(left: nil, checked: 46_000, total: 79_615), .retryAfterFailures)
        XCTAssertEqual(passStep(left: nil, checked: 79_615, total: 79_615), .retryAfterFailures)
    }

    func testOnlyRealCompletionFinishes() {
        XCTAssertEqual(passStep(left: 0, checked: 79_615, total: 79_615), .finished)
    }

    /// Reza's 39 unreadable photos: checked can never reach total, so the caller must not WAIT on .finished -
    /// it keeps retrying, and the caller hands the chain to the next pass regardless.
    func testUnreachableTotalKeepsRetryingAndNeverFinishes() {
        XCTAssertEqual(passStep(left: 0, checked: 79_576, total: 79_615), .retryAfterFailures)
    }
}
