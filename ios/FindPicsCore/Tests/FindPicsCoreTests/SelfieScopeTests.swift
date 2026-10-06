import XCTest
@testable import FindPicsCore

final class SelfieScopeTests: XCTestCase {
    /// Same rule as engine.scope_mask (tests/test_engine.py): with camera tags in the library, a selfie album drops
    /// back-camera photos and screenshots; with no tags at all it is a no-op.
    func testFrontCameraScope() {
        func item(_ id: String, _ cam: String, shot: Bool = false) -> LibraryItem {
            LibraryItem(id: id, media: "photo", taken: nil, localMinutes: nil, place: nil, camera: cam, isScreenshot: shot)
        }
        var a = Album(name: "s"); a.camera = "front"
        let tagged = [item("a", "front"), item("b", "back"), item("c", ""), item("d", "", shot: true)]
        XCTAssertEqual(scopeMask(tagged, a), [true, false, true, false])
        let untagged = [item("a", ""), item("b", ""), item("c", "", shot: true)]
        XCTAssertEqual(scopeMask(untagged, a), [true, true, true])
        XCTAssertEqual(scopeMask(tagged, Album(name: "plain")), [true, true, true, true])
    }
}
