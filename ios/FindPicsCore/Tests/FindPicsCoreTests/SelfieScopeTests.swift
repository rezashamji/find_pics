import XCTest
@testable import FindPicsCore

final class SelfieScopeTests: XCTestCase {
    func item(_ id: String, _ cam: String, shot: Bool = false) -> FindPicsCore.LibraryItem {
        FindPicsCore.LibraryItem(id: id, media: "photo", taken: nil, localMinutes: nil, place: nil, camera: cam, isScreenshot: shot)
    }

    /// Same rule as engine.scope_mask (tests/test_engine.py): with camera tags in the library, a selfie album drops
    /// back-camera photos and screenshots; with no tags at all it is a no-op. A photo whose camera is UNKNOWN stays
    /// in scope, because the judge still decides.
    func testFrontCameraScope() {
        var a = Album(name: "s"); a.camera = "front"; a.judgeQuestion = "Is this a selfie?"
        let tagged = [item("a", "front"), item("b", "back"), item("c", ""), item("d", "", shot: true)]
        XCTAssertEqual(scopeMask(tagged, a), [true, false, true, false])
        let untagged = [item("a", ""), item("b", ""), item("c", "", shot: true)]
        XCTAssertEqual(scopeMask(untagged, a), [true, true, true])
        XCTAssertEqual(scopeMask(tagged, Album(name: "plain")), [true, true, true, true])
    }

    /// A bare "selfies" has NO judge question (10-11: the camera tag is the answer and the judge cost 2-5 s a photo
    /// for nothing). Then nothing downstream ever looks at the photo, so an unknown camera must not be included -
    /// the answer is what the camera itself vouches for. NOTE: this case has no Python counterpart yet.
    func testUnjudgedSelfieScopeNeedsTheTag() {
        var a = Album(name: "selfies"); a.camera = "front"      // judgeQuestion stays nil
        let tagged = [item("a", "front"), item("b", "back"), item("c", ""), item("d", "front", shot: true)]
        XCTAssertEqual(scopeMask(tagged, a), [true, false, false, false])
        // and still a no-op on a library that carries no camera tags at all
        XCTAssertEqual(scopeMask([item("a", ""), item("b", "")], a), [true, true])
    }
}
