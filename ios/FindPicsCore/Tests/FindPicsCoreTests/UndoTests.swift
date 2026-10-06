import XCTest
@testable import FindPicsCore

final class UndoTests: XCTestCase {
    /// Same as tests/test_converse.py::test_partial_undo_keeps_the_rest_of_the_exclusion.
    func testPartialUndoKeepsTheRest() {
        var a = Album(name: "bread"); a.judgeQuestion = "Is there bread?"; a.excludeQuestion = "Is there a sandwich or a burger in this photo?"
        let cur = Plan(albums: [a])
        var b = Album(name: "bread"); b.judgeQuestion = "Is there bread?"
        XCTAssertEqual(keepPartialUndo(Plan(albums: [b]), current: cur, message: "actually keep the sandwiches").albums[0].excludeQuestion,
                       "Is there a burger in this photo?")
        XCTAssertNil(keepPartialUndo(Plan(albums: [b]), current: cur, message: "show more").albums[0].excludeQuestion)
    }
}
