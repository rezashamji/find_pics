import XCTest
@testable import FindPicsCore

final class JudgeCutoffTests: XCTestCase {
    func testSubjectFormIsStrictOnVisionJudge() {
        for q in ["Is this a photo of food?", "is this a picture of a sunset?", "Is this an image of a church?",
                  "  Is this a photo of flowers?", "Is this photo of a beach?"] {
            XCTAssertTrue(isSubjectPhotoQuestion(q), q)
            XCTAssertEqual(judgeCutoff(question: q, strictSubjects: true), 0.99, q)
            XCTAssertEqual(judgeCutoff(question: q, strictSubjects: false), 0.7, q)
        }
    }
    func testOtherFormsKeepDefault() {
        for q in ["Is there a real dog anywhere in this photo (not a drawing)?", "Is there bread anywhere in this photo?",
                  "Is the person in the red box looking heavier?", "Was this photo taken at a beach?",
                  "Is this a selfie?"] {
            XCTAssertFalse(isSubjectPhotoQuestion(q), q)
            XCTAssertEqual(judgeCutoff(question: q, strictSubjects: true), 0.7, q)
        }
    }
}
