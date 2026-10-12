import XCTest
@testable import FindPicsCore

final class JudgeCutoffTests: XCTestCase {
    func testSubjectFormIsStrictOnVisionJudge() {
        for q in ["Is this a photo of a mountain?", "is this a picture of a sunset?", "Is this an image of a church?",
                  "  Is this a photo of flowers?", "Is this photo of a beach?"] {
            XCTAssertTrue(isSubjectPhotoQuestion(q), q)
            XCTAssertEqual(judgeCutoff(question: q, strictSubjects: true), 0.999, q)
            XCTAssertEqual(judgeCutoff(question: q, strictSubjects: false), 0.7, q)
        }
    }
    /// RESULTS 47: beach's wording keeps the strict subject cut; food's subject question 0.9999; sunset unchanged.
    func testSceneWordExceptions() {
        XCTAssertEqual(judgeCutoff(question: beachSubjectQuestion, strictSubjects: true), 0.999)
        XCTAssertEqual(judgeCutoff(question: "is a beach clearly visible in this photo", strictSubjects: true), 0.999)
        XCTAssertEqual(judgeCutoff(question: beachSubjectQuestion, strictSubjects: false), 0.7)
        XCTAssertTrue(isSubjectPhotoQuestion(beachSubjectQuestion))      // same search kind (VectorFirst eligibility)
        for q in ["Is this a photo of food?", "Is this a picture of food", "  is this an image of food?"] {
            XCTAssertTrue(isFoodSubjectQuestion(q), q)
            XCTAssertEqual(judgeCutoff(question: q, strictSubjects: true), 0.9999, q)
            XCTAssertEqual(judgeCutoff(question: q, strictSubjects: false), 0.7, q)
        }
        for q in ["Is this a photo of a sunset?", "Is this a photo of a food truck?", "Is this a photo of a beach?",
                  "Is this a photo of food and drinks?"] {
            XCTAssertFalse(isFoodSubjectQuestion(q), q)
            XCTAssertEqual(judgeCutoff(question: q, strictSubjects: true), 0.999, q)
        }
        for q in ["Is there a real dog anywhere in this photo (not a drawing)?", "Is there food anywhere in this photo?",
                  "Is a dog clearly visible in this photo?"] {
            XCTAssertEqual(judgeCutoff(question: q, strictSubjects: true), 0.7, q)
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

final class IndexLocalCopyTests: XCTestCase {
    func testIndexTakesBigEnoughLocalCopyJudgeDoesNot() {
        XCTAssertTrue(indexAcceptsLocalCopy(.indexForeground, gotW: 512, gotH: 384))
        XCTAssertTrue(indexAcceptsLocalCopy(.indexBackground, gotW: 300, gotH: 448))
        XCTAssertFalse(indexAcceptsLocalCopy(.indexForeground, gotW: 320, gotH: 240))   // thumbnail: download if allowed
        XCTAssertFalse(indexAcceptsLocalCopy(.judge, gotW: 4000, gotH: 3000))           // judge rules unchanged
        XCTAssertTrue(indexAcceptsLocalCopy(.localOnly, gotW: 480, gotH: 360))      // the index's first, local pass
        XCTAssertFalse(indexAcceptsLocalCopy(.localOnly, gotW: 120, gotH: 90))      // grid thumbnail: not enough
    }
}
