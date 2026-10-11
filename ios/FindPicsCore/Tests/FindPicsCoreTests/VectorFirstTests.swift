import XCTest
@testable import FindPicsCore

final class VectorFirstTests: XCTestCase {
    func album(_ q: String?, looks: [String] = ["a dog"]) -> Album {
        var a = Album(name: "x"); a.looks = looks; a.judgeQuestion = q; return a
    }
    let obj = "Is there a real dog anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"

    func testKthLargest() {
        XCTAssertEqual(kthLargest([0.1, 0.5, 0.3, 0.9, 0.7], 1), 0.9)
        XCTAssertEqual(kthLargest([0.1, 0.5, 0.3, 0.9, 0.7], 3), 0.5)
        XCTAssertEqual(kthLargest([0.1, 0.5, 0.3, 0.9, 0.7], 5), 0.1)
        XCTAssertNil(kthLargest([0.1, 0.5], 3))
        XCTAssertEqual(kthLargest([0.2, .nan, 0.4, 0.4, 0.1], 2), 0.4)
        var xs = (0..<5000).map { Float($0) / 5000 }; xs.shuffle()
        XCTAssertEqual(kthLargest(xs, 20), Float(4980) / 5000)
    }

    func testManyClearMatchesGoToVectors() {
        let s = [Float](repeating: vectorEnoughScore + 0.01, count: 25) + [Float](repeating: 0.12, count: 1000)
        let d = firstScreenDecision(album: album(obj), scores: s)
        XCTAssertEqual(d.source, .vectors)
        XCTAssertEqual(d.kthScore, vectorEnoughScore + 0.01)
        XCTAssertEqual(firstScreenDecision(album: album("Is this a photo of a beach?", looks: ["a beach"]), scores: s).source,
                       .vectors)
        XCTAssertEqual(firstScreenDecision(album: album(nil), scores: s).source, .vectors)
    }

    func testFewMatchesGoToJudge() {
        // 19 strong photos are not a full screen: the 20th is a background photo
        let s = [Float](repeating: 0.25, count: 19) + [Float](repeating: 0.12, count: 1000)
        let d = firstScreenDecision(album: album(obj), scores: s)
        XCTAssertEqual(d.source, .judge)
        XCTAssertEqual(d.kthScore, 0.12)
        XCTAssertEqual(likelyAtOnce(album: album(obj), scores: s).count, 19)    // still shown at once, per photo
        XCTAssertEqual(firstScreenDecision(album: album(obj), scores: [0.3, 0.3]).source, .judge)
    }

    func testUnmeasuredKindsAlwaysGoToJudge() {
        let s = [Float](repeating: 0.3, count: 100)
        var a = album(obj); a.person = "me"
        XCTAssertEqual(firstScreenDecision(album: a, scores: s).source, .judge)
        a = album(obj); a.withPeople = ["Anna"]
        XCTAssertEqual(firstScreenDecision(album: a, scores: s).source, .judge)
        a = album(obj); a.camera = "front"
        XCTAssertEqual(firstScreenDecision(album: a, scores: s).source, .judge)
        a = album(obj); a.excludeQuestion = "Is it raining?"
        XCTAssertEqual(firstScreenDecision(album: a, scores: s).source, .judge)
        a = album(obj); a.anchor = Step(looks: ["a cake"])
        XCTAssertEqual(firstScreenDecision(album: a, scores: s).source, .judge)
        XCTAssertEqual(firstScreenDecision(album: album("Is this a selfie?", looks: []), scores: s).source, .judge)
        XCTAssertEqual(firstScreenDecision(album: album("Is the person in the red box looking heavier?"), scores: s).source,
                       .judge)
        XCTAssertEqual(likelyAtOnce(album: album("Is the person in the red box looking heavier?"), scores: s), [])
    }

    func testLikelyAtOnceOrderAndCap() {
        let s: [Float] = [0.30, 0.10, 0.25, 0.25, 0.40] + [Float](repeating: 0.5, count: 30)
        let got = likelyAtOnce(album: album(obj), scores: s)
        XCTAssertEqual(got.count, vectorFirstScreen)
        XCTAssertEqual(got.first, 5)                       // ties keep library order
        XCTAssertEqual(likelyAtOnce(album: album(obj), scores: Array(s.prefix(5)), threshold: 0.25), [4, 0, 2, 3])
    }

    func testQuestionForms() {
        XCTAssertTrue(isAnywhereObjectQuestion(obj))
        XCTAssertTrue(isAnywhereObjectQuestion("Is there bread anywhere in this photo?"))
        XCTAssertFalse(isAnywhereObjectQuestion("Is this photo taken at night?"))
        XCTAssertFalse(isAnywhereObjectQuestion("Is there a dog on a beach?"))
    }
}
