import Foundation
import XCTest
@testable import FindPicsCore

/// RESULTS 71 / M102 wired: the table is read off, not interpolated; the coverage level quoted is the greatest one
/// AT OR BELOW the phone's; below 10% coverage the note says nothing about recall; D5 is on where the cluster
/// measured precision < .90; and a request that fits several categories is quoted as the most cautious of them.
final class QuickAnswerNoteTests: XCTestCase {

    func testBelowTenPercentCoverageSaysNothingAboutRecall() {
        // Reza's phone today: 1,885 written lines of 148,451 photos in scope.
        let cov = 1885.0 / 148_451.0
        XCTAssertLessThan(cov, coverageFloor)
        for k in RequestKind.allCases {
            XCTAssertNil(njRecall(k, coverage: cov), "\(k) quoted a number at \(cov) coverage")
            XCTAssertNil(njRecallSentence(k, coverage: cov))
        }
    }

    func testQuotesTheNearestMeasuredLevelAtOrBelowTheCoverage() {
        // dev mean per request, objects: 10% .72, 25% .73, 50% .76, 100% .80.
        XCTAssertEqual(njRecall(.objects, coverage: 0.10), 0.72)
        XCTAssertEqual(njRecall(.objects, coverage: 0.24), 0.72)   // not the 25% row
        XCTAssertEqual(njRecall(.objects, coverage: 0.25), 0.73)
        XCTAssertEqual(njRecall(.objects, coverage: 0.99), 0.76)   // not the 100% row
        XCTAssertEqual(njRecall(.objects, coverage: 1.00), 0.80)
        // Never quotes a level above the phone's: the quoted value is monotone non-decreasing in coverage only
        // where the measurement is, so the check is on the LEVEL chosen, not on the value.
        for c in stride(from: 0.10, through: 1.00, by: 0.01) {
            let r = njRecall(.objects, coverage: c)!
            let level = measuredCoverages.last { c >= $0 }!
            XCTAssertEqual(r, njRecall(.objects, coverage: level))
        }
    }

    func testTheD1HeadlineRowIsWhatTheClusterReported() throws {
        let expected: [RequestKind: Double] = [
            .objects: 0.80, .attributes: 0.70, .personLook: 0.67, .timePlace: 0.83,
            .multiHop: 0.54, .vagueOnePhoto: 0.61, .twoAlbums: 0.78, .followUps: 0.66, .all: 0.70,
        ]
        for (k, v) in expected {
            XCTAssertEqual(try XCTUnwrap(njRecall(k, coverage: 1.0)), v, accuracy: 1e-9, "\(k)")
        }
    }

    func testSentenceWording() {
        XCTAssertEqual(njRecallSentence(.timePlace, coverage: 1.0),
                       "Usually finds about 8 in 10 of what a full check finds for this kind of question.")
        XCTAssertEqual(njRecallSentence(.vagueOnePhoto, coverage: 0.10),
                       "Usually finds about 4 in 10 of what a full check finds for this kind of question.")
        XCTAssertEqual(njRecallSentence(.all, coverage: 0.50),
                       "Usually finds about 7 in 10 of what a full check finds for this kind of question.")
    }

    func testD5IsOffExactlyWhereThePrecisionWasAtLeastNinety() {
        // dev precision mean per request: time/place .906, two albums .900 -> not offered; everything else below.
        XCTAssertFalse(verifyShownOffered(.timePlace))
        XCTAssertFalse(verifyShownOffered(.twoAlbums))
        for k in RequestKind.allCases where k != .timePlace && k != .twoAlbums {
            XCTAssertTrue(verifyShownOffered(k), "\(k) should offer verify-the-shown")
        }
    }

    // MARK: - which kind a plan is

    private func album(_ name: String = "a") -> Album { Album(name: name) }

    func testAPlainObjectQuestionIsObjects() {
        var a = album(); a.judgeQuestion = "Is this a photo of a dog?"
        let p = Plan(albums: [a])
        XCTAssertEqual(requestKindCandidates(p, isFollowUp: false), [.objects])
        XCTAssertEqual(requestKind(p, isFollowUp: false, coverage: 1.0), .objects)
    }

    func testAColourQuestionIsAttributes() {
        var a = album(); a.judgeQuestion = "Is this a photo of a red car?"
        XCTAssertTrue(requestKindCandidates(Plan(albums: [a]), isFollowUp: false).contains(.attributes))
    }

    func testAPhotoLookQuestionIsPersonLook() {
        var a = album(); a.judgeQuestion = "Is this a black-and-white photo?"
        XCTAssertTrue(requestKindCandidates(Plan(albums: [a]), isFollowUp: false).contains(.personLook))
    }

    func testDatesAndPlacesAreTimePlace() {
        var a = album(); a.judgeQuestion = "Is this a photo of a dog?"; a.place = "Lisbon"
        XCTAssertTrue(requestKindCandidates(Plan(albums: [a]), isFollowUp: false).contains(.timePlace))
        var b = album(); b.judgeQuestion = "Is this a photo of a dog?"; b.dateFrom = "2024-01-01"
        XCTAssertTrue(requestKindCandidates(Plan(albums: [b]), isFollowUp: false).contains(.timePlace))
    }

    func testTwoAlbumsAndFollowUps() {
        var a = album("one"); a.judgeQuestion = "Is this a photo of a dog?"
        var b = album("two"); b.judgeQuestion = "Is this a photo of a cat?"
        XCTAssertTrue(requestKindCandidates(Plan(albums: [a, b]), isFollowUp: false).contains(.twoAlbums))
        XCTAssertTrue(requestKindCandidates(Plan(albums: [a]), isFollowUp: true).contains(.followUps))
    }

    func testVagueOnePhoto() {
        var a = album(); a.want = "best"; a.maxItems = 1
        XCTAssertTrue(requestKindCandidates(Plan(albums: [a]), isFollowUp: false).contains(.vagueOnePhoto))
    }

    /// The point of the min rule: a FOLLOW-UP asking for TWO ALBUMS is quoted as follow-ups (.66), the lower of the
    /// two at 100% coverage - not as two albums (.78).
    func testSeveralCategoriesTakeTheLowestMeasuredRecall() {
        var a = album("one"); a.judgeQuestion = "Is this a photo of a dog?"
        var b = album("two"); b.judgeQuestion = "Is this a photo of a cat?"
        let p = Plan(albums: [a, b])
        XCTAssertEqual(requestKindCandidates(p, isFollowUp: true), [.twoAlbums, .followUps])
        XCTAssertEqual(requestKind(p, isFollowUp: true, coverage: 1.0), .followUps)
    }

    /// And the ordering really does depend on coverage, which is why `requestKind` takes it: multi-hop is ABOVE
    /// attributes at 10% coverage (.69 vs .62) and BELOW it at 100% (.54 vs .70).
    func testTheMostCautiousCategoryDependsOnCoverage() {
        var a = album(); a.judgeQuestion = "Is this a photo of a red car?"     // colour -> attributes
        a.anchor = Step(looks: ["a dog"])                                       // a chained step -> multi-hop
        let p = Plan(albums: [a])
        XCTAssertEqual(requestKindCandidates(p, isFollowUp: false), [.attributes, .multiHop])
        XCTAssertEqual(requestKind(p, isFollowUp: false, coverage: 0.10), .attributes)
        XCTAssertEqual(requestKind(p, isFollowUp: false, coverage: 1.00), .multiHop)
    }

    func testAnEmptyPlanIsAll() {
        XCTAssertEqual(requestKindCandidates(Plan(albums: []), isFollowUp: false), [.all])
        XCTAssertEqual(requestKind(Plan(albums: []), isFollowUp: false, coverage: 1.0), .all)
    }
}
