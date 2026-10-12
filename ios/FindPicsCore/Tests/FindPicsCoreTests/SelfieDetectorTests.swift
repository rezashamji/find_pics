import Testing
@testable import FindPicsCore

// One dot product decides "selfie" for the ~170k photos whose camera tag we will never read (M50).
@Suite struct SelfieDetectorTests {
    @Test func dotProductMatchesByHand() {
        let d = SelfieDetector(w: [1, -2, 0.5], b: -0.25)
        #expect(abs(d.logit([1, 1, 2]) - (1 - 2 + 1 - 0.25)) < 1e-6)
    }

    @Test func wrongWidthIsNeverASelfie() {
        // A vector from a different encoder must not silently score as something: the index outlives model changes.
        #expect(SelfieDetector(w: [1, 1], b: 0).logit([1, 1, 1]) == -.infinity)
    }

    @Test func perfectSeparationReadsAsPerfect() {
        let rows = (0..<10).map { (score: Float($0) + 5, isFront: true) }
            + (0..<10).map { (score: Float($0) - 10, isFront: false) }
        let c = checkSelfieDetector(rows)
        #expect(c.auc == 1.0)
        #expect(c.recallAtHalf == 1.0)
        #expect(c.falsePositiveRateAtHalf == 0.0)
        #expect(c.recallAt95 == 1.0)
    }

    @Test func noSeparationReadsAsChance() {
        let rows = (0..<20).map { (score: Float($0 % 2), isFront: $0 < 10) }
        let c = checkSelfieDetector(rows)
        #expect(abs(c.auc - 0.5) < 1e-9)
    }

    @Test func thresholdChosenForPrecisionNotRecall() {
        // 3 front photos high, 1 front low among many backs: the 95% cut must sit above the backs and so lose
        // the low one - precision is what makes an unjudged answer safe to show.
        var rows: [(score: Float, isFront: Bool)] = [(10, true), (9, true), (8, true), (0, true)]
        rows += (0..<20).map { (score: Float($0) * 0.1, isFront: false) }
        let c = checkSelfieDetector(rows)
        #expect(c.threshold95 != nil)
        #expect(c.threshold95! > 1.9)
        #expect(abs(c.recallAt95 - 0.75) < 1e-9)
    }

    @Test func noLabelsOnOneSideIsNotANumber() {
        let c = checkSelfieDetector([(1, true), (2, true)])
        #expect(c.auc.isNaN)
        #expect(c.threshold95 == nil)
    }
}
