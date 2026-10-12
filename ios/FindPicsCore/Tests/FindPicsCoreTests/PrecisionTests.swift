import Testing
@testable import FindPicsCore

// The sample-audit math (Completeness.swift). Reza cannot eye 41,541 results, so every precision number the app
// shows comes from a random hand-checked sample; these lock in that it stays honest at the edges.
@Suite struct PrecisionTests {
    @Test func allRightStillHasDoubt() {
        let p = precisionFromSample(labels: Array(repeating: true, count: 20), found: 41_541)
        #expect(p.point == 1)
        #expect(p.upper == 1)
        #expect(p.lower > 0.80 && p.lower < 0.90)   // 20/20 -> ~83% lower bound, not "100% certain"
    }

    @Test func noneRightStillHasDoubt() {
        let p = precisionFromSample(labels: Array(repeating: false, count: 20), found: 100)
        #expect(p.point == 0)
        #expect(p.lower == 0)
        #expect(p.upper > 0.10 && p.upper < 0.20)
    }

    @Test func halfRight() {
        let p = precisionFromSample(labels: (0..<20).map { $0 % 2 == 0 }, found: 1_000)
        #expect(p.right == 10 && p.checked == 20)
        #expect(abs(p.point - 0.5) < 1e-9)
        #expect(p.lower < 0.5 && p.upper > 0.5)
        #expect(p.rightInSet == 500)
    }

    @Test func moreChecksNarrowIt() {
        let a = precisionFromSample(labels: (0..<20).map { $0 % 2 == 0 }, found: 1_000)
        let b = precisionFromSample(labels: (0..<100).map { $0 % 2 == 0 }, found: 1_000)
        #expect((b.upper - b.lower) < (a.upper - a.lower))
    }

    @Test func nothingCheckedSaysSo() {
        let p = precisionFromSample(labels: [], found: 500)
        #expect(p.checked == 0)
        #expect(precisionNote(p) == "Nothing checked by hand yet.")
    }

    @Test func sampleSizeIsSmallEnoughToActuallyDo() {
        #expect(precisionSampleSize(width: 0.20, found: 41_541) == 25)
        #expect(precisionSampleSize(width: 0.20, found: 7) == 7)       // never more than the result set
        #expect(precisionSampleSize(found: 0) == 0)
    }

    @Test func lowerMirrorsUpper() {
        // cpLower(hits) == 1 - cpUpper(n - hits) for the same n and alpha.
        for hits in 0...10 {
            let lo = cpLower(hits: hits, n: 10, alpha: 0.05)
            let up = cpUpper(hits: 10 - hits, n: 10, alpha: 0.05)
            #expect(abs(lo - (1 - up)) < 1e-6)
        }
    }
}
