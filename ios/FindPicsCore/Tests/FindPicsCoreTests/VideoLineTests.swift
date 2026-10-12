import Testing
@testable import FindPicsCore

// The two video sweeps share one line, and one unreadable video used to pin it on sweep 1 for ever (Reza saw
// "39,000 out of 39,000" while sweep 2 sat at 11,734 of 39,579 unseen).
@Suite struct VideoLineTests {
    @Test func sweepOneWhileItIsStillTheOnlyOneRunning() {
        let l = videoFramesLine(sweep1Done: 12_000, sweep2Done: 0, videos: 39_579)
        #expect(l?.hasPrefix("Checking more of each video: 12,000 of 39,579") == true)
    }

    @Test func sweepTwoCountAloneDoesNotSwitchIt() {
        // sweep 2 reuses sweep 1's frames, so videos sampled in an earlier run count in both: a nonzero sweep 2
        // while sweep 1 still has readable videos left must NOT switch the line.
        let l = videoFramesLine(sweep1Done: 8_907, sweep2Done: 8_907, videos: 20_341, unreadable: 0)
        #expect(l?.hasPrefix("Checking more of each video: 8,907 of 20,341") == true)
    }

    @Test func switchesOnceSweepOneHasNothingLeftItCanRead() {
        // Reza's own phone, 10-11 22:23: one video sweep 1 cannot read, sweep 2 at 11,734 and invisible
        let l = videoFramesLine(sweep1Done: 39_578, sweep2Done: 11_734, videos: 39_579, unreadable: 1)
        #expect(l?.hasPrefix("Looking closer inside your longer videos: 11,734 of 39,579") == true)
        #expect(l?.contains("1 could not be read and was left out.") == true)
    }

    @Test func noLeftOutNoteWhenSweepOneGotEverything() {
        let l = videoFramesLine(sweep1Done: 39_579, sweep2Done: 500, videos: 39_579, unreadable: 0)
        #expect(l?.contains("could not be read") == false)
    }

    @Test func nothingToSayWhenBothAreDone() {
        #expect(videoFramesLine(sweep1Done: 39_579, sweep2Done: 39_579, videos: 39_579) == nil)
        #expect(videoFramesLine(sweep1Done: 0, sweep2Done: 0, videos: 0) == nil)
    }
}

// Sweep 2 is for long clips only (M52): the cluster measured a real gain for 3+ minute videos (15 -> 29 of 73
// matches, p < 0.01) and none that held up under 180 s.
@Suite struct Sweep2GateTests {
    @Test func shortClipsStopAfterSweepOne() {
        #expect(videoFramesNextSweep(stage: 1, staleImage: false, duration: 20) == nil)
        #expect(videoFramesNextSweep(stage: 1, staleImage: false, duration: 179) == nil)
        #expect(videoFramesNextSweep(stage: 1, staleImage: false, duration: 180) == 2)
        #expect(videoFramesNextSweep(stage: 1, staleImage: false, duration: 900) == 2)
    }

    @Test func unknownLengthWaitsForSweepOneToMeasureIt() {
        #expect(videoFramesNextSweep(stage: 0, staleImage: false, duration: nil) == 1)
        #expect(videoFramesNextSweep(stage: 1, staleImage: false, duration: nil) == nil)
    }

    @Test func staleVectorsAreResampledWhateverTheLength() {
        // Correctness, not coverage: frames made by an older image preparation cannot be compared against.
        #expect(videoFramesNextSweep(stage: 2, staleImage: true, duration: 5) == 2)
        #expect(videoFramesNextSweep(stage: 2, staleImage: false, duration: 5) == nil)
    }

    @Test func theQueueDropsShortClipsFromSweepTwo() {
        let vids = [VideoFramesItem(id: "long", taken: 2, stage: 1, staleImage: false, duration: 400),
                    VideoFramesItem(id: "short", taken: 3, stage: 1, staleImage: false, duration: 30),
                    VideoFramesItem(id: "new", taken: 1, stage: 0, staleImage: false, duration: nil)]
        let work = videoFramesWork(vids)
        #expect(work.map(\.id) == ["new", "long"])      // sweep 1 first, then the only long sweep-2 clip
        #expect(work.map(\.sweep) == [1, 2])
    }

    @Test func sweepTwoLineCountsAgainstTheLongClips() {
        let l = videoFramesLine(sweep1Done: 39_579, sweep2Done: 1_200, videos: 39_579, unreadable: 0, longVideos: 3_000)
        #expect(l?.contains("1,200 of 3,000") == true)
        // and it goes away once every long clip is done, however many short ones exist
        #expect(videoFramesLine(sweep1Done: 39_579, sweep2Done: 3_000, videos: 39_579, longVideos: 3_000) == nil)
    }
}
