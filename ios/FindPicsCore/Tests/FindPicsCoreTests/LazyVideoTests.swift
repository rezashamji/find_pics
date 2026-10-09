import XCTest
@testable import FindPicsCore

final class LazyVideoTests: XCTestCase {
    func testSampleTimesMatchServerLayout() {
        // media.sample_video_frames layout: n = min(40, dur // every + 1) evenly over [0, dur - 0.05]
        XCTAssertEqual(videoSampleTimes(duration: 1.5, everySeconds: 2), [0])
        let t = videoSampleTimes(duration: 9, everySeconds: 2)              // the old rule: 5 frames
        XCTAssertEqual(t.count, 5)
        XCTAssertEqual(t.first, 0); XCTAssertEqual(t.last!, 8.95, accuracy: 1e-9)
        XCTAssertEqual(videoSampleTimes(duration: 0), [0])
    }

    func testShippedRuleIsEvery4sCapped40() {
        // RESULTS 37: every 4 s found as many videos as every 2 s with about half the frames
        XCTAssertEqual(videoSampleTimes(duration: 3).count, 1)
        XCTAssertEqual(videoSampleTimes(duration: 9).count, 3)
        XCTAssertEqual(videoSampleTimes(duration: 20).count, 6)
        XCTAssertEqual(videoSampleTimes(duration: 60).count, 16)
        XCTAssertEqual(videoSampleTimes(duration: 600).count, 40)                 // capped
        XCTAssertEqual(videoSampleTimes(duration: 20).last!, 19.95, accuracy: 1e-9)
    }

    func testFramesWorkNewestFirstSkipsDoneAndSkipped() {
        let v = [VideoFramesItem(id: "old", taken: 100, stage: 0, staleImage: false),
                 VideoFramesItem(id: "new", taken: 300, stage: 0, staleImage: false),
                 VideoFramesItem(id: "done", taken: 400, stage: 2, staleImage: false),
                 VideoFramesItem(id: "stale", taken: 200, stage: 2, staleImage: true),
                 VideoFramesItem(id: "nodate", taken: nil, stage: 0, staleImage: false),
                 VideoFramesItem(id: "failed", taken: 500, stage: 0, staleImage: false)]
        XCTAssertEqual(videoFramesWork(v, skip: ["failed"]).map(\.id), ["new", "old", "nodate", "stale"])
        XCTAssertEqual(videoFramesWork(v).first, VideoFramesTask(id: "failed", sweep: 1))
    }

    func testEverySweep1VideoBeforeAnySweep2Video() {
        // the newest videos are already at stage 1: they still wait behind the OLDEST cover-only video
        let v = [VideoFramesItem(id: "s1-new", taken: 900, stage: 1, staleImage: false),
                 VideoFramesItem(id: "s1-old", taken: 50, stage: 1, staleImage: false),
                 VideoFramesItem(id: "s0-old", taken: 10, stage: 0, staleImage: false),
                 VideoFramesItem(id: "s0-nodate", taken: nil, stage: 0, staleImage: true),
                 VideoFramesItem(id: "s0-new", taken: 800, stage: 0, staleImage: false),
                 VideoFramesItem(id: "s2-stale", taken: 950, stage: 2, staleImage: true),
                 VideoFramesItem(id: "s2", taken: 999, stage: 2, staleImage: false)]
        let w = videoFramesWork(v)
        XCTAssertEqual(w.map(\.id), ["s0-new", "s0-old", "s0-nodate", "s2-stale", "s1-new", "s1-old"])
        XCTAssertEqual(w.map(\.sweep), [1, 1, 1, 2, 2, 2])
        let lastSweep1 = w.lastIndex { $0.sweep == 1 }!, firstSweep2 = w.firstIndex { $0.sweep == 2 }!
        XCTAssertLessThan(lastSweep1, firstSweep2)
        // a cover-only video that cannot be read now does not hold sweep 2 back this launch
        XCTAssertEqual(videoFramesWork(v, skip: ["s0-new", "s0-old", "s0-nodate"]).map(\.sweep), [2, 2, 2])
        // stages
        XCTAssertEqual(videoFramesNextSweep(stage: 0, staleImage: false), 1)
        XCTAssertEqual(videoFramesNextSweep(stage: 1, staleImage: false), 2)
        XCTAssertEqual(videoFramesNextSweep(stage: 2, staleImage: true), 2)
        XCTAssertNil(videoFramesNextSweep(stage: 2, staleImage: false))
    }

    func testSweep1TimesShortMediumLong() {
        XCTAssertEqual(videoSweep1Times(duration: 0.6), [])                  // under 1 s: the cover is the clip
        XCTAssertEqual(videoSweep1Times(duration: 1.5), [0.75])              // short: the middle frame only
        XCTAssertEqual(videoSweep1Times(duration: 2), [1])
        let s3 = videoSweep1Times(duration: 3)                               // middle + end once the middle is >= 1 s in
        XCTAssertEqual(s3.count, 2); XCTAssertEqual(s3[0], 1.475, accuracy: 1e-9); XCTAssertEqual(s3[1], 2.95, accuracy: 1e-9)
        // medium: exactly RESULTS 37's "fixed 3" layout minus its t = 0 (the cover stands in for it)
        let m = videoSweep1Times(duration: 20)
        XCTAssertEqual(m.count, 2); XCTAssertEqual(m[0], 9.975, accuracy: 1e-9); XCTAssertEqual(m[1], 19.95, accuracy: 1e-9)
        XCTAssertEqual(videoSampleTimes(duration: 20, everySeconds: 10).dropFirst().map { $0 }, m)   // 3 evenly spread
        let l = videoSweep1Times(duration: 600)                              // long: still 2 decoded frames
        XCTAssertEqual(l.count, 2); XCTAssertEqual(l[1], 599.95, accuracy: 1e-9)
        XCTAssertEqual(videoSweep1Times(duration: 0), [])
        XCTAssertEqual(videoSweep1Times(duration: .nan), [])
        // frames are >= 1 s from each other and from the cover (t = 0), so the generator's +-0.5 s cannot merge them
        for d in stride(from: 2.05, through: 90, by: 0.25) {
            let t = [0] + videoSweep1Times(duration: d)
            for k in 1..<t.count { XCTAssertGreaterThanOrEqual(t[k] - t[k - 1], 2 * videoFrameTolerance - 1e-9, "d=\(d)") }
            XCTAssertLessThan(t.last!, d)
        }
    }

    func testSweep2ReusesSweep1FramesAndAlwaysDecodesT0() {
        // 12 s: every 4 s = 0, 3.983, 7.967, 11.95; sweep 1 decoded 5.975 and 11.95 -> the end is not decoded again
        let s1 = videoSweep1Times(duration: 12)
        let t12 = videoSweep2Times(duration: 12, have: s1)
        XCTAssertEqual(t12.count, 3); XCTAssertEqual(t12[0], 0)
        XCTAssertEqual(t12[1], 11.95 / 3, accuracy: 1e-9); XCTAssertEqual(t12[2], 2 * 11.95 / 3, accuracy: 1e-9)
        // 20 s: grid 0, 3.99, 7.98, 11.97, 15.96, 19.95; the end (19.95) is already there -> 5 decoded
        XCTAssertEqual(videoSweep2Times(duration: 20, have: videoSweep1Times(duration: 20)).count, 5)
        // 9 s: grid 0, 4.475, 8.95 = the sweep-1 middle and end exactly -> only t = 0 is decoded
        XCTAssertEqual(videoSweep2Times(duration: 9, have: videoSweep1Times(duration: 9)), [0])
        // actual decoded times may be off by the generator's tolerance: still reused
        XCTAssertEqual(videoSweep2Times(duration: 9, have: [4.2, 8.6]), [0])
        // short (2 s): the grid is [0]; the middle frame is kept beside it
        XCTAssertEqual(videoSweep2Times(duration: 2, have: [1]), [0])
        // a sweep-1 frame decoded at t = 0 never stops t = 0 being decoded (the cover was only a stand-in)
        XCTAssertEqual(videoSweep2Times(duration: 1, have: [0]), [0])
        // long: capped at 40 like the full rule
        XCTAssertEqual(videoSweep2Times(duration: 600, have: videoSweep1Times(duration: 600)).count, 39)
        XCTAssertEqual(videoSweep2Times(duration: 60, have: []), videoSampleTimes(duration: 60))
    }

    func testFramesAfterSweep2IsExactlyTheEvery4sLayout() {
        func u(_ t: Double, _ x: Float) -> FrameUnit { FrameUnit(t: t, vector: [x], faces: []) }
        // 12 s: grid 0, 3.983, 7.967, 11.95; sweep 1 = cover + 5.975 + 11.95 -> the end is reused, the middle dropped
        let sweep1 = [u(0, 9), u(5.975, 1), u(11.95, 2)]                     // t = 0 is the cover stand-in
        let decoded = [u(0, 0), u(11.95 / 3, 3), u(2 * 11.95 / 3, 4)]       // what videoSweep2Times asked for
        let out = framesAfterSweep2(duration: 12, decoded: decoded, sweep1: sweep1)
        XCTAssertEqual(out.map(\.vector), [[0], [3], [4], [2]])
        XCTAssertEqual(out.count, videoSampleTimes(duration: 12).count)
        XCTAssertEqual(out[0].vector, [0])                                    // the decoded t = 0, not the cover
        // 9 s: grid 0, 4.475, 8.95 = sweep 1's middle and end (decoded 0.3 s late: within tolerance) -> both reused
        let o9 = framesAfterSweep2(duration: 9, decoded: [u(0, 0)], sweep1: [u(0, 9), u(4.775, 1), u(8.95, 2)])
        XCTAssertEqual(o9.map(\.vector), [[0], [1], [2]])
        // the times decoded + the times reused = the full layout, for any length
        for d in stride(from: 0.5, through: 200, by: 0.5) {
            let s1 = videoSweep1Times(duration: d).map { u($0, 1) }
            let dec = videoSweep2Times(duration: d, have: s1.map(\.t)).map { u($0, 0) }
            XCTAssertEqual(framesAfterSweep2(duration: d, decoded: dec, sweep1: [u(0, 9)] + s1).count,
                           videoSampleTimes(duration: d).count, "d=\(d)")
        }
    }

    func testFirstPassNeverTurnsFramesBackIntoACoverFrame() {
        XCTAssertTrue(rereadInFirstPass(isVideo: false, framesPending: false))   // photo
        XCTAssertTrue(rereadInFirstPass(isVideo: true, framesPending: true))     // cover-frame-only video
        XCTAssertFalse(rereadInFirstPass(isVideo: true, framesPending: false))   // has frames: the frames pass re-samples it
    }

    func testPendingIsVideoWithoutFrameTimes() {
        func rec(_ video: Bool, _ ts: [Double]?, faceSide: Double? = 448) -> IndexRecord {
            IndexRecord(id: "x", isVideo: video, taken: nil, localMinutes: nil, lat: nil, lon: nil, place: nil, camera: nil,
                        isScreenshot: nil, lowRes: nil, faceModel: nil, imageVersion: 2, faceSide: faceSide, vectorRow: 0,
                        frameRow: 0, frameTs: ts, facesKnown: true, faces: [])
        }
        XCTAssertTrue(rec(true, nil).videoFramesPending)
        XCTAssertFalse(rec(true, [0, 2]).videoFramesPending)
        XCTAssertFalse(rec(false, nil).videoFramesPending)
        // a cover frame's faces came from a 448 px read: not "full size" like sampled 1280 px frames
        XCTAssertEqual(rec(true, nil).faceSideEffective, 448)
        XCTAssertEqual(rec(true, [0]).faceSideEffective, faceReadSide)
    }

    func testBannerLineIsWholeJobAndGoesAway() {
        XCTAssertEqual(videoFramesLine(sweep1Done: 8907, sweep2Done: 8907, videos: 20341),
                       "Checking more of each video: 8,907 of 20,341. Until then a video is found by its cover frame.")
        // sweep 1 done for every video: the sweep 2 line, counted from the videos already sampled every 4 s
        XCTAssertEqual(videoFramesLine(sweep1Done: 20341, sweep2Done: 9000, videos: 20341),
                       "Looking closer inside videos: 9,000 of 20,341. Until then a video is found by a few of its frames.")
        XCTAssertNil(videoFramesLine(sweep1Done: 10, sweep2Done: 10, videos: 10))
        XCTAssertNil(videoFramesLine(sweep1Done: 0, sweep2Done: 0, videos: 0))
        // monotonic over a run: sweep 1's k rises to N, then the line switches and sweep 2's k rises to N
        var last = (-1, -1)
        for (s1, s2) in [(8907, 8907), (15000, 8907), (20341, 8907), (20341, 12000), (20341, 20340)] {
            XCTAssertNotNil(videoFramesLine(sweep1Done: s1, sweep2Done: s2, videos: 20341))
            XCTAssertTrue(s1 >= last.0 && s2 >= last.1); last = (s1, s2)
        }
        XCTAssertEqual(grouped(0), "0"); XCTAssertEqual(grouped(999), "999"); XCTAssertEqual(grouped(1000), "1,000")
        XCTAssertEqual(grouped(1234567), "1,234,567")
    }

    func testStageFromRecord() {
        func rec(_ video: Bool, _ ts: [Double]?, partial: Bool) -> IndexRecord {
            IndexRecord(id: "x", isVideo: video, taken: nil, localMinutes: nil, lat: nil, lon: nil, place: nil, camera: nil,
                        isScreenshot: nil, lowRes: nil, faceModel: nil, imageVersion: 2, faceSide: nil, vectorRow: 0,
                        frameRow: 0, frameTs: ts, facesKnown: true, faces: [], framesPartial: partial)
        }
        XCTAssertEqual(rec(true, nil, partial: false).videoFramesStage, 0)
        XCTAssertEqual(rec(true, [0, 5, 10], partial: true).videoFramesStage, 1)
        XCTAssertEqual(rec(true, [0, 4, 8], partial: false).videoFramesStage, 2)
        XCTAssertNil(rec(false, nil, partial: false).videoFramesStage)
        // sweep 1's frames come from the medium-quality movie (<= 1280 px): checked faces, like sweep 2's
        XCTAssertEqual(rec(true, [0, 5, 10], partial: true).faceSideEffective, faceReadSide)
    }

    func testCoverFrameNote() {
        XCTAssertEqual(coverFrameOnlyNote(media: "video", pendingInScope: 35269, foundVideos: 0),
                       "35,269 videos were only checked by their cover frame so far (find pics looks inside them next).")
        XCTAssertEqual(coverFrameOnlyNote(media: "any", pendingInScope: 1, foundVideos: 2),
                       "1 video was only checked by its cover frame so far (find pics looks inside it next).")
        XCTAssertNil(coverFrameOnlyNote(media: "any", pendingInScope: 50, foundVideos: 0))   // photos-only answer
        XCTAssertNil(coverFrameOnlyNote(media: "photo", pendingInScope: 50, foundVideos: 3))
        XCTAssertNil(coverFrameOnlyNote(media: "video", pendingInScope: 0, foundVideos: 3))
    }
}
