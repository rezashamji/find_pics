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
        let v = [VideoFramesItem(id: "old", taken: 100, framesPending: true, staleImage: false),
                 VideoFramesItem(id: "new", taken: 300, framesPending: true, staleImage: false),
                 VideoFramesItem(id: "done", taken: 400, framesPending: false, staleImage: false),
                 VideoFramesItem(id: "stale", taken: 200, framesPending: false, staleImage: true),
                 VideoFramesItem(id: "nodate", taken: nil, framesPending: true, staleImage: false),
                 VideoFramesItem(id: "failed", taken: 500, framesPending: true, staleImage: false)]
        XCTAssertEqual(videoFramesWork(v, skip: ["failed"]), ["new", "stale", "old", "nodate"])
        XCTAssertEqual(videoFramesWork(v).first, "failed")
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
        XCTAssertEqual(videoFramesLine(sampled: 1234, videos: 36497),
                       "Looking inside videos: 1,234 of 36,497. Until then a video is found by its cover frame.")
        XCTAssertNil(videoFramesLine(sampled: 10, videos: 10))
        XCTAssertNil(videoFramesLine(sampled: 0, videos: 0))
        XCTAssertEqual(grouped(0), "0"); XCTAssertEqual(grouped(999), "999"); XCTAssertEqual(grouped(1000), "1,000")
        XCTAssertEqual(grouped(1234567), "1,234,567")
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
