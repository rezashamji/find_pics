// Videos are indexed in passes (MAC_INBOX M20 + M24, JOURNAL 10-08 / 10-09). Reza's first run was 36,497 iCloud-only
// videos, each decoded with ~9 frames embedded: ~13 h at ~46 per minute, only while the app was in front.
//   1. Cover frame: a video not indexed yet gets ONE image vector (and faces) from the still PhotoKit already holds for
//      it, read exactly like a photo (448 px resizeMode .fast, network off first): photo speed, no movie decode. Such an
//      entry has no frame times (IndexRecord.videoFramesPending, videoFramesStage 0).
//   2. Frames pass, later, in TWO SWEEPS (eval/RESULTS.md 37, strict truth, 303 truth videos: cover only 191, 3 frames
//      per video (start / middle / end) 216, every 4 s ~6.1 frames 219):
//      sweep 1: EVERY cover-only video, newest first, gets its middle and end frames (videoSweep1Times; the measured
//               "fixed 3" layout, not 1/3 + 2/3, which found 211/303); the cover vector stays as the t = 0 unit.
//               Stage 1 (IndexRecord.framesPartial).
//      sweep 2: only once no cover-only video is left to try: every 4 s (videoSampleTimes), reusing the sweep-1
//               frames that fall on its times instead of decoding them again (videoSweep2Times, framesAfterSweep2).
//               Stage 2.
//      The units replace the entry's in one store put. downloadParallel in flight; in the foreground in chunks and in
//      the charger task to the end.
// Search stays honest meanwhile: a cover-frame-only video can match, and the album says how many were only checked by
// their cover frame (coverFrameOnlyNote). Decisions here; PhotoKit / AVFoundation in the app (Index.swift).
import Foundation

/// Videos per frames-pass chunk: the foreground works in chunks (new photos are indexed in between) and the charger
/// task alternates chunks with the face upgrade.
public let videoFramesChunk = 100

/// The frame times sampled from a video of `duration` seconds: `count` frames spread evenly from 0 to just before the
/// end (the server's media.sample_video_frames layout), count = duration / everySeconds + 1, capped at maxFrames.
/// EVERY 4 s, not the inherited 2 s (eval/RESULTS.md 37, sampling sweep on 522 Pexels clips, 24 queries, 9B judge on a
/// 1 s frame grid): 4 s found 219/303 truth videos vs 213/303 at 2 s (paired: +15 / -9, p = 0.31) with 6.1 instead of
/// 11.7 frames per clip; 1 s found 218/303 with 21.6. Not worse for short clips (<10 s: 31/39 vs 30/39). The phone
/// judges ONE best frame per video, so more frames mostly add near-duplicates. Untested above 77 s (none in the set).
public func videoSampleTimes(duration: Double, everySeconds: Double = 4, maxFrames: Int = 40) -> [Double] {
    guard duration > 0, everySeconds > 0, maxFrames > 0 else { return [0] }
    let n = max(1, min(maxFrames, Int(duration / everySeconds) + 1))
    if n == 1 { return [0] }
    let end = max(duration - 0.05, 0)
    return (0..<n).map { Double($0) * end / Double(n - 1) }
}

/// AVAssetImageGenerator's tolerance either side of a requested time (VideoFrames.sampleEach): a frame already decoded
/// within this of a requested time is an answer the generator itself could have given, so it is not decoded again.
public let videoFrameTolerance = 0.5

/// Sweep 1's frames beside the cover (which stands in for t = 0 and is not decoded): the MIDDLE and the END of the
/// clip, i.e. RESULTS 37's "fixed 3" layout (3 frames evenly over [0, duration - 0.05], the videoSampleTimes layout)
/// without its t = 0. Measured on 10-09 with the same harness (eval/video_sampling/analyze_sweeps.log, strict truth):
/// cover + middle + end found 216/303, cover + 1/3 + 2/3 211/303 at the same cost, cover only 191/303, every 4 s
/// 219/303. Frames at least 1 s apart (2 x videoFrameTolerance) are distinct, so middle + end need the middle >= 1 s
/// in (a clip of 2.05 s or more); from 1 s the middle frame alone; under 1 s none (the cover is the whole clip).
public func videoSweep1Times(duration: Double) -> [Double] {
    guard duration.isFinite else { return [] }
    let end = max(duration - 0.05, 0)
    if end / 2 >= 1 { return [end / 2, end] }
    if duration >= 1 { return [duration / 2] }
    return []
}

/// Sweep 2's frames to decode: every 4 s (videoSampleTimes) minus times a frame kept from sweep 1 (`have`, decoded
/// times; not the cover) already answers within videoFrameTolerance. t = 0 is always decoded: the cover was a stand-in.
public func videoSweep2Times(duration: Double, have: [Double]) -> [Double] {
    videoSampleTimes(duration: duration).filter { t in
        t == 0 || !have.contains { abs($0 - t) <= videoFrameTolerance }
    }
}

/// Sweep 2's units: exactly the every-4-s layout. The newly decoded frames plus those of sweep 1's decoded frames that
/// answer one of its times within videoFrameTolerance (the end always does; the middle when the frame count is odd);
/// sweep 1's other frames and the cover unit (sweep 2 decodes the real t = 0) are dropped. Keeping all of sweep 1's
/// frames measured WORSE (eval/video_sampling/analyze_sweeps.log, strict truth: 217/303 vs 219/303, +0 -2): an extra
/// frame can only change which frame wins the text score, never add a video the grid's frames would not.
public func framesAfterSweep2(duration: Double, decoded: [FrameUnit], sweep1: [FrameUnit]) -> [FrameUnit] {
    let grid = videoSampleTimes(duration: duration).filter { $0 > 0 }
    let keep = sweep1.filter { u in u.t > 0 && grid.contains { abs(u.t - $0) <= videoFrameTolerance } }
    return (decoded + keep).sorted { $0.t < $1.t }
}

/// One indexed video as the frames pass sees it.
public struct VideoFramesItem: Equatable, Sendable {
    public var id: String
    public var taken: Double?
    /// IndexRecord.videoFramesStage: 0 cover frame only, 1 sweep 1 done, 2 every 4 s.
    public var stage: Int
    /// Its frame vectors were made by another image preparation (Embedder.imageVersion): sampled again.
    public var staleImage: Bool
    public init(id: String, taken: Double?, stage: Int, staleImage: Bool) {
        self.id = id; self.taken = taken; self.stage = stage; self.staleImage = staleImage
    }
}

/// One video the frames pass works on next, and which sweep (1 or 2) it gets.
public struct VideoFramesTask: Equatable, Sendable {
    public var id: String
    public var sweep: Int
    public init(id: String, sweep: Int) { self.id = id; self.sweep = sweep }
}

/// The sweep a video still needs: 1 for a cover-frame-only video, 2 after sweep 1, 2 again for frames an older image
/// preparation made (re-sampled in full); nil when it is done.
public func videoFramesNextSweep(stage: Int, staleImage: Bool) -> Int? {
    switch stage {
    case 0: return 1
    case 1: return 2
    default: return staleImage ? 2 : nil
    }
}

/// The frames pass's queue: EVERY sweep-1 video before ANY sweep-2 video (sweep 2 waits until no cover-only video is
/// left to try), each sweep newest first (unknown dates last), minus `skip` (could not be read this launch; the charger
/// task retries them, and a retried cover-only video goes ahead of sweep 2 again).
public func videoFramesWork(_ videos: [VideoFramesItem], skip: Set<String> = []) -> [VideoFramesTask] {
    videos.compactMap { v -> (VideoFramesItem, Int)? in
        guard !skip.contains(v.id), let s = videoFramesNextSweep(stage: v.stage, staleImage: v.staleImage) else { return nil }
        return (v, s)
    }
    .sorted { a, b in
        if a.1 != b.1 { return a.1 < b.1 }
        return (a.0.taken ?? -.infinity, a.0.id) > (b.0.taken ?? -.infinity, b.0.id)
    }
    .map { VideoFramesTask(id: $0.0.id, sweep: $0.1) }
}

/// The image-preparation re-read (an entry's imageVersion is not the app's): true when the FIRST pass re-reads it. A
/// video WITH frames is left to the frames pass (it stays searchable with its old frames meanwhile), so the first pass
/// never turns frames back into a cover frame; a photo or a cover-frame-only video is re-read like a photo.
public func rereadInFirstPass(isVideo: Bool, framesPending: Bool) -> Bool { !isVideo || framesPending }

/// The banner's secondary line while the frames pass has work, whole job (counted from the index, so it never goes
/// down when the app is relaunched): sweep 1 first, "Checking more of each video: k of N" with k = videos past their
/// cover frame (stage >= 1); then sweep 2, "Looking closer inside videos: k of N" with k = videos sampled every 4 s
/// (stage 2). N = all indexed videos. nil when there is nothing to say.
public func videoFramesLine(sweep1Done: Int, sweep2Done: Int, videos: Int) -> String? {
    guard videos > 0 else { return nil }
    if sweep1Done < videos {
        return "Checking more of each video: \(grouped(max(0, sweep1Done))) of \(grouped(videos)). "
             + "Until then a video is found by its cover frame."
    }
    guard sweep2Done < videos else { return nil }
    return "Looking closer inside videos: \(grouped(max(0, sweep2Done))) of \(grouped(videos)). "
         + "Until then a video is found by a few of its frames."
}

/// The album's note about videos only checked by their cover frame so far (nil: none, or the album is about photos).
/// Said when the album asks for videos, or when it found any video: then a moment later in a video may still be
/// missing. `pendingInScope` = cover-frame-only videos among the album's candidates.
public func coverFrameOnlyNote(media: String, pendingInScope: Int, foundVideos: Int) -> String? {
    guard media != "photo", pendingInScope > 0, media == "video" || foundVideos > 0 else { return nil }
    let n = pendingInScope
    return "\(grouped(n)) video\(n == 1 ? " was" : "s were") only checked by \(n == 1 ? "its" : "their") cover frame so far "
         + "(find pics looks inside \(n == 1 ? "it" : "them") next)."
}

/// 36497 -> "36,497" (no Foundation formatter: same output on Linux and iOS, any locale).
func grouped(_ n: Int) -> String {
    let s = String(abs(n))
    var out = ""
    for (k, c) in s.enumerated() {
        if k > 0, (s.count - k) % 3 == 0 { out.append(",") }
        out.append(c)
    }
    return (n < 0 ? "-" : "") + out
}
