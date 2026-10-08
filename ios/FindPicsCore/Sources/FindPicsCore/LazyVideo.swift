// Videos are indexed in two passes (MAC_INBOX M20, JOURNAL 10-08). Reza's first run was 36,497 iCloud-only videos, each
// decoded with ~9 frames embedded: ~13 h at ~46 per minute, only while the app was in front.
//   1. Cover frame: a video not indexed yet gets ONE image vector (and faces) from the still PhotoKit already holds for
//      it, read exactly like a photo (448 px resizeMode .fast, network off first): photo speed, no movie decode. Such an
//      entry has no frame times (IndexRecord.videoFramesPending).
//   2. Frames, later: once the cover-frame pass is done, each such video is decoded and sampled (videoSampleTimes), and
//      its single vector is replaced by its frame vectors in one store put. Newest first, downloadParallel in flight;
//      in the foreground in chunks and in the charger task to the end.
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

/// One indexed video as the frames pass sees it.
public struct VideoFramesItem: Equatable, Sendable {
    public var id: String
    public var taken: Double?
    /// Indexed from its cover frame only (no frame vectors yet).
    public var framesPending: Bool
    /// Its frame vectors were made by another image preparation (Embedder.imageVersion): sampled again.
    public var staleImage: Bool
    public init(id: String, taken: Double?, framesPending: Bool, staleImage: Bool) {
        self.id = id; self.taken = taken; self.framesPending = framesPending; self.staleImage = staleImage
    }
}

/// The frames pass's queue: videos with only a cover frame, or with frames from an older image preparation, newest
/// first (unknown dates last), minus `skip` (could not be read this launch; the charger task retries them).
public func videoFramesWork(_ videos: [VideoFramesItem], skip: Set<String> = []) -> [String] {
    videos.filter { ($0.framesPending || $0.staleImage) && !skip.contains($0.id) }
        .sorted { ($0.taken ?? -.infinity, $0.id) > ($1.taken ?? -.infinity, $1.id) }
        .map(\.id)
}

/// The image-preparation re-read (an entry's imageVersion is not the app's): true when the FIRST pass re-reads it. A
/// video WITH frames is left to the frames pass (it stays searchable with its old frames meanwhile), so the first pass
/// never turns frames back into a cover frame; a photo or a cover-frame-only video is re-read like a photo.
public func rereadInFirstPass(isVideo: Bool, framesPending: Bool) -> Bool { !isVideo || framesPending }

/// The banner's secondary line while the frames pass has work: "Looking inside videos: 1,234 of 36,497 ..." where
/// `sampled` counts indexed videos that have frames and `videos` all indexed videos (the whole job: it never goes down
/// when the app is relaunched). nil when there is nothing to say.
public func videoFramesLine(sampled: Int, videos: Int) -> String? {
    guard videos > 0, sampled < videos else { return nil }
    return "Looking inside videos: \(grouped(max(0, sampled))) of \(grouped(videos)). Until then a video is found by its cover frame."
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
