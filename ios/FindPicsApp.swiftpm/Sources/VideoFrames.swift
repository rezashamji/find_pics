// Several frames per video, like the server's index (media.sample_video_frames: one every 2 s, at most 40, evenly
// thinned): each frame gets its own image vector and faces, so a video matches if ANY moment matches, and the judge
// looks at the frame that matched (not the cover image).
import AVFoundation
import CoreImage
import Photos

enum VideoFrames {
    static func avAsset(_ id: String) async -> AVAsset? {
        guard let a = PhotoLibrary.asset(id) else { return nil }
        let o = PHVideoRequestOptions(); o.isNetworkAccessAllowed = false; o.deliveryMode = .mediumQualityFormat
        return await withCheckedContinuation { c in
            PHImageManager.default().requestAVAsset(forVideo: a, options: o) { av, _, _ in c.resume(returning: av) }
        }
    }

    /// (seconds, frame) pairs: every 2 s, at most 40, spread evenly; frames are upright (preferred transform applied).
    static func sample(_ id: String, everySeconds: Double = 2, maxFrames: Int = 40, side: CGFloat = 1280) async -> [(Double, CIImage)] {
        guard let asset = await avAsset(id), let dur = try? await asset.load(.duration).seconds, dur > 0 else { return [] }
        let n = max(1, min(maxFrames, Int(dur / everySeconds) + 1))
        let times = (0..<n).map { n == 1 ? 0 : Double($0) * max(dur - 0.05, 0) / Double(n - 1) }
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: side, height: side)
        gen.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
        gen.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)
        var out = [(Double, CIImage)]()
        for t in times {
            if let (cg, actual) = try? await gen.image(at: CMTime(seconds: t, preferredTimescale: 600)) {
                out.append((actual.seconds, CIImage(cgImage: cg)))
            }
        }
        return out
    }

    /// The frame nearest a time (the matched one), for the judge and the preview.
    static func frame(_ id: String, at t: Double, side: CGFloat = 1280) async -> CIImage? {
        guard let asset = await avAsset(id) else { return nil }
        let gen = AVAssetImageGenerator(asset: asset); gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: side, height: side)
        return (try? await gen.image(at: CMTime(seconds: t, preferredTimescale: 600))).map { CIImage(cgImage: $0.image) }
    }
}
