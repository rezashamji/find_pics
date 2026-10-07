// Several frames per video, like the server's index (media.sample_video_frames: one every 2 s, at most 40, evenly
// thinned): each frame gets its own image vector and faces, so a video matches if ANY moment matches, and the judge
// looks at the frame that matched (not the cover image). Videos kept only in iCloud are downloaded (original quality,
// never a lower-quality stream: the judge must see full-resolution frames) when the purpose allows it.
import AVFoundation
import CoreImage
@preconcurrency import FindPicsCore
import Photos

enum VideoFrames {
    /// AVAsset is not Sendable; PhotoKit hands it over once and nothing else keeps it, so passing it out is safe.
    private struct Handoff: @unchecked Sendable { let asset: AVAsset?; let inCloud: Bool; let timedOut: Bool }

    private static func request(_ a: PHAsset, network: Bool) async -> Handoff {
        let o = PHVideoRequestOptions(); o.isNetworkAccessAllowed = network; o.deliveryMode = .highQualityFormat
        o.version = .current
        return await withCheckedContinuation { (c: CheckedContinuation<Handoff, Never>) in
            let box = OnceBox(c)
            o.progressHandler = { _, _, _, _ in box.touch() }
            let rid = PHImageManager.default().requestAVAsset(forVideo: a, options: o) { av, _, info in
                box.fire(Handoff(asset: av, inCloud: (info?[PHImageResultIsInCloudKey] as? Bool) == true, timedOut: false))
            }
            watchStall(box, stall: network ? ICloudTimeout.video : 30) {
                if box.fire(Handoff(asset: nil, inCloud: false, timedOut: true)) { PHImageManager.default().cancelImageRequest(rid) }
            }
        }
    }

    /// The video, or why not: local first, then an iCloud download if `purpose` allows it on this network.
    static func avAsset(_ id: String, purpose: FetchPurpose = .judge) async -> (AVAsset?, ReadOutcome?) {
        guard let a = PhotoLibrary.asset(id) else { return (nil, .unreadable) }
        let local = await request(a, network: false)
        if let av = local.asset { return (av, nil) }
        if !local.inCloud && !local.timedOut { return (nil, .unreadable) }
        guard iCloudDownloadAllowed(purpose, NetworkState.shared.path) else { return (nil, .waitingForICloud) }
        let net = await request(a, network: true)
        if let av = net.asset { return (av, nil) }
        return (nil, .downloadFailed)
    }

    /// (seconds, frame) pairs: every 2 s, at most 40, spread evenly; frames are upright (preferred transform applied).
    /// `outcome` is nil when the video was read.
    static func sample(_ id: String, purpose: FetchPurpose, everySeconds: Double = 2, maxFrames: Int = 40,
                       side: CGFloat = 1280) async -> (frames: [(Double, CIImage)], outcome: ReadOutcome?) {
        let (av, why) = await avAsset(id, purpose: purpose)
        guard let asset = av else { return ([], why) }
        guard let dur = try? await asset.load(.duration).seconds, dur > 0 else { return ([], .unreadable) }
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
        return (out, out.isEmpty ? .unreadable : nil)
    }

    /// The frames at these exact times (times `sample` returned), e.g. to re-embed their faces with a new face model;
    /// nil when the video cannot be read now (`purpose` decides whether iCloud may download).
    static func frames(_ id: String, at ts: [Double], purpose: FetchPurpose, side: CGFloat = 1280) async -> [Double: CIImage]? {
        guard let asset = await avAsset(id, purpose: purpose).0 else { return nil }
        let gen = AVAssetImageGenerator(asset: asset); gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: side, height: side)
        gen.requestedTimeToleranceBefore = .zero; gen.requestedTimeToleranceAfter = .zero
        var out = [Double: CIImage]()
        for t in ts {
            if let (cg, _) = try? await gen.image(at: CMTime(seconds: t, preferredTimescale: 600)) { out[t] = CIImage(cgImage: cg) }
        }
        return out
    }

    /// The frame nearest a time (the matched one), for the judge and the preview.
    static func frame(_ id: String, at t: Double, side: CGFloat = 1280) async -> CIImage? {
        guard let asset = await avAsset(id, purpose: .judge).0 else { return nil }
        let gen = AVAssetImageGenerator(asset: asset); gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: side, height: side)
        return (try? await gen.image(at: CMTime(seconds: t, preferredTimescale: 600))).map { CIImage(cgImage: $0.image) }
    }
}
