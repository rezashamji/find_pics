// Several frames per video (FindPicsCore.videoSampleTimes: one every 4 s, at most 40, evenly spread; the sampling
// sweep in eval/RESULTS.md 37 chose 4 s over the server's inherited 2 s; the frames pass gets there in two sweeps,
// FindPicsCore/LazyVideo.swift): each frame gets its own image vector and faces, so a video matches if ANY moment matches, and the judge
// looks at the frame that matched (not the cover image). Videos kept only in iCloud are downloaded when the purpose
// allows it: for the judge the original (full-resolution frames); for INDEXING PhotoKit's medium-quality derivative
// (FindPicsCore.videoDownload: frames are sampled at <= 1280 px, so a 16:9 original of any size gives 1280x720 frames
// either way), falling back to the original when PhotoKit has no derivative. MAC_INBOX M17: the one-at-a-time download
// pass spent ~10.5 s per item, and whole-original video downloads are the suspected cost.
import AVFoundation
import CoreImage
@preconcurrency import FindPicsCore
import Photos

enum VideoFrames {
    /// AVAsset is not Sendable; PhotoKit hands it over once and nothing else keeps it, so passing it out is safe.
    private struct Handoff: @unchecked Sendable { let asset: AVAsset?; let inCloud: Bool; let timedOut: Bool }

    private static func request(_ a: PHAsset, network: Bool, medium: Bool = false) async -> Handoff {
        let o = PHVideoRequestOptions(); o.isNetworkAccessAllowed = network
        o.deliveryMode = medium ? .mediumQualityFormat : .highQualityFormat
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
        let t0 = Date()
        if videoDownload(purpose) == .medium {
            let med = await request(a, network: true, medium: true)
            if let av = med.asset { IndexTiming.record("6 video download (medium)", Date().timeIntervalSince(t0)); return (av, nil) }
            // a stalled download is not retried as the (bigger) original now: the charger task retries it later
            if med.timedOut { return (nil, .downloadFailed) }
        }
        let net = await request(a, network: true)
        if let av = net.asset { IndexTiming.record("6 video download (original)", Date().timeIntervalSince(t0)); return (av, nil) }
        return (nil, .downloadFailed)
    }

    /// Frames of a video already read (avAsset) at the times `times` gives for its duration (FindPicsCore:
    /// videoSampleTimes, or a frames-pass sweep's times), upright (preferred transform applied). Each frame goes to
    /// `each` (index of the time asked for, actual time, image) as soon as it is decoded and is not kept: with
    /// downloadParallel videos in flight, holding up to 40 decoded 1280 px frames per video would cost hundreds of MB.
    /// Returns the duration and how many frames were asked for and delivered; nil when the duration cannot be read.
    static func sampleEach(_ asset: AVAsset, times: (Double) -> [Double], side: CGFloat = 1280,
                           each: (Int, Double, CGImage) -> Void) async -> (duration: Double, requested: Int, delivered: Int)? {
        guard let dur = try? await asset.load(.duration).seconds, dur > 0 else { return nil }
        let ts = times(dur)
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: side, height: side)
        gen.requestedTimeToleranceBefore = CMTime(seconds: videoFrameTolerance, preferredTimescale: 600)
        gen.requestedTimeToleranceAfter = CMTime(seconds: videoFrameTolerance, preferredTimescale: 600)
        var count = 0
        for (k, t) in ts.enumerated() {
            if let (cg, actual) = try? await gen.image(at: CMTime(seconds: t, preferredTimescale: 600)) {
                each(k, actual.seconds, cg); count += 1
            }
        }
        return (dur, ts.count, count)
    }

    /// The frames at these exact times (times `sampleEach` delivered), e.g. to re-embed their faces with a new face model;
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
