// Keeping the on-phone index current and reading photos whose originals live only in iCloud: the decisions, free of
// PhotoKit so they are tested on Linux. The app (PhotoLibrary.swift, Index.swift) asks these functions what to do.
import Foundation

// MARK: - iCloud downloads

/// Why a photo is being read: decides whether PhotoKit may download the original from iCloud.
public enum FetchPurpose: Sendable { case localOnly, indexForeground, indexBackground, judge }

/// What NWPathMonitor reports: `expensive` = cellular or a phone hotspot, `constrained` = Low Data Mode.
public struct NetworkPath: Equatable, Sendable {
    public var connected: Bool, expensive: Bool, constrained: Bool
    public init(connected: Bool, expensive: Bool, constrained: Bool) {
        self.connected = connected; self.expensive = expensive; self.constrained = constrained
    }
}

/// Indexing downloads only on Wi-Fi-like networks (not expensive, not Low Data Mode); the charger/idle background task
/// is where most of them happen. A search the person is waiting on (the judge) also downloads on cellular, never in
/// Low Data Mode: the judge must see the full-resolution photo or skip it.
public func iCloudDownloadAllowed(_ purpose: FetchPurpose, _ path: NetworkPath) -> Bool {
    if purpose == .localOnly || !path.connected || path.constrained { return false }
    switch purpose {
    case .indexForeground, .indexBackground: return !path.expensive
    case .judge, .localOnly: return purpose == .judge
    }
}

/// Per-asset limits for an iCloud download (a photo original is a few MB; a video can be hundreds).
public enum ICloudTimeout { public static let photo: Double = 60, video: Double = 180 }

/// True when PhotoKit gave the size asked for (or the original is smaller than that): a smaller image is a local
/// stand-in for an iCloud-only original. Unknown original size (0) counts as full.
public func isFullResolution(gotW: Double, gotH: Double, requestedSide: Double, originalW: Double, originalH: Double) -> Bool {
    let origLong = max(originalW, originalH)
    if origLong <= 0 { return true }
    return max(gotW, gotH) >= 0.9 * min(requestedSide, origLong)
}

/// A local stand-in at least this big may be INDEXED (image vector, faces) until the original downloads; the judge
/// never uses a stand-in.
public let minStandInSide: Double = 448

/// Size indexing asks PhotoKit for (image vector at 224 px + faces). Reza's phone 10-07 (MAC M9 correction / M10):
/// every sampled iCloud-only photo has a ~480 px rendition ON the phone, served offline at a 448 px resizeMode .fast
/// ask (200/200, long side 448-486), while a 1280 ask finds nothing local and makes indexing wait on downloads.
/// The public libraries all quality numbers were measured on are <= 500 px, so 448 matches the tested conditions.
public let indexReadSide: Double = 448

/// Indexing (image vector + faces) takes a local copy of at least `minStandInSide` px as FINAL: it never downloads that
/// photo's original. Reza's phone 10-07: 169,923 of 187,119 items iCloud-only; downloading every original (hundreds of
/// GB) does not fit on the phone and took the whole night for a few dozen. The image model looks at small images
/// anyway. Only the judge downloads originals, for the photos it actually checks.
public func indexAcceptsLocalCopy(_ purpose: FetchPurpose, gotW: Double, gotH: Double) -> Bool {
    (purpose == .indexForeground || purpose == .indexBackground || purpose == .localOnly) && max(gotW, gotH) >= minStandInSide
}

/// Why an asset is not (fully) in the index yet.
public enum ReadOutcome: String, Codable, Sendable {
    case waitingForICloud      // original only in iCloud, download not allowed right now (cellular / Low Data / no network)
    case downloadFailed        // download tried and failed or timed out
    case unreadable            // not in iCloud and PhotoKit still gave nothing (damaged / unsupported)
}

/// The line the search screen shows; nil when everything was read at full resolution.
public func notReadSummary(_ notRead: [String: ReadOutcome], lowRes: Int) -> String? {
    let w = notRead.values.filter { $0 == .waitingForICloud }.count
    let f = notRead.values.filter { $0 == .downloadFailed }.count
    let u = notRead.values.filter { $0 == .unreadable }.count
    var parts = [String]()
    if w > 0 { parts.append("\(w) stored only in iCloud wait for Wi-Fi (they download while the phone charges)") }
    if f > 0 { parts.append("\(f) could not be downloaded from iCloud (tried again on the charger)") }
    if u > 0 { parts.append("\(u) could not be read at all") }
    var s = parts.isEmpty ? "" : "Not searchable yet: " + parts.joined(separator: "; ") + "."
    if lowRes > 0 { s += (s.isEmpty ? "" : " ") + "\(lowRes) are indexed from a smaller copy until their originals download." }
    return s.isEmpty ? nil : s
}

// MARK: - what to index

/// Which assets to read now. `local`: new assets (and earlier unreadable ones), read from what is on the phone.
/// `download`: assets waiting for iCloud, indexed from a stand-in (`lowRes`), or (with `retryFailed`, the charger task)
/// whose download failed before; only when downloads are allowed. Library order is kept (newest first).
public func indexWork(library: [String], indexedLowRes: [String: Bool], notRead: [String: ReadOutcome], downloads: Bool,
                      retryFailed: Bool) -> (local: [String], download: [String]) {
    var local = [String](), download = [String]()
    for id in library {
        let o = notRead[id]
        if let low = indexedLowRes[id] {     // a stand-in whose download failed waits for the charger task too
            if low && downloads && (o != .downloadFailed || retryFailed) { download.append(id) }
            continue
        }
        // waitingForICloud goes back through the LOCAL pass (downloads: false): since 10-07 the local ~480 px copy is
        // enough to index it, and the 169k such photos on Reza's phone were recorded before that rule existed
        if o == nil || o == .unreadable || (o == .waitingForICloud && !downloads) { local.append(id); continue }
        if downloads && (o == .waitingForICloud || (o == .downloadFailed && retryFailed)) { download.append(id) }
    }
    return (local, download)
}

// MARK: - the download pass (MAC_INBOX M17)

/// Download pass: assets read at once. One at a time cost ~10.5 s per item over 7 h on Reza's phone (JOURNAL 10-08),
/// while MAC M10 measured iCloud photo requests latency-bound (5 in flight ~3x faster than one by one).
public let downloadParallel = 5

/// Download pass order: photos first, then videos, each kept in library order (newest first). A photo download is a
/// ~480 px derivative (MAC M10: 0/50 originals at a 448 ask); a video is a whole movie file, so photos must not wait
/// behind tens of thousands of videos.
public func downloadOrder(_ ids: [String], videos: Set<String>) -> [String] {
    ids.filter { !videos.contains($0) } + ids.filter { videos.contains($0) }
}

/// Which file PhotoKit should fetch from iCloud for a video.
public enum VideoDownload: Equatable, Sendable {
    /// the original movie (can be hundreds of MB)
    case original
    /// PhotoKit's medium-quality derivative (PHVideoRequestOptionsDeliveryMode.mediumQualityFormat: reported as 720p);
    /// falls back to the original when PhotoKit gives none
    case medium
}

/// Indexing samples frames at most 1280 px on the long side (VideoFrames.sample), so a 16:9 original of any size
/// becomes 1280x720 frames: a 720p derivative gives the same frame size for a fraction of the download. The judge
/// (and the matched-frame preview) keeps the original.
public func videoDownload(_ purpose: FetchPurpose) -> VideoDownload {
    switch purpose {
    case .indexForeground, .indexBackground: return .medium
    case .judge, .localOnly: return .original
    }
}

/// Buckets for the M17 queue breakdown: the long side (px) PhotoKit returned for a 448 .fast ask with the network off.
public func renditionBucket(_ longSide: Double) -> String {
    if longSide <= 0 { return "nothing" }
    if longSide < 224 { return "<224" }
    if longSide < minStandInSide { return "224-447" }
    return ">=448"
}

/// Buckets for the M17 queue breakdown: video length in seconds.
public func durationBucket(_ seconds: Double) -> String {
    if seconds < 10 { return "<10s" }
    if seconds < 30 { return "10-30s" }
    if seconds < 80 { return "30-80s" }
    return ">=80s"
}

/// Index entries whose asset is gone from the library (deleted while the app was closed).
public func removedFromLibrary(indexed: [String], library: Set<String>) -> [String] { indexed.filter { !library.contains($0) } }
