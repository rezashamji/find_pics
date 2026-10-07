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
        if o == nil || o == .unreadable { local.append(id); continue }
        if downloads && (o == .waitingForICloud || (o == .downloadFailed && retryFailed)) { download.append(id) }
    }
    return (local, download)
}

/// Index entries whose asset is gone from the library (deleted while the app was closed).
public func removedFromLibrary(indexed: [String], library: Set<String>) -> [String] { indexed.filter { !library.contains($0) } }
