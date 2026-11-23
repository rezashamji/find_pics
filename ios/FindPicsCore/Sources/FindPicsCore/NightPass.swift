// THE NIGHT PASS as part of normal indexing (11-22): the decisions, free of PhotoKit and MLX so they are tested on
// Linux. The app (App.swift describePhotos) asks these functions which photos to describe next, when to save, and
// whether it may run at all; FactsWriter does the writing.
//
// Why it matters: a search scores every photo from its stored fingerprint (fast, fuzzy) and, where one exists, from
// the line the writer wrote about it (RESULTS 59-69: same quality with ~65 photo-judge calls instead of ~1,200). Until
// now only developer #commands wrote lines, so 1,885 of 148,451 photos (1.3%) had one.
import Foundation

/// One photo the pass may describe. Videos are left out: the line is written from one frame, and a video's
/// fingerprint is built from several, so the two would not describe the same thing (MilliScorer).
public struct NightPassPhoto: Sendable, Equatable {
    public let id: String
    /// Capture time, seconds since 1970; nil when unknown.
    public let taken: Double?
    public init(id: String, taken: Double?) { self.id = id; self.taken = taken }
}

/// The photos still to describe, NEWEST FIRST: recent photos are the ones searched most, and photos added later
/// join at the front, so a library that is never fully described is described where it is used. Photos with a line
/// already, and photos that could not be read this run (`skipped`), are left out. Ties (same time, or no time) keep
/// a fixed order by id so a resumed pass continues where it stopped.
public func nightPassQueue(_ photos: [NightPassPhoto], described: (String) -> Bool,
                           skipped: Set<String> = []) -> [String] {
    photos
        .filter { !described($0.id) && !skipped.contains($0.id) }
        .sorted { a, b in
            switch (a.taken, b.taken) {
            case let (x?, y?) where x != y: return x > y
            case (.some, nil): return true
            case (nil, .some): return false
            default: return a.id < b.id
            }
        }
        .map(\.id)
}

/// Save the store after this many new lines or this many seconds, whichever comes first. The file is rewritten
/// whole (a torn append would make it unreadable), ~770 B per photo, so ~115 MB at 148k photos: saving every 25
/// photos, as the developer runner does, would rewrite that every ~50 s all night. Losing at most a few minutes of
/// lines to a hard kill is the better trade.
public enum NightPassCheckpoint {
    public static let lines = 200
    public static let seconds = 300.0
}

public func nightPassShouldSave(linesSinceSave: Int, secondsSinceSave: Double) -> Bool {
    linesSinceSave > 0 && (linesSinceSave >= NightPassCheckpoint.lines || secondsSinceSave >= NightPassCheckpoint.seconds)
}

/// Why the pass cannot run right now (nil = it can). Each is a "not now", never "finished": the caller must keep the
/// remaining count and try again in the next charger session (see passStep).
public enum NightPassBlock: Equatable, Sendable {
    /// The person has not agreed to the model download yet (guideline 4.2.3).
    case noConsent
    /// The writer (1.7 GB) is not on the phone yet and this network may not download it (cellular, Low Data Mode).
    case writerNotDownloadable
    /// Too little free memory for the writer; iOS would kill the app mid-load.
    case lowMemory
    /// The phone is hot; the pass waits rather than add heat on the charger.
    case tooHot
    /// The app is on screen: the person may search at any moment, and the writer and the planner do not fit together.
    case appInUse
    /// The app is in the background (phone locked, or another app in front). MEASURED 11-23: iOS refuses GPU work
    /// from a backgrounded app, including inside a BGProcessingTask, and the writer runs on the GPU - the first try
    /// crashed the app one second after it left the screen. Background GPU needs a BGContinuedProcessingTask with
    /// the com.apple.developer.background-tasks.continued-processing.gpu entitlement (iOS 26+, not every device).
    case noGPUInBackground
}

/// Minutes of no use after which an app left OPEN on screen counts as idle (seconds).
public let nightPassIdleSeconds = 900.0

/// Is the person using the app? MEASURED 11-23 00:10-03:10 on Reza's phone: iOS ran the charger task every ~8 min,
/// 24 times, and the pass refused all 24 because the app was on screen - the phone sat unlocked on the charger all
/// night with find pics open. "On screen" alone is the wrong test. In use = a search is running or still looking,
/// or the app is on screen and was opened or searched in the last `nightPassIdleSeconds`. A search that starts while
/// the pass runs makes it stop at the current photo and free the writer first (AppModel.yieldNightPass).
public func nightPassAppBusy(onScreen: Bool, searching: Bool, secondsSinceUse: Double) -> Bool {
    searching || (onScreen && secondsSinceUse < nightPassIdleSeconds)
}

/// Free memory (GB, os_proc_available_memory) the writer needs: 1.7 GB of weights plus working memory, with MLX's
/// buffer cache capped at 256 MB in Judge.load.
public let nightPassMinFreeGB = 3.0

public func nightPassBlock(consented: Bool, writerOnPhone: Bool, network: NetworkPath, freeGB: Double,
                           thermal: ThermalLevel, appInUse: Bool, inBackground: Bool = false) -> NightPassBlock? {
    if !consented { return .noConsent }
    if appInUse { return .appInUse }
    if inBackground { return .noGPUInBackground }
    if thermal >= .serious { return .tooHot }
    if !writerOnPhone, !iCloudDownloadAllowed(.indexBackground, network) { return .writerNotDownloadable }
    // freeGB <= 0 means the platform did not report it (os_proc_available_memory is 0 off-device): do not block on it
    if freeGB > 0, freeGB < nightPassMinFreeGB { return .lowMemory }
    return nil
}

/// The search screen's line about the pass; empty once every photo is described (or there is nothing to say).
public func nightPassNote(described: Int, photos: Int) -> String {
    guard photos > 0, described < photos else { return "" }
    let pct = Int((100 * Double(described) / Double(photos)).rounded(.down))
    return "Describing your photos so searches get faster and more accurate: \(described) of \(photos) (\(pct)%). "
         + "This runs while your phone is charging and locked."
}
