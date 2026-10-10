// find pics: type what you are looking for, like the Photos search bar; everything runs on this phone.
import BackgroundTasks
import CoreImage
@preconcurrency import FindPicsCore
import os
import SwiftUI
import UIKit

// FindPicsCore marks its plain value types Sendable itself (strings, numbers, arrays), so they pass between actors.

@main
struct FindPicsApp: App {
    @StateObject var model = AppModel.shared
    @Environment(\.scenePhase) var phase
    init() {
        BackgroundIndexing.register()            // BGTaskScheduler: before the app finishes launching
        _ = NetworkState.shared                  // start NWPathMonitor now: its first report takes a moment
    }
    var body: some Scene {
        WindowGroup { RootView().environmentObject(model) }
            .onChange(of: phase) { _, p in
                if p == .background { BackgroundIndexing.schedule() }
                if p == .active { model.becameActive() }
            }
    }
}

/// Indexing while the phone is idle on the charger (how Apple Photos runs its own analysis): a BGProcessingTask with
/// requiresExternalPower. Needs Info.plist keys BGTaskSchedulerPermittedIdentifiers + UIBackgroundModes "processing",
/// merged from FindPicsInfo.plist by Package.swift (additionalInfoPlistContentFilePath); docs/MAC_SESSION.md says how
/// to check they reached the built app. iCloud downloads in this task still need Wi-Fi (FindPicsCore.iCloudDownloadAllowed).
enum BackgroundIndexing {
    static let id = "com.rezashamji.findpics.index"      // == FindPicsInfo.plist BGTaskSchedulerPermittedIdentifiers
    nonisolated(unsafe) private static var registered = false   // touched only from App.init on the main thread

    /// BGTask is not Sendable; the scheduler hands it over once and only the main actor uses it afterwards.
    struct Handoff: @unchecked Sendable { let task: BGTask }

    static func register() {
        guard !registered else { return }
        registered = true
        // this closure is formed here, outside the main actor: BGTaskScheduler calls it on its own queue
        let ok = BGTaskScheduler.shared.register(forTaskWithIdentifier: id, using: nil) { task in
            let cancel = CancelBox()
            task.expirationHandler = { cancel.cancel() }        // iOS wants the time back: stop at the next photo
            let h = Handoff(task: task)
            Task { @MainActor in
                let finished = await AppModel.shared.runBackgroundIndexing(cancel: cancel)
                h.task.setTaskCompleted(success: finished)
            }
        }
        if !ok { Logger().error("find pics: background task not registered (identifier missing from Info.plist?)") }
    }

    /// The expiration handler may fire before the indexing task exists: remember it and cancel on arrival.
    final class CancelBox: @unchecked Sendable {     // state behind the lock
        private let lock = NSLock(); private var task: Task<Void, Never>?; private var cancelled = false
        func set(_ t: Task<Void, Never>) { lock.lock(); task = t; let c = cancelled; lock.unlock(); if c { t.cancel() } }
        func cancel() { lock.lock(); cancelled = true; let t = task; lock.unlock(); t?.cancel() }
        var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    }

    static func schedule() {
        let r = BGProcessingTaskRequest(identifier: id)
        r.requiresExternalPower = true            // only on the charger
        r.requiresNetworkConnectivity = false     // new local photos need no network; downloads check Wi-Fi themselves
        do { try BGTaskScheduler.shared.submit(r) }
        catch { Logger().error("find pics: background task not scheduled: \(String(describing: error))") }
    }
}

@MainActor
final class AppModel: ObservableObject {
    enum Stage: Equatable { case start, noAccess, askDownload, downloading(Double), indexing(Int, Int), ready, failed(String) }
    /// App Store guideline 4.2.3(ii): say how big the download is and ask before fetching the models.
    static let downloadGB = 3.1          // mlx-community/Qwen3.5-4B-4bit (3.06 GB) + tokenizer files
    // UserDefaults-backed @Published (not @AppStorage: inside an ObservableObject it does not notify the views)
    @Published var downloadAccepted = UserDefaults.standard.bool(forKey: "modelDownloadAccepted") {
        didSet { UserDefaults.standard.set(downloadAccepted, forKey: "modelDownloadAccepted") }
    }
    /// Which model judges photos and plans searches: "qwen" (downloaded, gives probabilities), "apple-rating" or
    /// "apple-yesno" (Apple's built-in model, iOS 27, no probabilities). For the side-by-side test on the phone.
    @Published var engine = UserDefaults.standard.string(forKey: "engine") ?? "qwen3vl" {   // phone default judge (10-07)
        didSet { UserDefaults.standard.set(engine, forKey: "engine") }
    }
    /// Set at launch when iOS will not give this app enough memory for the downloaded judge (no
    /// increased-memory-limit entitlement -> ~2.6 GB on a 12 GB iPhone 18 Pro, against ~3.6 GB needed). The Qwen
    /// engines cannot run in this process at all, so judging AND planning use Apple's built-in model, which runs
    /// out-of-process and is not charged to our cap. Deliberately NOT written into `engine`: that is the user's
    /// Model-menu choice and must come back unchanged once the entitlement raises the cap.
    @Published var qwenOutOfMemory = false
    /// One line under the search field explaining the fallback (empty when nothing is wrong).
    @Published var engineNote = ""
    private var appleJudges: [String: any PhotoJudge] = [:]
    /// Candidate photo judge (Qwen3-VL-4B, 4-bit, ~2.5 GB, downloaded on first use); the planner stays on `judge`.
    let visionJudge = Judge(modelID: Judge.visionJudgeCandidateID)
    private lazy var voteJudge = EnsembleJudge(first: visionJudge, second: judge)
    /// Which engine actually runs: the Model menu's choice, or Apple's rating model when the Qwen weights do not fit.
    var effectiveEngine: String {
        guard qwenOutOfMemory else { return engine }
        return engine.hasPrefix("apple") ? engine : "apple-rating"
    }
    var activeJudge: any PhotoJudge {
        let engine = effectiveEngine
        if engine == "qwen3vl" { return visionJudge }
        if engine == "vote" { return voteJudge }
        #if canImport(FoundationModels)
        if #available(iOS 27.0, *), engine.hasPrefix("apple"), AppleJudge.unavailableReason == nil {
            if let j = appleJudges[engine] { return j }
            let j = AppleJudge(mode: engine == "apple-yesno" ? .yesNo : engine == "apple-rating100" ? .rating100 : .rating)
            appleJudges[engine] = j; return j
        }
        #endif
        return judge
    }
    var activePlanner: Planner {
        #if canImport(FoundationModels)
        if #available(iOS 27.0, *), effectiveEngine.hasPrefix("apple"), AppleJudge.unavailableReason == nil {
            return Planner(generate: { try await AppleText.text($0) })
        }
        #endif
        return Planner(judge: judge)
    }
    static let shared = AppModel()     // one model: the UI and the background task share the index and stores
    @Published var stage: Stage = .start
    @Published var results: [AlbumResult] = []
    @Published var busy = false
    @Published var lastQuery = ""
    @Published var planNote = ""
    @Published var indexStatus = ""               // FindPicsCore.notReadSummary: what searches cannot see yet, and why
    /// The one-time conversion of the old index.json into the binary store (0...1; nil when not converting).
    @Published var indexConversion: Double? = nil
    @Published var indexProgress: IndexProgress?  // new photos / iCloud downloads in progress (banner)
    @Published var libraryCount = 0               // photos + videos in the library (banner: "X of Y searchable")
    /// Where the current pass started (kind, count, time): the banner's rate / time-left (FindPicsCore.progressEstimate).
    var progressStart: (kind: Int, done: Int, at: Date)? = nil
    var progressETA: String? {
        guard let p = indexProgress, let s = progressStart else { return nil }
        return progressEstimate(done: p.done, total: p.total, startDone: s.done, elapsed: Date().timeIntervalSince(s.at))
    }
    /// The frames pass (FindPicsCore/LazyVideo.swift): indexed videos past sweep 1 / sweep 2, of all indexed videos (the
    /// banner's secondary "Checking more of each video" / "Looking closer inside videos" line; FindPicsCore.videoFramesLine).
    @Published var videoFrames: VideoFramesProgress?
    /// Face-model change in progress or finished with photos still waiting (banner); nil when nothing to do.
    @Published var faceReindex: FaceReindexProgress?
    /// The index still holds faces of the old face model, or saved people wait to be re-derived: people searches wait
    /// (old and new vectors are never compared, and "Is this you?" is not asked while the old pick can be re-derived).
    @Published var faceChangePending = false
    var facesUpdating: Bool { faceChangePending || faceReindex?.running == true }
    /// A pass over the old-model faces ran to the end and saved people were re-derived (this launch).
    private var faceChangeHandled = false
    @Published var bursts: [UUID: [Int]] = [:]     // per album: burst group of each found photo (display only)

    /// Near-identical shots of one moment -> one stack ("+N similar"); every photo stays in the album.
    func refreshBursts() async {
        let entries = await index.entries
        for r in results where r.done {
            let v = await index.vectors(r.found).map { $0 ?? [] }, t = r.found.map { entries[$0]?.taken }
            if v.allSatisfy({ !$0.isEmpty }) { bursts[r.id] = burstIds(vectors: v, taken: t) }
        }
    }

    let index = PhotoIndex()
    let judge = Judge()
    let people = PeopleStore()
    let subjects = SubjectStore()
    var embedder: Embedder?
    var faceEngine: FaceEngine?
    // "Who is X?" sheet (People.swift FacePicker)
    @Published var askWhichFace = false
    @Published var faceGroupsShown: [FaceGroup] = []
    @Published var faceSuggestion: Int?
    @Published var faceOthers: [Int] = []
    @Published var faceNote = ""
    @Published var askingFor = "me"
    // "Show me Max" sheet (SearchView.swift SubjectAskSheet)
    @Published var askSubject = false
    @Published var pendingSubject: SubjectAsk?
    var owner = "me"
    var history: [String] = []
    var currentPlan: Plan?
    var currentAsks: [SubjectAsk] = []
    var searchTask: Task<Void, Never>?
    private var storesLoaded = false
    private var initialIndexDone = false
    private var groupsStale = true
    private var observer: LibraryObserver?
    private var indexChain: Task<Void, Never>?
    /// The queued / running foreground face-upgrade chunk (at most one; enqueueFaceUpgradeChunk).
    private var faceChunk: Task<Void, Never>?
    private var faceChunkNumber = 0
    /// The queued / running foreground frames-pass chunk (at most one; enqueueVideoFramesChunk).
    private var framesChunk: Task<Void, Never>?
    private var framesChunkNumber = 0
    /// Pending re-try of the foreground chunks after one COULD NOT RUN (retryChunksSoon).
    private var retryChunks: Task<Void, Never>?
    private var pendingChange = LibraryChange()
    private var changeDebounce: Task<Void, Never>?


    // The developer-only measurement runs below call probes that only exist in DEBUG builds, so they
    // must be DEBUG-only themselves: without this the app does not compile for Release at all.
    #if DEBUG
    /// Developer-only (launch argument `-local448`): M10's 448 px request shape, but with the network OFF, to find
    /// out whether that row was a download at all or a rendition already on the phone.
    func runLocal448() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        if embedder == nil { embedder = try? Embedder() }
        await loadStores()
        indexStatus = await index.summary() ?? ""
        lastQuery = "-local448 (developer measurement)"
        planNote = "Asking for 448 px with the network OFF..."
        stage = .ready
        let r = await PhotoLibrary.localAt448()
        let s = r.longSides.sorted()
        planNote = """
            448 px, resizeMode .fast, network OFF, \(r.measured) iCloud-only photos strided over the library \
            (scanned \(r.scanned); \(r.originalLocal) had the original, \(r.unknown) unknown).
            >= 448 px: \(r.atLeast448)   224-447: \(r.from224to447)   < 224: \(r.under224)   nothing: \(r.nothing)
            long side min \(Int(s.first ?? 0)) / median \(s.isEmpty ? 0 : Int(s[s.count / 2])) / max \(Int(s.last ?? 0))
            """
    }

    /// Developer-only (launch argument `-queueSizes`, docs/MAC_INBOX.md M17): what the iCloud download queue is made
    /// of (photos vs videos, video lengths, photos' local rendition at the index's 448 .fast ask with the network off).
    /// Early return like -localSizes: it does NOT index, so the queue is measured as the previous build left it.
    func runQueueSizes() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        if embedder == nil { embedder = try? Embedder() }
        await loadStores()
        lastQuery = "-queueSizes (developer measurement, M17)"
        planNote = "Reading the download queue..."
        stage = .ready
        let assets = PhotoLibrary.allAssets()
        let queue = await index.downloadQueue(assets: assets, retryFailed: false)
        let withRetry = await index.downloadQueue(assets: assets, retryFailed: true)
        let notRead = await index.notRead
        var reasons = [String: Int]()
        for id in withRetry { reasons[notRead[id]?.rawValue ?? "indexed from a stand-in", default: 0] += 1 }
        let r = await PhotoLibrary.queueBreakdown(queue)
        let ls = r.renditionLongSides.sorted()
        func b(_ d: [String: Int], _ keys: [String]) -> String { keys.map { "\($0): \(d[$0] ?? 0)" }.joined(separator: "   ") }
        planNote = """
            DOWNLOAD QUEUE (foreground pass, no failed retries): \(r.queue) = \(r.photos) photos + \(r.videos) videos \
            (\(r.missing) no longer in the library). With the charger's retries: \(withRetry.count); by reason: \
            \(reasons.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", ")).
            VIDEOS by length: \(b(r.videoDuration, ["<10s", "10-30s", "30-80s", ">=80s"]))
            video total \(String(format: "%.1f", r.videoSeconds / 3600)) h; frames to index \(r.videoFramesPlanned) \
            (\(r.videos > 0 ? String(format: "%.1f", Double(r.videoFramesPlanned) / Double(r.videos)) : "0") per video)
            PHOTOS, 448 px .fast network OFF, \(r.photosMeasured) measured (strided over the queue's photos):
            \(b(r.rendition, [">=448", "224-447", "<224", "nothing"]))
            long side min \(Int(ls.first ?? 0)) / median \(ls.isEmpty ? 0 : Int(ls[ls.count / 2])) / max \(Int(ls.last ?? 0))
            """
        if let data = try? JSONEncoder().encode(r),
           let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            try? data.write(to: dir.appendingPathComponent("queue_sizes.json"))
        }
    }

    /// Developer-only (launch argument `-downloadBench`, docs/MAC_INBOX.md M10): how long an iCloud-only photo
    /// takes to arrive at 448 / 896 / 1280 px, and whether iCloud sends a derivative or the whole original.
    /// Updates the screen as it goes: the whole run is many minutes and is read off screenshots.
    func runDownloadBench() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        if embedder == nil { embedder = try? Embedder() }
        await loadStores()
        indexStatus = await index.summary() ?? ""
        lastQuery = "-downloadBench (developer measurement)"
        planNote = "Starting download benchmark..."
        stage = .ready
        let rows = await PhotoLibrary.downloadBench(perSide: 50) { rows in
            Task { @MainActor in self.planNote = AppModel.renderDownloadRows(rows) }
        }
        planNote = AppModel.renderDownloadRows(rows) + "\n(done)"
    }

    /// Static so it can be captured by the benchmark's @Sendable progress closure (a local closure is not Sendable).
    static func renderDownloadRows(_ rows: [PhotoLibrary.DownloadRow]) -> String {
            rows.map { r in
                let secs = r.seconds.sorted()
                let med = secs.isEmpty ? 0 : secs[secs.count / 2]
                let longs = r.longSides.sorted()
                let medLong = longs.isEmpty ? 0 : longs[longs.count / 2]
                let per = r.done > 0 ? r.wallClock / Double(r.done) : 0
                return "side \(Int(r.side)) x\(r.parallel): \(r.done)/\(r.wanted) done, \(r.failed) failed; "
                     + "median \(String(format: "%.1f", med)) s/photo, wall \(String(format: "%.1f", per)) s/photo; "
                     + "long side median \(Int(medLong)); original arrived \(r.originalBecameLocal), "
                     + "derivative only \(r.stayedRemote)"
            }.joined(separator: "\n")
    }



    /// Developer-only (launch argument `-selfCheck`, docs/MAC_INBOX.md M3): run the Self-check without anyone
    /// tapping, and stop there. It needs only the bundled Core ML models, so it must NOT sit behind start()'s
    /// `await t.value`: indexing awaits the whole face re-embed (8,430 photos, ~13 min) before it returns.
    func runSelfCheck() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        if embedder == nil { embedder = try? Embedder() }
        if faceEngine == nil { faceEngine = try? FaceEngine() }
        lastQuery = "-selfCheck (developer measurement)"
        planNote = "Running self-check..."
        stage = .ready
        let lines = SelfCheck.run(embedder: embedder, faces: faceEngine)
        planNote = lines.joined(separator: "\n")
            + "\n(app memory available now: "
            + String(format: "%.2f", Double(os_proc_available_memory()) / 1_073_741_824) + " GB)"
        keepIndexingAfterDeveloperRun()
    }

    /// Developer-only (launch argument `-localSizes`, docs/MAC_INBOX.md M4): measure what PhotoKit returns for
    /// iCloud-only photos at 896 px with the network OFF. Writes the numbers to Documents/local_sizes.json (pulled
    /// off the phone with `xcrun devicectl device copy from`) and puts the histogram on screen so a screenshot
    /// records it too. Sizes only: no identifiers and no photo content leave the device.
    func runLocalSizes() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        // The index counts below are read from the saved store, so it has to be loaded: this is an early-return
        // path and does not go through start()'s normal setup. Without this the counts read 0 of everything.
        if embedder == nil { embedder = try? Embedder() }
        await loadStores()
        lastQuery = "-localSizes (developer measurement)"
        planNote = "Measuring what PhotoKit returns at 896 px with the network off..."
        stage = .ready
        let rep = await PhotoLibrary.localCopySizes(side: 896, wanted: 200)
        // M9(a): what is actually in the index right now.
        let entries = await index.entries, notRead = await index.notRead, lowRes = await index.lowResCount
        let libraryCount = PhotoLibrary.allAssets().count
        var byReason = [String: Int]()
        for (_, r) in notRead { byReason[r.rawValue, default: 0] += 1 }
        let storeLine = await index.storeLine()
        let indexLine = storeLine + "\nINDEX: \(entries.count) of \(libraryCount) library items have an image vector "
            + "(\(lowRes) of those from a smaller local copy). Not read: "
            + (byReason.isEmpty ? "none"
               : byReason.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))
        let ex = rep.exactLongSides.sorted(), nat = rep.nativeLongSides.sorted(), bst = rep.bestLongSides.sorted()
        func med(_ a: [Double]) -> Int { a.isEmpty ? 0 : Int(a[a.count / 2]) }
        planNote = indexLine + "\n" + """
            896 px request, network OFF, \(rep.measured) iCloud-only photos sampled across the whole library \
            (scanned \(rep.scanned); \(rep.originalLocal) had the original on the phone, \
            \(rep.unknownAvailability) unknown).
            >= 806 px (app calls this full): \(rep.atLeast806)
            448-805 px (indexable, judge must download): \(rep.from448to805)
            < 448 px (too small to index): \(rep.below448)
            nothing returned: \(rep.nothing)
            OF THE >= 806, UPSCALED FROM A SMALLER LOCAL RENDITION: \(rep.upscaled)
            exact long side  min \(Int(ex.first ?? 0)) / median \(med(ex)) / max \(Int(ex.last ?? 0))
            native long side min \(Int(nat.first ?? 0)) / median \(med(nat)) / max \(Int(nat.last ?? 0))
            LARGEST local rendition (opportunistic, degraded kept, max size, network off):
            < 224 px: \(rep.bestUnder224)   224-447: \(rep.best224to447)   448-805: \(rep.best448to805)   \
            >= 806: \(rep.bestAtLeast806)   nothing: \(rep.bestNothing)
            best long side min \(Int(bst.first ?? 0)) / median \(med(bst)) / max \(Int(bst.last ?? 0))
            """
        if let data = try? JSONEncoder().encode(rep),
           let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            try? data.write(to: dir.appendingPathComponent("local_sizes.json"))
        }
    }

    #endif
    /// Developer-only (launch argument `-demoUI`): skip the model and fill example albums from the library's photos, so
    /// the Simulator (no GPU for MLX) can show and screenshot the search screens. Never used in normal runs.
    func startDemoUI() async {
        _ = await PhotoLibrary.requestAccess()
        let ids = PhotoLibrary.allAssets().map(\.id)
        func album(_ name: String, _ ids: [String], note: String) -> AlbumResult {
            var r = AlbumResult(name: name); r.found = ids; r.judged = 120; r.inScope = 120; r.note = note; r.done = true; return r
        }
        lastQuery = "photos of me looking heavier vs photos of me looking fit"
        planNote = "Two albums of you; each photo goes to the album it clearly matches more."
        results = [album("me heavier", Array(ids.prefix(6)), note: "Demo data (Simulator): not a real search."),
                   album("me fit", Array(ids.dropFirst(6).prefix(6)), note: "Demo data (Simulator): not a real search.")]
        stage = .ready
    }

    private var startTask: Task<Void, Never>?
    /// From RootView's .task. start() runs in its own task: SwiftUI cancels a view's .task when the view goes away
    /// (here: as soon as the stage changes), which would cancel the model download and the indexing loop with it.
    /// A failure in words the person can act on. The raw text is still shown underneath, small, for us.
    static func friendlyFailure(_ e: String) -> String {
        let t = e.lowercased()
        if t.contains("-1005") || t.contains("network connection was lost") || t.contains("-1009")
            || t.contains("offline") || t.contains("-1001") || t.contains("timed out") {
            return "The download was interrupted. find pics needs Wi-Fi to fetch its AI once (about 3 GB); "
                 + "it picks up where it left off, so nothing is wasted."
        }
        if t.contains("judge") { return "The on-phone AI could not start. Tap Try again; if it keeps failing, reopen find pics." }
        if t.contains("space") || t.contains("no space") { return "This iPhone is out of storage. Free up a few GB and tap Try again." }
        return "Something went wrong. Tap Try again."
    }

    func launch() {
        guard startTask == nil else { return }
        startTask = Task { @MainActor in await self.start(); self.startTask = nil }
    }

    /// The one line the search screen shows when the downloaded judge does not fit in this app's memory.
    private func noteQwenOutOfMemory(_ availableGB: Double) {
        let have = String(format: "%.1f", availableGB)
        let why = "This iPhone lets an app use \(have) GB of memory, too little for the downloaded judge (about 3.6 GB)"
        #if canImport(FoundationModels)
        if #available(iOS 27.0, *) {
            if let reason = AppleJudge.unavailableReason {
                engineNote = "\(why), and Apple's built-in model is unavailable (\(reason)). Searches by date, media "
                           + "type and time of day still work; searches that have to look at a photo do not."
            } else {
                engineNote = "\(why), so photos are judged by Apple's built-in model instead."
            }
            return
        }
        #endif
        engineNote = "\(why), and this iOS is too old for Apple's built-in model. Searches by date, media type and "
                   + "time of day still work; searches that have to look at a photo do not."
    }

    /// Delete partial model downloads left in tmp/ by failed attempts (FindPicsCore.isStaleDownloadTemp). Only the app's
    /// own temporary directory; never anything in the photo library.
    nonisolated static func removeStaleDownloadTemps() {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: fm.temporaryDirectory,
                                                      includingPropertiesForKeys: [.contentModificationDateKey]) else { return }
        for u in items {
            let modified = (try? u.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? Date()
            if isStaleDownloadTemp(name: u.lastPathComponent, ageSeconds: -modified.timeIntervalSinceNow) { try? fm.removeItem(at: u) }
        }
    }

    func start() async {
        Self.removeStaleDownloadTemps()      // MAC 10-10: a 2.69 GB partial download was orphaned in tmp/
        #if DEBUG
        if ProcessInfo.processInfo.arguments.contains("-demoUI") { await startDemoUI(); return }
        if ProcessInfo.processInfo.arguments.contains("-localSizes") { await runLocalSizes(); return }
        if ProcessInfo.processInfo.arguments.contains("-selfCheck") { await runSelfCheck(); return }
        if ProcessInfo.processInfo.arguments.contains("-downloadBench") { await runDownloadBench(); return }
        if ProcessInfo.processInfo.arguments.contains("-local448") { await runLocal448(); return }
        if ProcessInfo.processInfo.arguments.contains("-queueSizes") { await runQueueSizes(); return }
        #endif
        // Outside the #if DEBUG on purpose: M14 wants these numbers from a RELEASE build, where everything above is
        // compiled out. Costs two Date() reads per stage when the argument is absent.
        if ProcessInfo.processInfo.arguments.contains("-timeIndex") { IndexTiming.on = true }
        // BELOW THE #endif ON PURPOSE. -runQuery and -runQueries must work on a RELEASE build: Debug is ~85x
        // slower on the Swift pixel loops, and the whole point is to measure the shipped thing. Both sat inside
        // the DEBUG block and silently did nothing on three separate launches (10-10 13:25, 13:28, 13:33) - the
        // app just indexed and I read the ordinary screen as "the search found nothing".
        if let q = AppModel.debugQuery() { await runDebugQuery(q); return }
        if ProcessInfo.processInfo.arguments.contains("-runQueries") { await runBatchQueries(); return }
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        do {
            if embedder == nil { embedder = try Embedder() }
            if faceEngine == nil { faceEngine = try? FaceEngine() }
            await loadStores()
            // The 4-bit 4B needs ~3.1 GB of weights plus working memory; below this iOS would kill the app mid-load.
            // Too little memory is NOT a reason to stop: indexing, the library observer and the charger task need far
            // less, and Apple's built-in model runs out-of-process, so searches still work through it. Only the
            // downloaded Qwen judge/planner is given up (and with it, its 3.1 GB download and consent screen).
            let availableGB = Double(os_proc_available_memory()) / 1_073_741_824
            qwenOutOfMemory = availableGB > 0 && availableGB < 3.6
            if qwenOutOfMemory {
                noteQwenOutOfMemory(availableGB)
            } else {
                guard downloadAccepted else { stage = .askDownload; return }
                try await judge.load { p in Task { @MainActor in self.stage = .downloading(p) } }
            }
            stage = .indexing(0, 0)
            startObserver()
            // the local pass blocks the first screen (progress); the iCloud pass then continues behind the search screen
            let t = enqueueIndexing { await self.indexLibrary(purpose: .indexForeground, retryFailed: false) }
            await t.value
            if case .indexing = stage { stage = .ready }
            BackgroundIndexing.schedule()

        } catch { stage = .failed(String(describing: error)) }
    }

    /// True once all three stores are read (retried while the phone is still locked since a restart).
    @discardableResult private func loadStores() async -> Bool {
        if storesLoaded { return true }
        let a = await people.load(), b = await subjects.load()
        let c = await index.load(progress: { p in Task { @MainActor in self.indexConversion = p < 1 ? p : nil } })
        indexConversion = nil
        storesLoaded = a && b && c
        indexStatus = await index.summary() ?? ""
        videoFrames = await index.videoFramesCounts()
        return storesLoaded
    }

    // MARK: keeping the index current

    /// What indexing needs (no judge, no download consent: the image / face models are bundled).
    private func prepareIndexing() async -> Bool {
        guard PhotoLibrary.hasAccess else { return false }
        if embedder == nil { embedder = try? Embedder() }
        if faceEngine == nil { faceEngine = try? FaceEngine() }
        guard await loadStores() else { return false }
        // a face-model change: where each saved person's faces are must be known BEFORE any face is re-embedded
        await people.recordSources(index: index)
        let stale = await index.staleFaceCount()
        let waitingPeople = await people.awaitingRederive()
        faceChangePending = !faceChangeHandled && (stale > 0 || waitingPeople)
        return embedder != nil
    }

    /// One indexing job at a time (launch, library changes, the charger task): each waits for the one before.
    @discardableResult
    private func enqueueIndexing(_ job: @escaping @MainActor @Sendable () async -> Void) -> Task<Void, Never> {
        let prev = indexChain
        let t = Task { @MainActor in
            await prev?.value
            if Task.isCancelled { return }
            await job()
        }
        indexChain = t
        return t
    }

    private func progressHandler() -> @Sendable (IndexProgress) -> Void {
        { p in Task { @MainActor in AppModel.shared.showProgress(p) } }
    }

    private func faceProgressHandler() -> @Sendable (FaceReindexProgress) -> Void {
        { p in Task { @MainActor in AppModel.shared.showFaceProgress(p) } }
    }

    private func videoFramesHandler() -> @Sendable (VideoFramesProgress) -> Void {
        { p in Task { @MainActor in AppModel.shared.videoFrames = p } }
    }

    /// Re-embedding faces does not block image searches: the first screen gives way to the search screen.
    /// Reports arrive through Tasks, maybe out of order: an older pass's, or one after its pass's final, is ignored.
    private func showFaceProgress(_ p: FaceReindexProgress) {
        if let c = faceReindex, p.pass < c.pass || (p.pass == c.pass && !c.running) { return }
        if case .indexing = stage { stage = .ready }
        faceReindex = p
    }

    /// One line for the search screen while / after faces are re-embedded for a new face model.
    var faceReindexNote: String {
        guard let p = faceReindex else {
            return faceChangePending ? "find pics is switching to a new face-recognition model: it re-reads the photos with "
                + "faces after adding new photos. Searches for people wait until then; other searches work now." : ""
        }
        if p.running {
            return "Updating face recognition (new face model): \(p.done) of \(p.total) photos with faces. "
                 + "Searches for people wait until it finishes; other searches work now."
        }
        return p.waiting > 0 ? "\(p.waiting) photo(s) with faces could not be re-read for the new face model yet (in iCloud: "
                             + "retried on Wi-Fi while charging); people searches leave them out until then." : ""
    }

    private func showProgress(_ p: IndexProgress) {
        let kind = p.faces ? 2 : p.downloading ? 1 : 0
        if progressStart?.kind != kind || p.done < (progressStart?.done ?? 0) { progressStart = (kind, p.done, Date()) }
        if !initialIndexDone, !p.downloading, !p.faces, case .indexing = stage {
            indexProgress = p; stage = .indexing(p.done, p.total); return   // indexProgress feeds progressETA (MAC 07:22)
        }
        if p.downloading || p.faces, case .indexing = stage { stage = .ready }      // local pass done: searchable now
        indexProgress = p
    }

    /// The whole library against the index: deleted assets out, new ones in, then iCloud downloads if allowed.
    private func indexLibrary(purpose: FetchPurpose, retryFailed: Bool) async {
        guard await prepareIndexing(), let emb = embedder else { return }
        let assets = PhotoLibrary.allAssets()
        libraryCount = assets.count
        let indexed = Array(await index.entries.keys)
        await index.remove(removedFromLibrary(indexed: indexed, library: Set(assets.map(\.id))))
        let fp = await index.update(assets: assets, embedder: emb, faceEngine: faceEngine, purpose: purpose, retryFailed: retryFailed,
                                    progress: progressHandler(), faceProgress: faceProgressHandler())
        await afterIndexChange(facePass: fp)
        if purpose == .indexBackground {
            // the charger task: the face upgrade and the frames pass (FindPicsCore/LazyVideo.swift) take turns in chunks,
            // so neither waits behind the other for the whole grant. A pass whose chunk made no progress stops.
            var facesLeft = Int.max, framesLeft = Int.max, first = true
            while !Task.isCancelled, facesLeft > 0 || framesLeft > 0 {
                if facesLeft > 0 {
                    // nil = could not run in this grant at all; 0 or no progress = done with it for this grant
                    let f = await improveFaces(purpose: purpose, retryFailed: retryFailed && first, limit: faceUpgradeForegroundChunk)
                    facesLeft = (f.map { $0 < facesLeft ? $0 : 0 }) ?? 0
                }
                if framesLeft > 0, !Task.isCancelled {
                    let v = await lookInsideVideos(purpose: purpose, retryFailed: retryFailed && first, limit: videoFramesChunk)
                    framesLeft = (v.map { $0 < framesLeft ? $0 : 0 }) ?? 0
                }
                first = false
            }
            indexProgress = nil
        } else { enqueueImprovementPasses() }
    }

    /// FRAMES PASS (FindPicsCore/LazyVideo.swift), after the cover-frame pass: sweep 1 gives every cover-frame-only video
    /// its middle and end frames, then sweep 2 samples every 4 s. Returns how many videos still wait (0 when it could
    /// not run now).
    @discardableResult
    /// nil (NOT 0) when it could not run at all: "could not run now" and "no videos left" are different answers and
    /// collapsing them into 0 silently ended the foreground chain (JOURNAL 10-09 08:25).
    private func lookInsideVideos(purpose: FetchPurpose, retryFailed: Bool, limit: Int?) async -> Int? {
        guard let emb = embedder, !Task.isCancelled else { return nil }
        let left = await index.sampleVideoFrames(embedder: emb, faceEngine: faceEngine, purpose: purpose, limit: limit,
                                                 retryFailed: retryFailed, progress: videoFramesHandler())
        groupsStale = true                     // video faces changed: "Who is X?" groups are rebuilt when next needed
        return left
    }

    /// The next foreground frames-pass chunk, behind whatever indexing is queued (new photos are indexed in between);
    /// it queues the one after it while videos are left. Only while the app is in front (the charger task does it in
    /// the background and cancels a chunk still queued or running; `framesChunkNumber` then tells the old chunk).
    private func enqueueVideoFramesChunk(retryFailed: Bool = false) {
        guard framesChunk == nil else { return }
        framesChunkNumber += 1
        let n = framesChunkNumber
        framesChunk = enqueueIndexing {
            var left: Int?
            if UIApplication.shared.applicationState == .active, await self.prepareIndexing() {
                left = await self.lookInsideVideos(purpose: .indexForeground, retryFailed: retryFailed, limit: videoFramesChunk)
            }
            guard self.framesChunkNumber == n else { return }
            self.framesChunk = nil
            guard !Task.isCancelled else { return }
            // FindPicsCore.passStep decides this, with tests (PassStepTests): "could not run" and "everything
            // left already failed once" must never be read as "finished".
            let c = await self.index.videoFramesCounts()
            switch passStep(left: left, checked: c.sweep2Done, total: c.videos) {
            case .more: self.enqueueVideoFramesChunk()
            case .retryAfterFailures: self.retryPassAfterFailures()
            case .finished: break
            }
        }
    }

    /// FACE UPGRADE (FindPicsCore/FaceUpgrade.swift): photos whose faces were found on the ~480 px index read are read
    /// again at faceReadSide, only where indexing may download (FindPicsCore.iCloudDownloadAllowed: unconstrained
    /// Wi-Fi). The charger task runs it to the end (or until iOS takes the time back); the foreground in chunks of
    /// faceUpgradeForegroundChunk photos, each its own indexing job, so new photos are indexed in between.
    /// Returns how many photos still need it, or nil when it COULD NOT RUN (no face engine, cancelled, or this
    /// network does not allow iCloud downloads). nil and 0 must stay distinct: see lookInsideVideos.
    /// `limit`: at most that many photos (default: a foreground chunk, or all in the background).
    @discardableResult
    private func improveFaces(purpose: FetchPurpose, retryFailed: Bool, limit: Int? = nil) async -> Int? {
        guard let fe = faceEngine, !Task.isCancelled, iCloudDownloadAllowed(purpose, NetworkState.shared.path) else { return nil }
        let foreground = purpose != .indexBackground
        let run = await index.upgradeFaces(faceEngine: fe, purpose: purpose, limit: limit ?? (foreground ? faceUpgradeForegroundChunk : nil),
                                           retryFailed: retryFailed, progress: progressHandler())
        if !run.upgraded.isEmpty {
            groupsStale = true                    // "Who is X?" groups prefer the newly checked faces
            // saved people whose reference faces were just re-read: new vectors of the same faces (or asked again)
            let reask = await people.refreshAfterUpgrade(index: index, upgraded: run.upgraded)
            if !reask.isEmpty { Logger().info("find pics: face upgrade; \(reask.count) saved person(s) will be asked again") }
        }
        return run.left
    }

    /// The next foreground face-upgrade chunk, behind whatever indexing is queued; it queues the one after it while
    /// photos are left. Only while the app is in front: in the background the charger task does it (and cancels a
    /// chunk still queued or running; `faceChunkNumber` then tells the old chunk it was replaced).
    private func enqueueFaceUpgradeChunk(retryFailed: Bool = false) {
        guard faceChunk == nil else { return }
        faceChunkNumber += 1
        let n = faceChunkNumber
        faceChunk = enqueueIndexing {
            var left: Int?
            if UIApplication.shared.applicationState == .active, await self.prepareIndexing() {
                left = await self.improveFaces(purpose: .indexForeground, retryFailed: retryFailed)
            }
            guard self.faceChunkNumber == n else { return }
            self.faceChunk = nil
            guard !Task.isCancelled else { return }
            // See FindPicsCore.passStep (tested). Faces additionally hands the chain over the moment it has
            // nothing to do RIGHT NOW, because its own completion target can be unreachable - see below.
            let (checked, total) = await self.index.faceUpgradeCounts()
            switch passStep(left: left, checked: checked, total: total) {
            case .more: self.enqueueFaceUpgradeChunk(); return
            case .retryAfterFailures, .finished: break
            }
            // THE 7-MINUTE DEATH (MEASURED 10-10 04:18, Reza watching the counter freeze at 47,041 of 79,993).
            // PhotoIndex.upgradeFaces computes `left` as "needs upgrade AND NOT in upgradeSkipped", and every photo
            // whose face read failed this launch goes into upgradeSkipped (almost always a transient iCloud read).
            // Nothing clears that set except retryFailed: true, which the foreground chain never passed. So after a
            // few minutes every remaining photo had failed once, left became 0, and the chain ended as if the pass
            // were COMPLETE - with 33,000 photos still unupgraded. Only relaunching the app (a fresh PhotoIndex,
            // empty skip set) restarted it, which is why it ran ~7 min per launch all night.
            // HAND OVER TO FRAMES NOW. left == 0 means the faces pass can do nothing more RIGHT NOW, which is not
            // the same as checked == total: Reza's library stalled at 79,576 of 79,615 because the last 39 photos
            // are simply unreadable (MEASURED 10-10 06:33). Gating the frames pass on faces reaching 100% would
            // therefore block it FOREVER - a bug I introduced with the faces-first ordering a few hours earlier.
            // So: start frames unconditionally, and keep retrying the stragglers in the background.
            self.indexProgress = nil
            self.enqueueVideoFramesChunk()
            if checked < total { self.retryPassAfterFailures() }
        }
    }

    /// Both passes exhausted everything they had not already failed on. The failures are transient (an iCloud read
    /// that did not come back), so wait and then try them AGAIN with retryFailed, which clears the skip sets - the
    /// same thing a relaunch used to do by accident. Not a tight loop: a minute between sweeps.
    private func retryPassAfterFailures() {
        guard retryChunks == nil else { return }
        retryChunks = Task { @MainActor in
            try? await Task.sleep(for: .seconds(60))
            self.retryChunks = nil
            guard !Task.isCancelled, self.stage == .ready else { return }
            // Both, not either: faces may have a permanently unreadable tail that never completes, and the frames
            // pass must not wait on it (see enqueueFaceUpgradeChunk).
            let (checked, total) = await self.index.faceUpgradeCounts()
            if checked < total { self.enqueueFaceUpgradeChunk(retryFailed: true) }
            self.enqueueVideoFramesChunk(retryFailed: true)
        }
    }

    /// FACES FIRST, THEN FRAMES. Both passes go through the SAME serial indexing chain, so running them together
    /// halves each one's rate. They are not equally urgent (measured on Reza's phone, 10-10):
    ///   faces  ~190 photos/min, ~3 h left  - and "photos of me" is WRONG until it finishes
    ///   frames  ~10 videos/min, ~43 h left - and every video is ALREADY findable by its cover frame; the pass
    ///           only lifts video recall from 191/303 to 219/303 (eval/RESULTS.md 37)
    /// So the cheap, urgent, correctness-affecting pass goes first and finishes in hours instead of crawling
    /// alongside a two-day job. Nothing is dropped: frames starts the moment faces has nothing left.
    func enqueueImprovementPasses() {
        Task { @MainActor in
            let (checked, total) = await self.index.faceUpgradeCounts()
            if checked < total { self.enqueueFaceUpgradeChunk() } else { self.enqueueVideoFramesChunk() }
        }
    }

    /// A foreground chunk that COULD NOT RUN is not the same as one that found nothing left to do, and must not end
    /// the chain. Two ways it happens: the screen went off (applicationState stops being .active while the app is
    /// still frontmost) and the network stopped allowing iCloud downloads (Low Data Mode, or cellular).
    /// MEASURED 10-09: the chain died at 04:19 when the screen went off and did NOT restart even with the app back in
    /// front for four hours, because becameActive() only fires on a real background -> active transition and the
    /// screen going off and on again is not one. 14,875 of 80,156 photos sat still the whole time. Retry on a timer.
    private func retryChunksSoon() {
        guard retryChunks == nil else { return }
        retryChunks = Task { @MainActor in
            try? await Task.sleep(for: .seconds(30))
            self.retryChunks = nil
            guard !Task.isCancelled, self.stage == .ready else { return }
            self.enqueueFaceUpgradeChunk(); self.enqueueVideoFramesChunk()
        }
    }

    /// `facePass`: the last face re-embedding pass of this indexing run (nil: none ran).
    private func afterIndexChange(facePass: FaceReindexProgress? = nil) async {
        initialIndexDone = true
        groupsStale = true                 // face groups are rebuilt when the face picker next needs them
        indexStatus = await index.summary() ?? ""
        indexProgress = nil
        if let fp = facePass { faceReindex = fp }     // its final state (reports still in flight are ignored: same pass, final)
        // Saved people of the old face model get the new vectors of the same faces once every old-model photo was tried
        // (none left, or a pass ran to the end; the ones still waiting for iCloud count as not found). Never without a
        // face model: then nothing was re-read and everyone would be asked again for nothing.
        let stale = await index.staleFaceCount()
        if faceEngine != nil, !Task.isCancelled, stale == 0 || facePass?.finished == true {
            let reask = await people.rederive(index: index, faceEngine: faceEngine)
            if !reask.isEmpty { Logger().info("find pics: face model changed; \(reask.count) saved person(s) will be asked again") }
            faceChangePending = false; faceChangeHandled = true
        }
        if let p = faceReindex, !p.running, p.waiting == 0 { faceReindex = nil }
    }

    /// PHPhotoLibraryChangeObserver while the app is open: new photos / videos indexed, deleted ones dropped.
    private func startObserver() {
        guard observer == nil else { return }
        observer = LibraryObserver { change in Task { @MainActor in AppModel.shared.libraryChanged(change) } }
    }

    private func libraryChanged(_ c: LibraryChange) {
        pendingChange.inserted += c.inserted; pendingChange.removed += c.removed; pendingChange.full = pendingChange.full || c.full
        changeDebounce?.cancel()            // a burst (an import, a shared album syncing) -> one pass, 2 s after the last change
        changeDebounce = Task { @MainActor in
            try? await Task.sleep(for: .seconds(2))
            if Task.isCancelled { return }
            self.flushLibraryChange()
        }
    }

    private func flushLibraryChange() {
        let c = pendingChange; pendingChange = LibraryChange()
        enqueueIndexing {
            if c.full { await self.indexLibrary(purpose: .indexForeground, retryFailed: false); return }
            let ins = Set(c.inserted.map(\.id))
            await self.index.remove(c.removed.filter { !ins.contains($0) })
            guard !c.inserted.isEmpty, await self.prepareIndexing(), let emb = self.embedder else { await self.afterIndexChange(); return }
            let fp = await self.index.update(assets: c.inserted, embedder: emb, faceEngine: self.faceEngine, purpose: .indexForeground,
                                             retryFailed: false, progress: self.progressHandler(), faceProgress: self.faceProgressHandler())
            await self.afterIndexChange(facePass: fp)
            self.enqueueImprovementPasses()         // new photos' faces at 448; new videos on their cover frame
        }
    }

    /// Back in the foreground: catch up on what changed while the app was suspended (and retry waiting downloads if
    /// the phone is on Wi-Fi now).
    func becameActive() {
        guard initialIndexDone, stage == .ready else { return }
        enqueueIndexing { await self.indexLibrary(purpose: .indexForeground, retryFailed: false) }
    }

    /// The charger/idle BGProcessingTask: index new photos, download iCloud-only originals on Wi-Fi (retrying failed
    /// ones), until done or iOS takes the time back (`cancel`). True when it finished.
    func runBackgroundIndexing(cancel: BackgroundIndexing.CancelBox) async -> Bool {
        BackgroundIndexing.schedule()            // the next charger session
        faceChunk?.cancel(); faceChunk = nil; faceChunkNumber += 1   // a foreground face-upgrade chunk must not hold the background time
        framesChunk?.cancel(); framesChunk = nil; framesChunkNumber += 1   // nor a foreground frames-pass chunk
        let t = enqueueIndexing { await self.indexLibrary(purpose: .indexBackground, retryFailed: true) }
        cancel.set(t)
        await t.value
        return !cancel.isCancelled
    }

    /// F5 (docs/MAC_INBOX.md). A developer launch argument takes start()'s early-return path, which skips every
    /// bit of normal setup - including the indexing chain. NOTHING then indexes, and NOTHING on screen says so:
    /// the banner still shows its last counters and the process looks healthy. That cost ten hours on 10-09, when
    /// the app sat in -runQuery mode from 10:37 to 20:25 and I did not notice because the app was "running".
    /// So every developer run now starts the normal indexing work too. The search or measurement still happens;
    /// it just no longer silently parks the whole product.
    func keepIndexingAfterDeveloperRun() {
        initialIndexDone = true
        startObserver()
        enqueueIndexing { await self.indexLibrary(purpose: .indexForeground, retryFailed: false) }
        BackgroundIndexing.schedule()
    }

    // -runQuery / -runQueries / the F6 list are OUTSIDE #if DEBUG on purpose: they must run on a
    // RELEASE build (Debug is ~85x slower on the Swift pixel loops, and the point is to measure the
    // shipped thing). They were inside it and silently did nothing on three launches, 10-10.
    /// The ten fixed F6 queries (docs/MAC_INBOX.md): five Apple Photos is known to handle, five that need real
    /// understanding. Fixed in advance so neither side can be tuned to the result.
    static let f6Queries = [
        "photos of a dog", "food photos", "selfies", "beach photos", "sunset photos",
        "me looking heavier vs me looking fit", "photos where I look tired",
        "the whiteboard with the diagram on it", "photos of my passport or ID", "the night we got dumplings",
    ]

    /// `-runQueries`: run all ten F6 queries back to back and write the numbers to Documents/f6_results.json,
    /// which the Mac pulls with `devicectl device copy from`. Built 10-10 so the comparison can be run WITHOUT
    /// Reza typing: devicectl cannot inject taps on a physical iPhone, and screenshots proved a terrible
    /// instrument (a crash and a stall look identical in a picture). Results are written after EVERY query, so a
    /// kill half way still leaves everything up to that point.
    func runBatchQueries() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        if embedder == nil { embedder = try? Embedder() }
        if faceEngine == nil { faceEngine = try? FaceEngine() }
        await loadStores()
        // LOAD THE PLANNER, exactly as runDebugQuery does. Without it every query dies with
        // "Error Domain=Judge Code=1" (container nil) in 0 seconds - which is what the first batch produced,
        // 10 rows of nothing (10-10 13:39). start() does this on the normal path; this runner skips start().
        let availableGB = Double(os_proc_available_memory()) / 1_073_741_824
        qwenOutOfMemory = availableGB > 0 && availableGB < 3.6
        if qwenOutOfMemory { noteQwenOutOfMemory(availableGB) }
        else { do { try await judge.load { _ in } } catch { planNote = "planner did not load: \(error)" } }
        indexStatus = await index.summary() ?? ""
        stage = .ready
        var out: [[String: Any]] = []
        let file = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("f6_results.json")
        for q in AppModel.f6Queries {
            Judge.logMem("Q start: \(q)")
            let t0 = Date()
            search(q)
            while busy, !Task.isCancelled { try? await Task.sleep(for: .milliseconds(500)) }
            let secs = Date().timeIntervalSince(t0)
            for r in results {
                out.append(["query": q, "album": r.name, "found": r.found.count, "judged": r.judged,
                            "inScope": r.inScope, "seconds": Int(secs), "note": r.note,
                            "engine": effectiveEngine, "topIds": Array(r.found.prefix(12))])
            }
            if results.isEmpty {
                out.append(["query": q, "album": "", "found": 0, "judged": 0, "inScope": 0,
                            "seconds": Int(secs), "note": planNote, "engine": effectiveEngine, "topIds": []])
            }
            if let file, let d = try? JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted]) {
                try? d.write(to: file)
            }
            Judge.logMem("Q done (\(Int(secs)) s): \(q)")
        }
        planNote = "F6 batch done: \(out.count) rows written to Documents/f6_results.json"
        keepIndexingAfterDeveloperRun()
    }

    /// The text after `-runQuery` on the command line, if any.
    static func debugQuery() -> String? {
        let a = ProcessInfo.processInfo.arguments
        guard let i = a.firstIndex(of: "-runQuery"), i + 1 < a.count else { return nil }
        return a[i + 1]
    }
    /// Developer-only (launch argument `-runQuery "<text>"`, docs/MAC_INBOX.md M5/M6): run ONE search without
    /// anyone typing, so a timed run can be driven from the Mac and repeated identically on another judge.
    /// Like -selfCheck this must not sit behind start()'s indexing await: the index is already on disk, and the
    /// re-embed that await waits for takes ~13 min. It deliberately does NOT re-index first.
    func runDebugQuery(_ q: String) async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        if embedder == nil { embedder = try? Embedder() }
        if faceEngine == nil { faceEngine = try? FaceEngine() }
        await loadStores()
        let availableGB = Double(os_proc_available_memory()) / 1_073_741_824
        qwenOutOfMemory = availableGB > 0 && availableGB < 3.6
        if qwenOutOfMemory {
            noteQwenOutOfMemory(availableGB)
        } else {
            // LOAD THE JUDGE. Without this the run reaches the judge with container == nil and dies with
            // "Error Domain=Judge Code=1" (MEASURED 10-10 02:33, and it cost a test run to find out): start() loads
            // it on the normal path, and this runner deliberately skips start(). Before the increased-memory
            // entitlement the omission was invisible, because every search fell back to Apple's model instead.
            do { try await judge.load { p in Task { @MainActor in self.stage = .downloading(p) } } }
            catch { stage = .failed("judge did not load: \(error)"); return }
        }
        indexStatus = await index.summary() ?? ""
        stage = .ready
        search(q)
        keepIndexingAfterDeveloperRun()
    }

    // MARK: searching

    /// "This specific dog / thing / place": the person picked example photos in the library (SubjectSearch.swift).
    /// A name is remembered with the photos, so "Max at the beach" later needs no picking.
    func searchSubject(ids: [String], name: String, kind: String) {
        guard let embedder = embedder else { return }
        searchTask?.cancel()
        busy = true; lastQuery = "\(name) (\(kind))"; results = [AlbumResult(name: name)]; planNote = ""
        searchTask = Task {
            if name != "it" { await subjects.add(SavedSubject(name: name, kind: kind, words: kind, ids: ids)) }
            var examples: [CIImage] = []
            for id in ids { if let im = await PhotoLibrary.ciImage(id, side: 1280) { examples.append(im) } }
            let engine = SearchEngine(index: index, embedder: embedder, judge: activeJudge)
            do {
                let r = try await engine.runSubject(examples: examples, name: name, kind: kind, album: Album(name: name)) { r in
                    Task { @MainActor in if !self.results.isEmpty { self.results[0] = r } }
                }
                self.results = [r]
            } catch { self.results[0].note = "Search failed: \(error.localizedDescription)"; self.results[0].done = true }
            self.busy = false
            await self.refreshBursts()
        }
    }

    /// Apple's model refuses some photos (safety filter); say how many were left out instead of failing the search.
    func noteAppleRefusals() async {
        #if canImport(FoundationModels)
        if #available(iOS 27.0, *), let aj = activeJudge as? AppleJudge {
            let n = await aj.takeRefused()
            if n > 0 { planNote += " Apple's model refused to look at \(n) photo(s) (its safety filter); they are left out." }
        }
        #endif
    }

    /// A new request, or a follow-up that edits the current search ("only the ones outdoors").
    func search(_ text: String, exhaustive: Bool = false, followUp: Bool = false) {
        guard embedder != nil else { return }
        searchTask?.cancel()
        let today = Day(iso: ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate]))!
        let hist = followUp ? history : [], cur = followUp ? currentPlan : nil
        busy = true; lastQuery = text; results = []; planNote = ""
        searchTask = Task {
            var phase = "planning"      // which half threw: Apple's planner and its judge fail differently
            do {
                // SWAP THE MODELS BACK. execute() unloads the planner to make room for the photo judge, so on
                // the SECOND and later searches the planner is gone and planning dies with "Judge Code=1".
                // MEASURED 10-10 14:24: query 1 found 209 dogs in 42 min, then queries 2-10 each failed in 0 s.
                // Only one of the two ~2.9 GB models fits at a time (2.83 + 2.88 vs ~5.8 GB free), so this is a
                // swap, not a both-resident fix: drop the judge, bring the planner back.
                if !(await judge.isLoaded), !qwenOutOfMemory {
                    await self.visionJudge.unload()
                    try await self.judge.load { _ in }
                }
                let plan = try await activePlanner.plan(text, history: hist, current: cur, today: today)
                self.currentPlan = plan; self.history = hist + [text]; self.planNote = plan.notes
                let saved = await self.subjects.saved
                self.currentAsks = namedSubjects(plan, message: text, history: hist, saved: saved)
                phase = "judging"
                try await self.execute(plan, exhaustive: exhaustive)
            } catch {
                self.planNote = "Could not run this search: \(error)"
                #if DEBUG
                self.planNote += " [phase: \(phase), error type: \(type(of: error))]"
                #endif
            }
            await self.noteAppleRefusals()
            await self.refreshBursts()
            self.busy = false
        }
    }

    /// The current plan again, without planning: after a face or a pet's photos were picked, or "Look at everything".
    func rerun(exhaustive: Bool = false) {
        guard let plan = currentPlan, embedder != nil else { return }
        searchTask?.cancel()
        busy = true; results = []
        let msg = history.last ?? lastQuery, earlier = Array(history.dropLast())
        searchTask = Task {
            do {
                let saved = await self.subjects.saved
                self.currentAsks = namedSubjects(plan, message: msg, history: earlier, saved: saved)
                try await self.execute(plan, exhaustive: exhaustive)
            } catch { self.planNote = "Could not run this search: \(error)" }
            await self.noteAppleRefusals()
            await self.refreshBursts()
            self.busy = false
        }
    }

    private func execute(_ plan0: Plan, exhaustive: Bool) async throws {
        guard let embedder = embedder else { return }
        // effectiveEngine, not engine: when the weights do not fit, the menu may still say qwen3vl/vote but Apple's
        // model is what runs, and loading ~2.5 GB here would be exactly the kill we are avoiding.
        let e = self.effectiveEngine
        if e == "qwen3vl" {
            // THE CRASH REZA HIT ON HIS FIRST REAL SEARCH (10-10 10:08: "i tried photos of a dog and then it quit
            // and opened back up"). The app holds the PLANNER (Qwen3.5-4B, ~2.9 GB, loaded at launch) and was
            // then loading the PHOTO JUDGE (Qwen3-VL-4B, ~2.9 GB) on top of it, plus the embedder, the face
            // engine and the mapped index. That is ~6 GB of weights alone - more than iOS allows even WITH the
            // increased-memory entitlement, so the system killed the app. The startup guard did not catch it
            // because it only ever asked whether ONE model fits.
            // The planner has already produced this search's plan and is not needed again until the next
            // question, and reloading it reads from disk with no download. So: free it, then load the judge.
            await self.judge.unload()
            // MEASURE, then decide. Reza's first two real searches both killed the app (10-10 10:08 and 10:20),
            // and guessing cost two builds. The photo judge needs ~2.9 GB of weights plus its vision tower
            // activations at 896 px; if that does not fit after freeing the planner, FALL BACK to Apple's model
            // with the numbers on screen instead of letting iOS kill the app mid-search.
            let before = Judge.memoryNote()
            // WRITE IT TO DISK FIRST. The load can be a hard kill by iOS (jetsam), not a Swift error we can
            // catch - so anything only shown on screen dies with the app. Reza's search crashed three times and
            // each time the numbers went with it. Documents/memcheck.txt survives, and the Mac can read it.
            if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                let line = "\(Date()) about to load \(Judge.visionJudgeCandidateID): \(before)\n"
                let f = dir.appendingPathComponent("memcheck.txt")
                if let h = try? FileHandle(forWritingTo: f) { h.seekToEndOfFile(); h.write(Data(line.utf8)); try? h.close() }
                else { try? line.write(to: f, atomically: true, encoding: .utf8) }
            }
            // 4.6 GB, not 3.4: os_proc_available_memory reports HEADROOM, and loading 2.88 GB of weights spikes
            // well past the weights themselves (vision tower activations at 896 px, plus the KV cache). The 3.4
            // guard passed and the app was killed anyway (MEASURED 10-10 12:30). Falling back is always better
            // than being killed mid-search.
            // M36(B): the guard is deliberately LOW for this experiment - we need the load to be ATTEMPTED so
            // the per-stage log shows where it dies. Put it back to a safe value once (B) has its answer.
            if Judge.availableGB < 3.0 {
                qwenOutOfMemory = true
                planNote = "Not enough memory for the downloaded photo judge, so Apple's built-in model answered "
                         + "this search instead (\(before))."
            } else {
                do {
                    try await self.visionJudge.load { _ in }
                    if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
                       let h = try? FileHandle(forWritingTo: dir.appendingPathComponent("memcheck.txt")) {
                        h.seekToEndOfFile(); h.write(Data("  LOADED OK: \(Judge.memoryNote())\n".utf8)); try? h.close()
                    }
                }
                catch {
                    qwenOutOfMemory = true
                    planNote = "The downloaded photo judge could not start (\(before)); Apple's built-in model "
                             + "answered this search instead."
                }
            }
        } else if e == "vote" {
            // Two models at once by definition. Until the memory is measured, do not pretend it works.
            throw NSError(domain: "Judge", code: 2, userInfo: [NSLocalizedDescriptionKey:
                "The two-model vote needs more memory than this iPhone allows. Pick the Qwen3-VL photo judge or Apple's model."])
        }
        let engine = SearchEngine(index: index, embedder: embedder, judge: activeJudge)
        // a named pet / thing ("my dog Max"): its album becomes a subject search (FindPicsCore.subjectPlan)
        let asks = currentAsks
        let plan = subjectPlan(plan0, asks: asks)
        var personScores: [Int: PersonScored] = [:]
        var asked = false                       // one question per run (face picker or pet photos)
        let libItems = await index.libraryItems(order: Array(await index.entries.keys))
        let coverOnly = await index.coverFrameOnlyIds()     // videos not looked inside yet (FindPicsCore/LazyVideo.swift)
        let videoIds = Set(libItems.lazy.filter { $0.media == "video" }.map(\.id))
        for (k, album0) in plan.albums.enumerated() {
            // "only from Paris" -> GPS place filter; a place no photo has ("beach") -> a visual condition
            let album = placeOrLook(libItems, filterToPlace(libItems, album0))
            self.results.append(AlbumResult(name: album.name))
            if let ask = asks.first(where: { $0.albums.contains(k) }) {
                try await runSubjectAlbum(k, album, ask: ask, engine: engine, asked: &asked)
                continue
            }
            if facesUpdating, (album.person.map { !$0.isEmpty } ?? false) || !album.withPeople.isEmpty {
                self.results[k].note = faceReindexNote; self.results[k].done = true
                continue
            }
            if let person = album.person, !person.isEmpty {
                guard let refs = await people.refs(for: person, owner: owner) else {
                    if !asked { asked = true; await openFacePicker(for: person) }
                    self.results[k].note = "Who is \(person)? Confirm their face once."; self.results[k].done = true
                    continue
                }
                await ensureGroups()
                let others = await people.otherGroups(excluding: refs)
                let ps = try await engine.runPerson(album, refs: refs, others: others) { r in Task { @MainActor in
                    if k < self.results.count { self.results[k].judged = r.judged; self.results[k].inScope = r.inScope } } }
                personScores[k] = ps
                var r = AlbumResult(name: album.name); r.inScope = ps.ids.count; r.judged = ps.pYes.count; r.done = true
                r.note = uncheckedFacesNote(ps.unchecked) ?? ""    // small-copy faces: not in the album, counted here
                if album.judgeQuestion == nil { r.found = ps.ids } else {
                    let rel = withinPersonRank(ps.pYes)
                    // a fact about the photo ("a selfie", "at the beach"): a clear yes counts even outside the
                    // person's top half; how someone LOOKS stays relative to their own photos (engine.LOOK_WORDS)
                    let fact = album.judgeQuestion?.range(of: #"(?i)\b(look|looks|looking|appear|appears|appearing|seem|seems|seeming)\b"#,
                                                          options: .regularExpression) == nil
                    r.found = ps.ids.filter { (rel[$0] ?? 0) > 0.5 || (fact && (ps.pYes[$0] ?? 0) >= SearchEngine.accept) }
                        .sorted { (ps.pYes[$0] ?? 0) > (ps.pYes[$1] ?? 0) }
                }
                // videos only checked by their cover frame: the person may appear later in them (FindPicsCore)
                let pendingInScope = zip(libItems, scopeMask(libItems, album)).filter { $0.1 && coverOnly.contains($0.0.id) }.count
                if let n = coverFrameOnlyNote(media: album.media, pendingInScope: pendingInScope,
                                              foundVideos: r.found.filter { videoIds.contains($0) }.count) {
                    r.note += (r.note.isEmpty ? "" : " ") + n
                }
                self.results[k] = r
            } else {
                // the moment first ("the week I went to X"), then the album inside its window; dates/place belonged
                // to finding the moment
                let (scope, momentNote) = try await engine.momentScope(album)
                if let sc = scope, sc.isEmpty {
                    self.results[k].note = momentNote; self.results[k].done = true; continue
                }
                var inner = album
                if scope != nil { inner.dateFrom = nil; inner.dateTo = nil; inner.timePhrase = nil; inner.place = nil }
                let (withOK, unknown, withUnchecked) = await engine.withPeople(album, people: people, owner: owner)
                try await engine.run(inner, exhaustive: exhaustive, restrictTo: scope) { r in Task { @MainActor in
                    guard k < self.results.count else { return }
                    var r = r
                    if let ok = withOK {
                        let notChecked = r.found.filter { !ok.contains($0) && withUnchecked.contains($0) }.count
                        r.found = r.found.filter { ok.contains($0) }
                        if let n = uncheckedFacesNote(notChecked) { r.note += (r.note.isEmpty ? "" : " ") + n }
                    }
                    if !momentNote.isEmpty { r.note = momentNote + " " + r.note }
                    if !unknown.isEmpty { r.note += " Not known yet: \(unknown.joined(separator: ", ")) (pick their face once to include them)." }
                    self.results[k] = r } }
            }
        }
        // two opposite looks of the same person ("heavier" vs "fit"): split by the two groups the scores form
        let pk = personScores.keys.sorted()
        if pk.count == 2, plan.albums[pk[0]].person == plan.albums[pk[1]].person,
           let a = personScores[pk[0]], let b = personScores[pk[1]] {
            let entries = await index.entries
            let ids = Array(Set(a.pYes.keys).intersection(b.pYes.keys))
            let ev = events(ids.map { entries[$0]?.taken })
            let eventOf = Dictionary(uniqueKeysWithValues: zip(ids, ev.map { String($0) }))
            // two clear groups -> the split; otherwise the rank margin (engine.make_exclusive fallback)
            if let split = splitPair(pA: a.pYes, pB: b.pYes, eventOf: eventOf) ?? Optional(rankMarginPair(pA: a.pYes, pB: b.pYes)) {
                for (slot, k) in pk.enumerated() {
                    self.results[k].found = split.filter { $0.value == slot }.map { $0.key }
                        .sorted { (personScores[k]!.pYes[$0] ?? 0) > (personScores[k]!.pYes[$1] ?? 0) }
                    self.results[k].note = "Paired with '\(plan.albums[pk[1 - slot]].name)': each photo goes to the album its scores clearly belong to; \(split.values.filter { $0 == nil }.count) photo(s) are in neither."
                        + (uncheckedFacesNote(personScores[k]?.unchecked ?? 0).map { " " + $0 } ?? "")
                }
            }
        }
    }

    /// "my dog Max at the beach": Max's saved example photos -> the subject search, scoped by the rest of the album
    /// (moment window, dates, place, media; the condition judged on each match; people who must also be in it).
    /// Not saved yet -> ask "Show me Max: pick 1-3 clear photos of Max." and search again once picked.
    private func runSubjectAlbum(_ k: Int, _ album: Album, ask: SubjectAsk, engine: SearchEngine, asked: inout Bool) async throws {
        let saved = await subjects.saved
        var examples: [CIImage] = []
        if let si = resolveSubject(ask, saved: saved) {
            for id in saved[si].ids { if let im = await PhotoLibrary.ciImage(id, side: 1280) { examples.append(im) } }
        }
        guard !examples.isEmpty else {
            if !asked { asked = true; pendingSubject = ask; askSubject = true }
            results[k].note = ask.prompt; results[k].done = true
            return
        }
        let (scope, momentNote) = try await engine.momentScope(album)
        if let sc = scope, sc.isEmpty { results[k].note = momentNote; results[k].done = true; return }
        var inner = album
        if scope != nil { inner.dateFrom = nil; inner.dateTo = nil; inner.timePhrase = nil; inner.place = nil }
        let (withOK, unknown, withUnchecked) = await engine.withPeople(album, people: people, owner: owner)
        var r = try await engine.runSubject(examples: examples, name: ask.judgeName, kind: ask.kind, album: inner, restrictTo: scope) { r in
            Task { @MainActor in
                guard k < self.results.count else { return }
                var r = r
                if let ok = withOK { r.found = r.found.filter { ok.contains($0) } }
                self.results[k] = r
            }
        }
        if let ok = withOK {
            let notChecked = r.found.filter { !ok.contains($0) && withUnchecked.contains($0) }.count
            r.found = r.found.filter { ok.contains($0) }
            if let n = uncheckedFacesNote(notChecked) { r.note += (r.note.isEmpty ? "" : " ") + n }
        }
        if !momentNote.isEmpty { r.note = momentNote + " " + r.note }
        if !unknown.isEmpty { r.note += " Not known yet: \(unknown.joined(separator: ", ")) (pick their face once to include them)." }
        if k < results.count { results[k] = r }
    }

    /// The pet's / thing's photos were picked in the "Show me Max" sheet: remember them, run the same plan again.
    func provideSubjectPhotos(_ ids: [String]) {
        guard let ask = pendingSubject, !ids.isEmpty else { return }
        askSubject = false; pendingSubject = nil
        Task {
            await subjects.add(SavedSubject(name: ask.name, kind: ask.kind, words: ask.words, ids: ids))
            rerun()
        }
    }

    // MARK: who is X

    /// Face groups are O(faces^2) to build: rebuilt only when a person search or the picker needs them after a change.
    private func ensureGroups() async {
        if groupsStale { await people.refreshGroups(index: index); groupsStale = false }
    }

    func openFacePicker(for person: String) async {
        await ensureGroups()
        let s = await people.suggestion(for: person, owner: owner, index: index)
        faceGroupsShown = await people.groups
        faceSuggestion = s.suggested; faceOthers = s.others; askingFor = person
        faceNote = await people.reaskNote(person, owner: owner) ?? ""
        askWhichFace = true
    }

    private var askingKey: String { isOwnerWord(askingFor) ? owner : askingFor }

    func chooseFace(_ g: FaceGroup) {
        Task {
            await people.name(group: g, as: askingKey)
            askWhichFace = false
            rerun()
        }
    }

    /// "Add a photo of them": faces found in 1-3 picked photos (the one face they share, FindPicsCore.pickRefFaces).
    func addFacePhotos(_ ids: [String]) {
        guard let fe = faceEngine else { faceNote = "The face model is not loaded."; return }
        Task {
            var photos: [[(px: Double, emb: [Float])]] = []
            var found: [(id: String, face: DetectedFace)] = []         // to remember where each picked face is
            for id in ids {
                var ui: UIImage? = nil
                if case .full(let u) = await PhotoLibrary.read(id, side: 1280, purpose: .judge) { ui = u }
                guard let cg = ui?.cgImage, let fs = try? fe.faces(in: cg), !fs.isEmpty else { continue }
                photos.append(fs.map { (px: $0.px, emb: $0.embedding) })
                found += fs.map { (id: id, face: $0) }
            }
            let refs = pickRefFaces(photos: photos, floor: fe.profile.pickFloor)
            guard !refs.isEmpty else {
                faceNote = "No face found in those photos. Pick photos where the face is large and clear."
                return
            }
            let sources = refs.compactMap { r in found.first(where: { $0.face.embedding == r }).map {
                FaceSource(id: $0.id, t: nil, box: $0.face.box, imageW: $0.face.imageW, imageH: $0.face.imageH,
                           side: faceReadSide) } }        // read at 1280 (.judge, full size only)
            await people.name(refs: refs, sources: sources, as: askingKey)
            askWhichFace = false
            rerun()
        }
    }
}

struct RootView: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        switch model.stage {
        case .start:
            VStack(spacing: 12) {
                if let p = model.indexConversion {
                    ProgressView(value: p)
                    Text("Updating the photo index to a faster format (once): \(Int(p * 100))%")
                } else { ProgressView("Starting…") }
            }.padding().task { model.launch() }
        case .noAccess: Text("find pics needs access to your photos to search them. Settings > Privacy > Photos > find pics.").padding()
        case .askDownload:
            VStack(spacing: 16) {
                Text("find pics needs to download its on-phone AI once: about \(String(format: "%.1f", AppModel.downloadGB)) GB.")
                Text("Use Wi-Fi if you can. After this download, your photos are searched entirely on this phone and nothing is uploaded.")
                    .font(.footnote).foregroundStyle(.secondary)
                Text("Photos stored only in iCloud are downloaded to this phone to be read; nothing is uploaded.")
                    .font(.footnote).foregroundStyle(.secondary)
                Button("Download now") { model.downloadAccepted = true; model.stage = .start }.buttonStyle(.borderedProminent)
            }.multilineTextAlignment(.center).padding()
        case .downloading(let p): VStack { ProgressView(value: p); Text("One-time download of the on-phone AI (\(Int(p * 100))%). After this, nothing leaves your phone.") }.padding()
        case .indexing(let d, let t):
            VStack {
                ProgressView(value: Double(d), total: Double(max(t, 1)))
                Text("Reading your library once: \(d) of \(t). You can use your phone: this is fastest with find pics open, and it also continues while the phone charges if you just switch apps (iOS decides when; swiping find pics away stops it).")
                Text(model.progressETA ?? "Working… (estimating time left)").font(.caption).foregroundStyle(.secondary)
                // -timeIndex only (docs/MAC_INBOX.md M14): per-stage medians, read off a screenshot
                if IndexTiming.on { Text(IndexTiming.report()).font(.caption.monospaced()).padding(.top, 8) }
            }.padding()
        case .failed(let e):
            // What Reza actually saw at 02:50 and 04:09 on 10-09/10-10: forty lines of
            // "Error Domain=NSURLErrorDomain Code=-1005 ... _kCFStreamErrorCodeKey=53 ...". A person cannot act on
            // that, and the app sat there dead until it was relaunched from the Mac. Plain words and a button.
            VStack(spacing: 16) {
                Text(AppModel.friendlyFailure(e)).multilineTextAlignment(.center)
                Button("Try again") { model.stage = .start; model.launch() }.buttonStyle(.borderedProminent)
                Text(e).font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(4)
            }.padding()
        case .ready: SearchView()
        }
    }
}
