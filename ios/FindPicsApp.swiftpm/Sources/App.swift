// find pics: type what you are looking for, like the Photos search bar; everything runs on this phone.
import BackgroundTasks
import CoreImage
import CryptoKit
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
            // DID IOS EVER ACTUALLY RUN THIS? Open since early in the project and never answered: PLAN.md lists
            // "iOS background (charger) task never yet seen running on the device" as an open risk, and every
            // measurement we have was taken with the app open on screen. The whole night-pass architecture - the
            // thing the product rests on - assumes this fires while the phone is locked on a charger. One line in
            // the log settles it, and without it "it probably works" was the plan.
            Judge.logMem("BGTASK fired")
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
        do {
            try BGTaskScheduler.shared.submit(r)
            Judge.logMem("BGTASK scheduled")
        } catch {
            Judge.logMem("BGTASK NOT scheduled: \(String(describing: error))")
            Logger().error("find pics: background task not scheduled: \(String(describing: error))")
        }
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
    /// M57(4): the cluster's public food photos to label, in their order; empty = nothing to label.
    @Published var foodNotion: [URL] = []
    /// Finished searches, so their results survive closing the app (PastSearches.swift).
    let pastSearches = PastSearchStore()
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
    /// The groups the open picker shows were built at this PeopleStore.groupsGen (stale numbers are never saved).
    var faceGen = 0
    /// The picker was opened from the People screen: closing it does not re-run the last search.
    var pickerFromPeople = false
    /// Bumped when a saved person changes (the People screen reloads).
    @Published var peopleVersion = 0
    // "Show me Max" sheet (SearchView.swift SubjectAskSheet)
    @Published var askSubject = false
    @Published var pendingSubject: SubjectAsk?
    var owner = "me"
    var history: [String] = []
    var currentPlan: Plan?
    var currentAsks: [SubjectAsk] = []
    var searchTask: Task<Void, Never>?
    /// The search keeps looking after its first answer (RESULTS 44b): one continuation per album, run after every
    /// album's first answer is on screen; `lookingOn` while one runs (the Stop button).
    private var pendingLooks: [@MainActor () async throws -> Void] = []
    @Published var lookingOn = false
    /// Bumped by every new search: updates from a search being stopped must not land in the next one's results.
    private var searchGen = 0
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
        func album(_ name: String, _ ids: [String], judged: Int = 120, check: Int? = nil,
                   note: String) -> AlbumResult {
            var r = AlbumResult(name: name); r.found = ids; r.judged = judged; r.inScope = 148_428
            r.note = note; r.done = true
            r.checkCalls = check
            r.checkSeconds = check.map { Double($0) * SearchEngine.secondsPerJudgeCall }
            return r
        }
        lastQuery = "food photos"
        planNote = "Demo data (Simulator): not a real search."
        // THE TWO STATES THE NEW FLOW HAS, so the Simulator can show what he will actually read: the judge-free
        // answer with the priced button under it, and the same album after the careful check has run.
        results = [
            album("food photos (quick answer)", Array(ids.prefix(6)), judged: 0, check: 7_421,
                  note: "1,509 found by the quick check, which looked at 148,369 of the 148,428 photos in scope. "
                      + "No photo was opened by the AI, which is why it was quick - and why it can be wrong."),
            album("food photos (after the careful check)", Array(ids.dropFirst(6).prefix(6)), judged: 7_421,
                  note: "The AI opened 7,421 photos itself, the ones the quick check ranked highest, and kept 1,342. "
                      + "It stopped extending where matches ran out - under half of the last block it checked were "
                      + "matches. Against the quick answer: 94 added, 261 removed."),
        ]
        stage = .ready
    }

    private var startTask: Task<Void, Never>?
    /// From RootView's .task. start() runs in its own task: SwiftUI cancels a view's .task when the view goes away
    /// (here: as soon as the stage changes), which would cancel the model download and the indexing loop with it.
    /// Plan notes a PERSON can read. FindPicsCore writes its notes for us, in brackets - Reza was shown
    /// "[could not express 'Is the owner visible in this photo?' as a question about one photo; searching for
    /// what it looks like instead]" on 10-11 and could not have known what it meant, let alone that his search
    /// had quietly become a worse one. Keep the meaning, drop the developer voice; anything unrecognised is
    /// summarised rather than printed raw, because a bracket the user cannot act on is worse than silence.
    static func plainNotes(_ notes: String) -> String {
        var out = [String](), sawUnknown = false
        var rest = notes
        while let open = rest.firstIndex(of: "["), let close = rest[open...].firstIndex(of: "]") {
            let before = rest[rest.startIndex..<open].trimmingCharacters(in: .whitespaces)
            if !before.isEmpty { out.append(before) }
            let inner = String(rest[rest.index(after: open)..<close])
            let low = inner.lowercased()
            if low.hasPrefix("could not express") {
                out.append("I could not check that exactly, so I searched for what it looks like instead.")
            } else if low.hasPrefix("showing everything from that moment") {
                out.append("Showing everything from that moment.")
            } else if low.hasPrefix("searching for the thing itself") {
                out.append("Searching for the thing itself, not everything around it.")
            } else if low.hasPrefix("dropped the end moment") {
                out.append("I ignored the end of your request: no single photo can show it.")
            } else {
                sawUnknown = true
            }
            rest = String(rest[rest.index(after: close)...])
        }
        let tail = rest.trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty { out.append(tail) }
        if sawUnknown { out.append("Part of your request could not be checked exactly.") }
        return out.joined(separator: " ")
    }

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
        // A QUERY FROM A FILE, NOT FROM A LAUNCH ARGUMENT. Twice tonight a device measurement ran the wrong
        // search: devicectl persists launch arguments per bundle id, and when the app restarts - by itself, or
        // because iOS relaunched it - it comes back with the OLD argument. 21:16 ran "photos of a dog" while I
        // timed "selfies"; at 21:48 the app relaunched mid-run and started "photos of a dog" AGAIN, killing the
        // selfies run at 18 minutes. A file cannot do that: the Mac writes Documents/next_query.txt with
        // `devicectl copy to`, the app reads it, DELETES it, and runs it once. A relaunch finds no file and
        // indexes normally.
        if let q = AppModel.fileQuery() {
            // The same consumable file can also ask for a measurement, so no developer run ever needs a sticky
            // launch argument again.
            if q.lowercased() == "#selfiegeometry" { await runSelfieGeometry(); return }
            if q.lowercased() == "#selfievalidate" { await runSelfieValidate(); return }
            if q.lowercased() == "#eventcensus" { await runEventCensus(); return }
            if q.lowercased().hasPrefix("#factsfor ") {
                // "#factsFor 500 food photos": write descriptions for the N photos the library ranks highest for
                // that query. A strided sample measures the COST honestly but cannot show the BENEFIT - at 1 photo
                // in 187 the shortcut fires on nothing. This writes where a real search will actually look, which
                // is what a fully indexed library looks like for one query.
                let rest = String(q.dropFirst("#factsFor ".count))
                let n = rest.split(separator: " ").first.flatMap { Int($0) } ?? 500
                let query = rest.split(separator: " ").dropFirst().joined(separator: " ")
                await runFactsFor(query: query, count: n); return
            }
            if q.lowercased().hasPrefix("#factsindex") {
                await runFactsIndex(count: q.split(separator: " ").compactMap { Int($0) }.first ?? 300); return
            }
            if q.lowercased() == "#factsparity" { await runFactsParity(); return }
            if q.lowercased() == "#writerparity" { await runWriterParity(); return }
            if q.lowercased() == "#writerparityexact" { await runWriterParityExact(); return }
            if q.lowercased().hasPrefix("#factsonly") {
                await runFactsOnly(count: q.split(separator: " ").compactMap { Int($0) }.first ?? 300); return
            }
            if q.lowercased() == "#factscompare" { await runFactsCompare(); return }
            if q.lowercased() == "#factsshape" { await runFactsShape(); return }
            if q.lowercased().hasPrefix("#factsspeed") {
                let n = q.split(separator: " ").compactMap { Int($0) }.first ?? 20
                await runFactsSpeed(modelID: q.contains("4b") ? Judge.visionJudgeCandidateID
                                                              : "mlx-community/Qwen3-VL-2B-Instruct-4bit",
                                    count: n); return
            }
            if q.lowercased() == "#foodnotion" { await showFoodNotion(); return }
            if q.lowercased().hasPrefix("#selfieband") {
                // "#selfieBand 0.85 0.90" = the band between those two precisions; bare = 0.90 vs 0.95.
                let p = q.split(separator: " ").compactMap { Double($0) }
                await runSelfieBand(lower: p.first ?? 0.90, upper: p.count > 1 ? p[1] : 0.95); return
            }
            if q.lowercased().hasPrefix("#composition ") {
                await runComposition(String(q.dropFirst("#composition ".count))); return
            }
            await runDebugQuery(q); return
        }
        // `-runQuery` IS GONE ON PURPOSE. devicectl keeps launch arguments attached to the bundle id, and iOS
        // reuses them on relaunches we never asked for, so a query from HOURS ago kept winning: it cost the
        // selfies timing (10-11 21:16 and again at 21:48) and the flagship run twice tonight (22:06 and 22:15,
        // both planning "Is this a photo of a dog?" while I was timing something else). A one-shot file the app
        // consumes cannot do that. Anything that still needs an argument should be given a #task in next_query.txt.
        if ProcessInfo.processInfo.arguments.contains("-runQueries") { await runBatchQueries(); return }
        // NO LAUNCH ARGUMENT FOR THE PARITY RUNNERS. I added -factsParity / -writerParity / -factsOnly this
        // morning for exactly the right reason (devicectl cannot tap, so a typed #command needs Reza), and for
        // exactly the wrong mechanism: the comment above says devicectl PERSISTS arguments per bundle id and iOS
        // replays them on relaunches we never asked for, which already cost this project two timing runs. A
        // replayed -factsOnly 300 would spend 300 judge calls unasked. All three are already reachable through
        // the consumable next_query.txt above (scripts/run_device_task.sh writes it), which cannot run twice.
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
        // ALSO TO DISK. Reza, 10-11 22:18: "Is the videos thing done? I see it's 39,000 out of 39,000" - and I
        // could not answer it, because the two sweeps share one line and nothing outside the app ever knew their
        // counts. Growing index files told me work was happening but not which sweep or how far. One small file
        // the Mac can read with `devicectl copy from` ends that guessing for good.
        { p in Task { @MainActor in
            AppModel.shared.videoFrames = p
            if let f = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
                .appendingPathComponent("progress.json") {
                let d: [String: Any] = ["videos": p.videos, "sweep1Done": p.sweep1Done, "sweep2Done": p.sweep2Done,
                                        "unreadable": p.unreadable, "longVideos": p.longVideos,
                                        "at": ISO8601DateFormatter().string(from: Date())]
                if let j = try? JSONSerialization.data(withJSONObject: d, options: [.prettyPrinted]) {
                    try? j.write(to: f, options: [.atomic])
                }
            }
        } }
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

    /// M46: the face geometry of every photo whose camera is KNOWN, so the cluster can fit "this is a selfie" on
    /// face shape and apply it to the 170k photos whose EXIF we will never download.
    ///
    /// The 267 front-camera photos are labelled positives and the back-camera ones labelled negatives, on Reza's
    /// own library, so this needs no labelling and no judge. It also reads NO images: every face box is already in
    /// the index with the size of the image it was found in. Rows carry geometry only - no asset ids, no face
    /// embeddings, no dates - and land in Documents/selfie_geometry.csv for the Mac to pull into data/private/.
    func runSelfieGeometry() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        await loadStores()
        stage = .ready
        var rows = ["camera,isScreenshot,faces,biggestAreaFraction,biggestCentreOffset,biggestAspect,imageAspect"]
        var front = 0, back = 0
        for e in await index.entries.values {
            guard !e.isVideo, let cam = e.camera, cam == "front" || cam == "back" else { continue }
            cam == "front" ? (front += 1) : (back += 1)
            let faces = e.faces ?? []
            // The biggest face by area: the one a selfie is built around.
            let big = faces.max { a, b in
                ((a.box[2] - a.box[0]) * (a.box[3] - a.box[1])) < ((b.box[2] - b.box[0]) * (b.box[3] - b.box[1]))
            }
            var area = 0.0, offset = 0.0, aspect = 0.0, imgAspect = 0.0
            if let f = big, f.imageW > 0, f.imageH > 0 {
                let w = f.box[2] - f.box[0], h = f.box[3] - f.box[1]
                area = (w * h) / (f.imageW * f.imageH)
                // how far the face's centre sits from the image centre, in half-widths (0 = dead centre, 1 = edge)
                let cx = (f.box[0] + f.box[2]) / 2 / f.imageW - 0.5, cy = (f.box[1] + f.box[3]) / 2 / f.imageH - 0.5
                offset = (cx * cx + cy * cy).squareRoot() * 2
                aspect = h > 0 ? w / h : 0
                imgAspect = f.imageW / f.imageH
            }
            rows.append([cam, e.isScreenshot == true ? "1" : "0", String(faces.count),
                         String(format: "%.6f", area), String(format: "%.4f", offset),
                         String(format: "%.4f", aspect), String(format: "%.4f", imgAspect)].joined(separator: ","))
        }
        if let f = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("selfie_geometry.csv") {
            try? rows.joined(separator: "\n").write(to: f, atomically: true, encoding: .utf8)
        }
        Judge.logMem("SELFIEGEOM front=\(front) back=\(back) rows=\(rows.count - 1)")
        planNote = "M46: wrote \(rows.count - 1) geometry rows (\(front) front, \(back) back)."
        keepIndexingAfterDeveloperRun()
    }

    /// The night pass over `count` photos, NEWEST FIRST (M78(b)). Writes one line per photo and stores it with its
    /// 384 numbers. Logs the rate and thermal state so the real sustained cost is measured rather than extrapolated
    /// from a cool 20-photo run - which is exactly the mistake M77 made.
    func runFactsIndex(count: Int) async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        await loadStores()
        stage = .ready
        let entries = await index.entries
        // STRIDED ACROSS THE WHOLE LIBRARY, not the newest slice (cluster M84(3), and my own M78(b) lesson): the
        // 300 newest photos contained no dogs and one beach, so two of three wiring questions could measure
        // nothing. A stride spans every year he has photos from, which is also the mix the real pass will meet.
        let all = entries.filter { !$0.value.isVideo }
            .sorted { ($0.value.taken ?? 0) > ($1.value.taken ?? 0) }
            .map { $0.key }
        let step = max(1, all.count / max(1, count))
        let ids = stride(from: 0, to: all.count, by: step).prefix(count).map { all[$0] }
        await writeFacts(ids: ids, tag: "FACTSINDEX")
    }

    /// The shared writing loop: write a line per photo, store it with its numbers, log the rate, the read/model
    /// split and the thermal state every 25 photos. Used by both the strided cost pass and the targeted one.
    func writeFacts(ids: [String], tag: String) async {
        let entries = await index.entries
        let store = FactsStore()
        _ = await store.load()
        let writer = FactsWriter()
        do { try await writer.load() } catch { runFailed(tag, "writer did not load: \(error)"); return }
        Judge.logMem("\(tag) start n=\(ids.count) already=\(await store.count)")
        let t0 = Date()
        var done = 0, failed = 0
        var tokenCounts = [Int]()
        for id in ids {
            if Task.isCancelled { break }
            if await store.has(id) { continue }
            do {
                let e = entries[id]
                guard let r = try await writer.write(id: id, place: e?.place,
                                                     taken: e?.taken.map { Date(timeIntervalSince1970: $0) }) else {
                    failed += 1; continue
                }
                await store.put(id: id, line: r.line, vector: r.vector)
                done += 1
                tokenCounts.append(r.line.split(whereSeparator: { $0 == " " }).count)
            } catch { failed += 1 }
            if done % 25 == 0, done > 0 {
                await store.save()
                let s = Date().timeIntervalSince(t0)
                let rs = await writer.readSecs, ms = await writer.modelSecs, rn = max(1, await writer.reads)
                Judge.logMem(String(format: "%@ %d of %d, %.2f s each (read %.2f, model %.2f), thermal=%@",
                                    tag, done, ids.count, s / Double(done), rs / Double(rn), ms / Double(rn),
                                    ThermalLevel(ProcessInfo.processInfo.thermalState).name))
            }
        }
        await store.save()
        await writer.unload()
        let s = Date().timeIntervalSince(t0)
        let sorted = tokenCounts.sorted()
        let mean = tokenCounts.isEmpty ? 0 : Double(tokenCounts.reduce(0, +)) / Double(tokenCounts.count)
        let p90 = sorted.isEmpty ? 0 : sorted[min(sorted.count - 1, Int(0.9 * Double(sorted.count)))]
        let line = String(format: "%@ done %d written, %d unreadable, %.1f s total, %.2f s each, words mean %.1f p90 %d, thermal=%@",
                          tag, done, failed, s, done > 0 ? s / Double(done) : 0, mean, p90,
                          ThermalLevel(ProcessInfo.processInfo.thermalState).name)
        Judge.logMem(line); planNote = line
        keepIndexingAfterDeveloperRun()
    }

    /// Write descriptions for the photos a given query ranks highest - the ones a real search reaches first.
    func runFactsFor(query: String, count: Int) async {
        guard await PhotoLibrary.requestAccess() else { runFailed("FACTSFOR", "no photo access"); return }
        if embedder == nil { embedder = try? Embedder() }
        await loadStores()
        stage = .ready
        guard let emb = embedder, !query.isEmpty else { runFailed("FACTSFOR", "no embedder or empty query"); return }
        let entries = await index.entries
        let ids = entries.filter { !$0.value.isVideo }.map { $0.key }
        guard let look = try? emb.vector(of: query) else { runFailed("FACTSFOR", "could not embed the query"); return }
        let (units, unitItem, _) = await index.units(ids: ids)
        let cos = lookScores(units: units, unitItem: unitItem, nItems: ids.count, looks: [look], avoid: [])
        let pick = cos.indices.sorted { cos[$0] > cos[$1] }.prefix(count).map { ids[$0] }
        Judge.logMem("FACTSFOR \(query): top \(pick.count) of \(ids.count) by look score")
        await writeFacts(ids: pick, tag: "FACTSFOR")
    }

    /// IS THE WRITER WRITING FIELD LINES, OR JUST CAPTIONS? The cluster found (11-05) that the AUC wiring check
    /// cannot tell the difference - base-model captions score about as well as trained field lines - so a broken
    /// or wrong writer would pass it silently. This checks the SHAPE of what was written instead: the trained
    /// student emits "scene: ...; objects: ...; people: ..." and the base model emits a sentence.
    /// Counts only - field names and lengths, never the content, which describes his photos.
    func runFactsShape() async {
        let store = FactsStore()
        guard await store.load(), await store.count > 0 else { runFailed("FACTSSHAPE", "no facts stored"); return }
        let rows = await store.rows
        var fieldCounts = [Int](), withColon = 0, names = [String: Int](), words = [Int]()
        for (_, r) in rows {
            let parts = r.line.split(separator: ";")
            var n = 0
            for part in parts {
                let t = part.trimmingCharacters(in: .whitespaces)
                guard let c = t.firstIndex(of: ":") else { continue }
                let name = String(t[t.startIndex..<c]).lowercased()
                // a field name is a word or two, not a sentence that happens to contain a colon
                guard name.split(separator: " ").count <= 3, !name.isEmpty else { continue }
                n += 1
                names[name, default: 0] += 1
            }
            if n > 0 { withColon += 1 }
            fieldCounts.append(n)
            words.append(r.line.split(whereSeparator: { $0 == " " }).count)
        }
        let n = max(1, rows.count)
        let meanFields = Double(fieldCounts.reduce(0, +)) / Double(n)
        let meanWords = Double(words.reduce(0, +)) / Double(n)
        let top = names.sorted { $0.value > $1.value }.prefix(8)
            .map { "\($0.key)=\($0.value)" }.joined(separator: " ")
        let line = String(format: "FACTSSHAPE lines=%d withFields=%d (%.0f%%) meanFields=%.1f meanWords=%.1f | %@",
                          rows.count, withColon, 100 * Double(withColon) / Double(n), meanFields, meanWords, top)
        Judge.logMem(line); planNote = line
    }

    /// #writerparityexact (M101(1)): separate "the pixels" from "the runtime" in the .815 writer gap.
    /// MAC-DEVICE2 found the phone's word F1 against the cluster's lines at .815, below the minimum (.879) of their
    /// MLX-vs-vLLM calibration. The suspect was the image path: they resized once with Pillow BICUBIC and read raw
    /// values; the phone resized a Commons thumbnail with CoreImage AND colour-managed 16 of 20 files that carry
    /// Display P3 / Apple Wide Color profiles. So the cluster published PARITY-EXACT PNGs - one Pillow resize
    /// straight to the processor's own size, no ICC, no metadata - which this Mac rebuilt and verified pixel-for-pixel
    /// (pixels 20/20) before pushing them to Documents/parity_exact/.
    /// Those sizes make the phone's own 448 best-fit a NO-OP (max side is already 448), so the shipped generate()
    /// path resamples nothing. What is left to control is colour: the PNGs carry no profile, so they are read with
    /// colour management OFF, which is what Pillow effectively did.
    /// THE CLUSTER'S OWN CALIBRATION FOR THIS: their old input route vs these exact PNGs moves THEIR lines to mean
    /// word F1 .890 (min .681) - so pixels alone move lines about as much as our whole gap. Reading:
    ///   device >= ~.88  -> it was the pixels, and the runtime is fine
    ///   device still ~.8 -> something in the runtime differs and the pixels were a red herring
    /// "Ze Seewjinu" (544x128) is EXCLUDED by M101: its 448 fit falls under the processor's min_pixels, so the
    /// phone would rescale it and the input would not be parity-exact.
    func runWriterParityExact() async {
        struct Photo: Decodable {
            let key: String; let W: Int; let H: Int
            let png_sha256: String; let line_trained_k48: String
        }
        struct File: Decodable { let photos: [Photo] }
        guard let url = Bundle.module.url(forResource: "SelfCheck/writer_parity_exact", withExtension: "json"),
              let data = try? Data(contentsOf: url) else {
            runFailed("WPEXACT", "SelfCheck/writer_parity_exact.json is not in the bundle"); return
        }
        let F: File
        do { F = try JSONDecoder().decode(File.self, from: data) }
        catch { runFailed("WPEXACT", "writer_parity_exact.json did not decode: \(error)"); return }
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            runFailed("WPEXACT", "no documents directory"); return
        }
        let dir = docs.appendingPathComponent("parity_exact")
        guard FileManager.default.fileExists(atPath: dir.path) else {
            runFailed("WPEXACT", "Documents/parity_exact is not on the phone - run scripts/push_parity_exact.sh"); return
        }
        guard let mData = try? Data(contentsOf: dir.appendingPathComponent("manifest.json")),
              let manifest = try? JSONDecoder().decode([String: String].self, from: mData) else {
            runFailed("WPEXACT", "Documents/parity_exact/manifest.json missing - re-run scripts/push_parity_exact.sh"); return
        }
        func words(_ s: String) -> [String: Int] {
            var out = [String: Int](), cur = ""
            for ch in s.lowercased() {
                if ch.isASCII && (ch.isLetter || ch.isNumber) { cur.append(ch) }
                else if !cur.isEmpty { out[cur, default: 0] += 1; cur = "" }
            }
            if !cur.isEmpty { out[cur, default: 0] += 1 }
            return out
        }
        func wordF1(_ a: String, _ b: String) -> Double {
            let wa = words(a), wb = words(b)
            let na = wa.values.reduce(0, +), nb = wb.values.reduce(0, +)
            guard na > 0, nb > 0 else { return 0 }
            let overlap = wa.reduce(0) { $0 + min($1.value, wb[$1.key] ?? 0) }
            return 2 * Double(overlap) / Double(na + nb)
        }
        let writer = FactsWriter()
        do { try await writer.load() } catch { runFailed("WPEXACT", "writer did not load: \(error)"); return }
        var f1s = [Double](), exact = 0, missing = 0, shaFail = 0, decodeFail = 0, skipped = 0
        for (i, p) in F.photos.enumerated() {
            // M101: the 544x128 panorama is not parity-exact on a phone, so it is not scored.
            if max(p.W, p.H) > 448 { skipped += 1; continue }
            let f = dir.appendingPathComponent(String(format: "%02d.png", i))
            guard let bytes = try? Data(contentsOf: f) else { missing += 1; continue }
            // AGAINST THE MAC'S MANIFEST, NOT THE CLUSTER'S png_sha256. The recipe says it plainly and I gated on
            // the wrong hash anyway: PNG FILE bytes differ between zlib builds while the PIXELS are identical
            // (the Mac's rebuild was pixels 20/20, png files 0/20), so comparing file hashes failed all 19.
            // The pixel equivalence is established on the Mac; what the phone can usefully check is that the file
            // it is reading is the file that was pushed.
            let got = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            guard let want = manifest[String(format: "%02d.png", i)] else { shaFail += 1; continue }
            guard got == want else { shaFail += 1; continue }
            // COLOUR MANAGEMENT OFF: these PNGs carry no profile and the cluster read raw values. With it on,
            // CoreImage would convert into its working space and the pixels the vision tower sees would not be
            // the pixels the lines were written from - which is the whole thing being tested.
            guard let img = CIImage(contentsOf: f, options: [.colorSpace: NSNull()]) else { decodeFail += 1; continue }
            guard let line = try? await writer.parityLine(image: img) else { decodeFail += 1; continue }
            let s = wordF1(line, p.line_trained_k48)
            if line == p.line_trained_k48 { exact += 1 }
            f1s.append(s)
        }
        let modelS = await writer.modelSecs
        await writer.unload()
        let mean = f1s.isEmpty ? 0 : f1s.reduce(0, +) / Double(f1s.count)
        let verdict = f1s.count < 15 ? "TOO FEW PHOTOS TO JUDGE"
                    : mean >= 0.88 ? "PIXELS: the exact inputs close the gap"
                    : mean >= 0.84 ? "PARTLY THE PIXELS - between .815 and the cluster's .890"
                                   : "NOT THE PIXELS: the runtime differs"
        Judge.logMem(String(format: "WPEXACT n=%d of %d (skipped %d over 448) meanWordF1=%.3f exact=%d min=%.3f "
                            + "max=%.3f | was .815 on the old route, cluster .890 on these | missing=%d shaMismatch=%d "
                            + "decodeFail=%d | model %.1f s (%.2f s/photo) | %@",
                            f1s.count, F.photos.count, skipped, mean, exact, f1s.min() ?? 0, f1s.max() ?? 0,
                            missing, shaFail, decodeFail, modelS,
                            f1s.isEmpty ? 0 : modelS / Double(f1s.count), verdict))
        planNote = "#writerparityexact done - see the log."
        keepIndexingAfterDeveloperRun()
    }

    /// #writerparity (M91): is the writer ON THIS PHONE the same writer the cluster measured, or a different one?
    /// 20 PUBLIC Wikimedia Commons iPhone photos (not in any training set), each with the cluster's MLX line at
    /// K = 48 and the sha256 of the exact file it was given. The phone downloads the same bytes, checks the hash,
    /// and runs the SHIPPED writing path (FactsWriter.parityLine: same prompt, K = 48, greedy, rep 1.05 / 4096,
    /// repeat-stop). STATISTIC: per photo, word F1 against line_trained_k48 (words = lowercase [a-z0-9] runs,
    /// multiset overlap), then the mean over the 20. PRE-REGISTERED BAR: pass iff mean >= .70 - the cluster's
    /// calibration has a correct chain at .943 (95% .907-.973) and ANOTHER photo's line at .231, so this separates
    /// "same model, different runtime" from "wrong model" with a wide margin.
    /// The bytes stay in memory and are never written to the photo library or to disk (M91 says never into Photos;
    /// not touching the filesystem at all is strictly less than that). Needs the network for the downloads.
    /// Numbers only: the lines themselves are public, but there is no reason to log any of them.
    func runWriterParity() async {
        struct Photo: Decodable { let url: String; let sha256: String; let line_trained_k48: String }
        struct File: Decodable { let photos: [Photo] }
        guard let url = Bundle.module.url(forResource: "SelfCheck/writer_parity", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let F = try? JSONDecoder().decode(File.self, from: data) else {
            runFailed("WRITERPARITY", "SelfCheck/writer_parity.json missing or unreadable"); return
        }
        func words(_ s: String) -> [String: Int] {
            var out = [String: Int](), cur = ""
            for ch in s.lowercased() {
                if ch.isASCII && (ch.isLetter || ch.isNumber) { cur.append(ch) }
                else if !cur.isEmpty { out[cur, default: 0] += 1; cur = "" }
            }
            if !cur.isEmpty { out[cur, default: 0] += 1 }
            return out
        }
        func wordF1(_ a: String, _ b: String) -> Double {
            let wa = words(a), wb = words(b)
            let na = wa.values.reduce(0, +), nb = wb.values.reduce(0, +)
            guard na > 0, nb > 0 else { return 0 }
            let overlap = wa.reduce(0) { $0 + min($1.value, wb[$1.key] ?? 0) }
            return 2 * Double(overlap) / Double(na + nb)
        }
        let writer = FactsWriter()
        do { try await writer.load() } catch {
            runFailed("WRITERPARITY", "writer did not load: \(error)"); return
        }
        var f1s = [Double](), exact = 0, dlFail = 0, shaFail = 0, genFail = 0
        for p in F.photos {
            guard let u = URL(string: p.url) else { dlFail += 1; continue }
            var bytes: Data? = nil
            do { bytes = try await URLSession.shared.data(from: u).0 } catch { dlFail += 1; continue }
            guard let d = bytes else { dlFail += 1; continue }
            // the hash first: a Commons thumbnail that changed is a different photo, and a parity number computed
            // on different pixels is worse than no number at all
            let got = SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined()
            guard got == p.sha256 else { shaFail += 1; continue }
            guard let img = CIImage(data: d) else { genFail += 1; continue }
            guard let line = try? await writer.parityLine(image: img) else { genFail += 1; continue }
            let f = wordF1(line, p.line_trained_k48)
            if line == p.line_trained_k48 { exact += 1 }
            f1s.append(f)
            if f1s.count % 5 == 0 {
                Judge.logMem(String(format: "WRITERPARITY %d of %d done, mean F1 so far %.3f",
                                    f1s.count, F.photos.count, f1s.reduce(0, +) / Double(f1s.count)))
            }
        }
        let modelS = await writer.modelSecs
        await writer.unload()
        let mean = f1s.isEmpty ? 0 : f1s.reduce(0, +) / Double(f1s.count)
        Judge.logMem(String(format: "WRITERPARITY n=%d of %d meanWordF1=%.3f (bar .70) exact=%d min=%.3f max=%.3f "
                            + "| downloadFail=%d shaMismatch=%d generateFail=%d | model %.1f s (%.2f s/photo) | %@",
                            f1s.count, F.photos.count, mean, exact, f1s.min() ?? 0, f1s.max() ?? 0,
                            dlFail, shaFail, genFail, modelS,
                            f1s.isEmpty ? 0 : modelS / Double(f1s.count),
                            f1s.count >= 15 ? (mean >= 0.70 ? "PASS" : "FAIL") : "TOO FEW PHOTOS TO JUDGE"))
        planNote = "#writerparity done - see the log."
        keepIndexingAfterDeveloperRun()
    }

    /// #factsonly [n] (M89): does the facts CHAIN work, or is the head carrying it with the image vector?
    /// The cluster's ablations showed the full P barely moves when the lines are shuffled between photos, because
    /// the head sees the image vector too - so "the full P separates well" is not evidence that the WRITING and the
    /// ENCODING are right. Zero the image vector and the ranking inside one question depends only on the facts
    /// vector: that facts-only AUC is ~.93 for a correct chain and ~.5 for a broken one (wrong photo's vector,
    /// garbage text, constant f). PRE-REGISTERED BAR: pass iff >= .70 (cluster simulation at 28 yes / 272 no:
    /// correct median .926, 95% .863-.974; broken .507, .411-.610; 0 of 2,000 draws on the wrong side of .70).
    /// The judge's answers are CACHED in Documents/judge_p_cache.json, so the first run costs n calls and every
    /// re-run costs none - which is what makes M89's "0 new judge calls" true the second time.
    /// Numbers only: no ids, no written lines, nothing about what the photos contain.
    func runFactsOnly(count: Int = 300) async {
        guard await PhotoLibrary.requestAccess() else { runFailed("FACTSONLY", "no photo access"); stage = .noAccess; return }
        if embedder == nil { embedder = try? Embedder() }
        await loadStores()
        stage = .ready
        guard let emb = embedder else { runFailed("FACTSONLY", "no embedder"); return }
        guard let headURL = Bundle.module.url(forResource: "towervec_head", withExtension: "fphead",
                                              subdirectory: "Models/facts"),
              let head = try? ScorerHead(contentsOf: headURL) else {
            runFailed("FACTSONLY", "towervec_head.fphead missing"); return
        }
        let store = FactsStore()
        _ = await store.load()
        var have = await store.rows
        let writer = FactsWriter()
        do { try await writer.loadEncoderOnly() } catch {
            runFailed("FACTSONLY", "encoder did not load: \(error)"); return
        }
        // M89's question, the one whose judge answers are worth caching: it is the question the cluster's own
        // facts-only simulation was calibrated on.
        let question = "Is this a photo of food?"
        guard let qv = try? await writer.questionVector(scorerQuestion(question)),
              let tv = try? emb.vector(of: scorerQuestion(question)) else {
            await writer.unload(); runFailed("FACTSONLY", "could not embed the question"); return
        }
        await writer.unload()
        let folded = head.fold(q: qv, t: tv)

        // THE 300 NEWEST, which is M89's pre-registered slice - NOT the top of the look ranking.
        // I picked by look score first ("the photos a real search reaches first") and it destroyed the statistic:
        // 295 of 298 came back judge-yes, so an AUC meant to separate yes from no had three negatives to work with
        // and returned .714 on nothing. The calibration this bar comes from (300 photos, 28 judge-yes, correct
        // chain .926 vs broken .507) assumes a broad slice with both classes in it. A ranked slice cannot test a
        // ranker. Any picked photo without a trained K = 48 line gets one written below.
        let entries = await index.entries
        let ids = entries.filter { !$0.value.isVideo }.map { $0.key }
        let picked = entries.filter { !$0.value.isVideo }
            .sorted { ($0.value.taken ?? 0) > ($1.value.taken ?? 0) }
            .prefix(count).map { $0.key }
        let (units, unitItem, _) = await index.units(ids: ids)
        let dim = units.dim
        let want = Set(picked)
        var imgVec = [String: [Float]]()
        units.withRows(0, units.count) { buf in
            for u in 0..<units.count {
                let id = ids[unitItem[u]]
                guard want.contains(id), imgVec[id] == nil else { continue }
                imgVec[id] = Array(UnsafeBufferPointer(rebasing: buf[(u * dim)..<((u + 1) * dim)]))
            }
        }

        // PHASE 1, the lines: the shared writing loop, which skips ids that already have one and reports the token
        // counts - so this run also gives the K = 48 re-time (M87(4) / M92(3)) on these photos. The writer (1.7 GB)
        // and the judge (2.9 GB) are NEVER resident together: 11-05 measured 1.07 GB free at 4.60 GB with both,
        // which is the territory the app was killed in. writeFacts unloads the writer before it returns.
        let missing = picked.filter { have[$0] == nil }
        if !missing.isEmpty {
            await writeFacts(ids: missing, tag: "FACTSONLYLINES")
            // A FRESH STORE, because writeFacts writes through its OWN FactsStore and FactsStore.load() returns
            // early once `loaded` is true. Re-loading this instance was a no-op, so after writing 298 lines this
            // read still held the snapshot from before them: "lines ready: 2 of 300", 298 counted unreadable, and
            // the statistic computed on two photos.
            let fresh = FactsStore()
            _ = await fresh.load()
            have = await fresh.rows
        }
        let withLine = picked.filter { have[$0] != nil }.count
        Judge.logMem("FACTSONLY lines ready: \(withLine) of \(picked.count) picked")
        if withLine < picked.count / 2 {
            runFailed("FACTSONLY", "only \(withLine) of \(picked.count) picked photos have a line - not computing a statistic on that")
            return
        }

        let cacheURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("judge_p_cache.json")
        var cache = [String: Double]()
        if let u = cacheURL, let d = try? Data(contentsOf: u),
           let c = try? JSONDecoder().decode([String: Double].self, from: d) { cache = c }
        let before = cache.count
        var fullP = [Float](), onlyP = [Float](), yes = [Bool](), fresh = 0, reused = 0
        // THREE DIFFERENT FAILURES, THREE COUNTERS. One "unreadable=298" hid the fact that the lines were there and
        // my stale read could not see them; it read like a photo-library problem.
        var noLine = 0, noVector = 0, noJudge = 0
        let zeros = [Float](repeating: 0, count: dim)
        for id in picked {
            guard let row = have[id] else { noLine += 1; continue }
            guard let v = imgVec[id] else { noVector += 1; continue }
            let key = "\(question)|\(id)"
            var pj = cache[key]
            if pj == nil {
                if !(await visionJudge.isLoaded) {
                    await judge.unload()
                    do { try await visionJudge.load { _ in } } catch {
                        runFailed("FACTSONLY", "judge did not load: \(error)"); return
                    }
                }
                guard let img = await PhotoLibrary.ciImage(id, side: 896),
                      let p = try? await visionJudge.pYes(img, question: question) else { noJudge += 1; continue }
                pj = p; cache[key] = p; fresh += 1
                if fresh % 25 == 0, let u = cacheURL, let d = try? JSONEncoder().encode(cache) { try? d.write(to: u) }
            } else { reused += 1 }
            guard let p = pj else { continue }
            fullP.append(folded.p(f: row.vector, v: v))
            onlyP.append(folded.p(f: row.vector, v: zeros))
            yes.append(p >= SearchEngine.accept)
        }
        if let u = cacheURL, let d = try? JSONEncoder().encode(cache) { try? d.write(to: u) }
        let aucFull = rocAUC(scores: fullP, positive: yes)
        let aucOnly = rocAUC(scores: onlyP, positive: yes)
        let judgeYes = yes.filter { $0 }.count
        let pass = (aucOnly ?? 0) >= 0.70
        Judge.logMem(String(format: "FACTSONLY n=%d judgeYes=%d aucFactsOnly=%@ aucFull=%@ (bar .70) "
                            + "newJudgeCalls=%d cached=%d noLine=%d noVector=%d noJudge=%d cacheSize=%d(+%d) | %@",
                            yes.count, judgeYes,
                            aucOnly.map { String(format: "%.3f", $0) } ?? "n/a",
                            aucFull.map { String(format: "%.3f", $0) } ?? "n/a",
                            fresh, reused, noLine, noVector, noJudge, cache.count, cache.count - before,
                            judgeYes >= 5 ? (pass ? "PASS" : "FAIL") : "NOT ENOUGH JUDGE-YES TO JUDGE"))
        planNote = "#factsonly done - see the log."
        keepIndexingAfterDeveloperRun()
    }

    /// #factsparity (M85): prove the FACTS TEXT ENCODER on this phone, with no photos and no judge.
    /// The cluster's golden fixture carries, for 28 facts lines and 9 questions made from PUBLIC dev photos, the
    /// word-piece ids, the fp32 384-d vector, and the P of 26 (question, line) pairs through the towervec head.
    /// Three things must hold here, and each one catches a different silent wrongness:
    ///   ids       - the BUNDLED vocab tokenizes exactly as the cluster's did (a vocab mismatch changes every
    ///               vector while still producing plausible numbers);
    ///   cosine    - >= .9999 against fp32 for EVERY text. Omitting tokenTypeIds dropped this to .72 in
    ///               RESULTS 68, which looked like a bad model rather than a bad call;
    ///   P         - within 4e-3 for EVERY pair. The bar is 2x what fp16 weights alone move P (max 1.60e-3 over
    ///               8,091 cluster pairs), so the requested 1e-3 would have failed a correct encoder (M85).
    /// It also checks the sha256 of the two bundled files the fixture was made with, so "the encoder is fine" can
    /// never be said of a different encoder. ~37 encodes, seconds. Numbers only: the fixture's texts are public.
    func runFactsParity() async {
        // v / t are NULL for three entries (two UTC month-boundary facts and the question over the 64-word-piece
        // cap): those are ids-and-cosine cases with no head input, and no pair references them. Decoding them as
        // non-optional rejected the whole file - and the error said "missing or unreadable", which sent me looking
        // for a bundling problem instead of a decoding one.
        struct Fact: Decodable { let name: String; let text: String; let ids: [Int]; let fp32: [Float]; let v: [Float]? }
        struct Question: Decodable { let name: String; let question: String; let ids: [Int]; let fp32: [Float]; let t: [Float]? }
        struct Pair: Decodable { let fact: Int; let question: Int; let p: Float }
        struct Golden: Decodable {
            let cos_bar: Float; let p_bar: Float
            let encoder_safetensors_sha256: String; let head_sha256: String
            let facts: [Fact]; let questions: [Question]; let pairs: [Pair]
        }
        guard let url = Bundle.module.url(forResource: "SelfCheck/facts_encoder_golden", withExtension: "json"),
              let data = try? Data(contentsOf: url) else {
            runFailed("FACTSPARITY", "SelfCheck/facts_encoder_golden.json is not in the bundle"); return
        }
        let G: Golden
        do { G = try JSONDecoder().decode(Golden.self, from: data) }
        catch { runFailed("FACTSPARITY", "golden json did not decode: \(error)"); return }
        guard let headURL = Bundle.module.url(forResource: "towervec_head", withExtension: "fphead",
                                              subdirectory: "Models/facts"),
              let head = try? ScorerHead(contentsOf: headURL) else {
            runFailed("FACTSPARITY", "towervec_head.fphead missing"); return
        }
        // The files, before the maths: a parity run against weights the fixture was not made with proves nothing.
        func sha256(_ u: URL) -> String {
            guard let d = try? Data(contentsOf: u, options: .mappedIfSafe) else { return "unreadable" }
            return SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined()
        }
        let encDir = Bundle.module.url(forResource: "towervec_encoder_fp16", withExtension: nil,
                                       subdirectory: "Models/facts")
        let encSHA = encDir.map { sha256($0.appendingPathComponent("model.safetensors")) } ?? "missing"
        let headSHA = sha256(headURL)
        let encOK = encSHA == G.encoder_safetensors_sha256, headOK = headSHA == G.head_sha256
        Judge.logMem("FACTSPARITY files encoder=\(encOK ? "OK" : "MISMATCH") head=\(headOK ? "OK" : "MISMATCH")")

        let writer = FactsWriter()
        do { try await writer.loadEncoderOnly() } catch {
            runFailed("FACTSPARITY", "encoder did not load: \(error)"); return
        }
        func cosine(_ a: [Float], _ b: [Float]) -> Float {
            guard a.count == b.count, !a.isEmpty else { return -2 }
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            for i in a.indices { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
            return (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : -2
        }
        var factV = [[Float]](), questV = [[Float]]()
        var idsBad = 0, idsBadQ = 0, minCosF: Float = 2, minCosQ: Float = 2
        var worstF = "", worstQ = "", firstIdsBad = ""
        for f in G.facts {
            guard let got = try? await writer.parityVector(f.text, question: false) else {
                runFailed("FACTSPARITY", "encoder returned nothing for a facts text"); await writer.unload(); return
            }
            if got.ids != f.ids { idsBad += 1; if firstIdsBad.isEmpty { firstIdsBad = f.name } }
            let c = cosine(got.v, f.fp32)
            if c < minCosF { minCosF = c; worstF = f.name }
            factV.append(got.v)
        }
        for q in G.questions {
            guard let got = try? await writer.parityVector(q.question, question: true) else {
                runFailed("FACTSPARITY", "encoder returned nothing for a question"); await writer.unload(); return
            }
            if got.ids != q.ids { idsBadQ += 1; if firstIdsBad.isEmpty { firstIdsBad = q.name } }
            let c = cosine(got.v, q.fp32)
            if c < minCosQ { minCosQ = c; worstQ = q.name }
            questV.append(got.v)
        }
        await writer.unload()
        // P: the device's OWN vectors through the folded head, against the cluster's P for that pair.
        var maxPDiff: Float = 0, pairsOver = 0, worstPair = ""
        var pairsSkipped = 0
        for pr in G.pairs {
            guard pr.fact < factV.count, pr.question < questV.count,
                  let t = G.questions[pr.question].t, let v = G.facts[pr.fact].v else { pairsSkipped += 1; continue }
            let folded = head.fold(q: questV[pr.question], t: t)
            let p = folded.p(f: factV[pr.fact], v: v)
            let d = abs(p - pr.p)
            if d > maxPDiff { maxPDiff = d; worstPair = "\(pr.question)/\(pr.fact)" }
            if d > G.p_bar { pairsOver += 1 }
        }
        let pass = idsBad == 0 && idsBadQ == 0 && minCosF >= G.cos_bar && minCosQ >= G.cos_bar
                   && pairsOver == 0 && encOK && headOK
        Judge.logMem(String(format: "FACTSPARITY ids bad %d/%d facts %d/%d questions | minCos facts %.6f "
                            + "questions %.6f (bar %.4f) | maxPdiff %.2e over %d pairs, %d over bar %.3f | %@",
                            idsBad, G.facts.count, idsBadQ, G.questions.count, minCosF, minCosQ, G.cos_bar,
                            maxPDiff, G.pairs.count - pairsSkipped, pairsOver, G.p_bar, pass ? "PASS" : "FAIL"))
        if pairsSkipped > 0 { Judge.logMem("FACTSPARITY \(pairsSkipped) pair(s) had no head input and were not scored") }
        if !pass {
            Judge.logMem("FACTSPARITY worst: facts=\(worstF) question=\(worstQ) pair=\(worstPair) "
                         + "firstIdsMismatch=\(firstIdsBad.isEmpty ? "none" : firstIdsBad)")
        }
        planNote = pass ? "#factsparity PASS - see the log for the numbers."
                        : "#factsparity FAIL - see the log."
    }

    /// A WIRING CHECK, NOT THE SHIP DECISION (M78(b), reframed by the cluster in M82 - my first version of this
    /// comment was wrong and said the opposite).
    /// The facts scorer alone is EXPECTED to disagree with the judge: it finds ~79% of the judge's matches, and
    /// the cascade's ~65 judge calls per search exist to close exactly that gap. So ~80% agreement is the
    /// PASSING outcome, not a failure, and "if they disagree nothing downstream is worth building" was wrong.
    /// What the device can check is that the pieces are wired right. The statistic for that is the threshold-free
    /// AUC - how well the facts score separates judge-yes photos from judge-no ones - because a wiring mistake
    /// (pooling, tokenTypeIds, prompt) lands near .5 whatever cut is used. The cluster computes the same AUC on
    /// public iPhone photos with the same three questions, judge, head and writer, and PRE-REGISTERED the bar
    /// before any device number: pass iff device AUC >= cluster AUC - .10 for each question with >= 5 judge-yes
    /// photos. With ~300 photos the AUC moves +-.1 by chance, so this is a gross-error check only.
    /// Numbers only: no photo ids, no written text, nothing about what his photos contain.
    func runFactsCompare() async {
        // EVERY EXIT LOGS. 11-05: this returned silently because `embedder` was nil (I never built it here, unlike
        // every other runner), and the failure went to planNote - which the device log never sees. Nine minutes
        // looked exactly like "still starting". A runner whose failures are invisible is worse than no runner.
        func fail(_ why: String) { Judge.logMem("FACTSCOMPARE abort: \(why)"); planNote = "M78: \(why)" }
        guard await PhotoLibrary.requestAccess() else { fail("no photo access"); stage = .noAccess; return }
        if embedder == nil { embedder = try? Embedder() }
        if faceEngine == nil { faceEngine = try? FaceEngine() }
        await loadStores()
        stage = .ready
        Judge.logMem("FACTSCOMPARE start")
        let store = FactsStore()
        _ = await store.load()
        let have = await store.rows
        guard !have.isEmpty else { fail("no facts yet - run #factsIndex first"); return }
        guard let headURL = Bundle.module.url(forResource: "towervec_head", withExtension: "fphead",
                                              subdirectory: "Models/facts"),
              let head = try? ScorerHead(data: Data(contentsOf: headURL)) else {
            fail("towervec_head.fphead missing"); return
        }
        guard let emb = embedder else { fail("no embedder"); return }
        let writer = FactsWriter()
        // ENCODER ONLY: the comparison needs the 33M text encoder, not the 1.7 GB writer. Loading both alongside
        // the judge left 1.07 GB free at 4.60 GB resident (measured 11-05) - the territory the app was killed in.
        do { try await writer.loadEncoderOnly() } catch { fail("encoder did not load: \(error)"); return }
        // the PHOTO judge is what this compares against, so load that one (not the planner)
        if !(await visionJudge.isLoaded) {
            await judge.unload()
            do { try await visionJudge.load { _ in } } catch { fail("judge did not load: \(error)"); return }
        }
        Judge.logMem("FACTSCOMPARE models ready, photos=\(have.count)")
        let questions = ["Is there a real dog anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?",
                         "Is this a photo of food?",
                         "Is a beach clearly visible in this photo?"]
        // SAMPLE WHERE THE POSITIVES ARE. My first run took whatever had facts and asked three questions of it;
        // the newest 300 photos held no dogs and one beach, so two of three questions had nothing to separate and
        // 20 minutes of judge time measured nothing. Now, per question, the photos are RANKED by the stored image
        // vector and the judge sees the top `judgeEach` - the same photos a real search would reach first, and the
        // ones most likely to contain the thing being asked about.
        let allWithFacts = Array(have.keys)
        let judgeEach = 100
        let (units, unitItem, _) = await index.units(ids: allWithFacts)
        let ids = allWithFacts
        var imgVec = [String: [Float]]()
        units.withRows(0, units.count) { buf in
            let d = units.dim
            for u in 0..<units.count where imgVec[ids[unitItem[u]]] == nil {
                imgVec[ids[unitItem[u]]] = Array(UnsafeBufferPointer(rebasing: buf[(u * d)..<((u + 1) * d)]))
            }
        }
        for q in questions {
            guard let qv = try? await writer.questionVector(scorerQuestion(q)),
                  let tv = try? emb.vector(of: scorerQuestion(q)) else { continue }
            let folded = head.fold(q: qv, t: tv)
            // the judge's slice: the top `judgeEach` by look score for THIS question
            let look = (try? emb.vector(of: q)).map { [$0] } ?? []
            let cos = look.isEmpty ? [Float](repeating: 0, count: ids.count)
                : lookScores(units: units, unitItem: unitItem, nItems: ids.count, looks: look, avoid: [])
            let pick = Set(cos.indices.sorted { cos[$0] > cos[$1] }.prefix(judgeEach).map { ids[$0] })
            var agree = 0, both = 0, factsYes = 0, judgeYes = 0, missed = 0, falsePos = 0
            var fScores = [Float](), jYes = [Bool]()
            for id in ids where pick.contains(id) {
                guard let row = have[id], let v = imgVec[id] else { continue }
                let pf = folded.p(f: row.vector, v: v)
                guard let img = await PhotoLibrary.ciImage(id, side: 896),
                      let pj = try? await visionJudge.pYes(img, question: q) else { continue }
                both += 1
                let f = Double(pf) >= 0.7, j = pj >= SearchEngine.accept
                if f == j { agree += 1 } else if j { missed += 1 } else { falsePos += 1 }
                if f { factsYes += 1 }; if j { judgeYes += 1 }
                fScores.append(pf); jYes.append(j)
            }
            let auc = rocAUC(scores: fScores, positive: jYes)
            Judge.logMem("FACTSCOMPARE n=\(both) auc=\(auc.map { String(format: "%.3f", $0) } ?? "n/a") "
                         + "agree=\(agree) factsYes=\(factsYes) judgeYes=\(judgeYes) "
                         + "missedByFacts=\(missed) extraFromFacts=\(falsePos)")
        }
        await writer.unload()
        planNote = "M78(b): comparison done - see the log."
        keepIndexingAfterDeveloperRun()
    }

    /// EVERY DEVELOPER-RUN FAILURE GOES TO THE LOG, not just to the screen. Reza, 11-05: "why the HECK is there
    /// aver afilautre you dont see". Because these runners reported failures in `planNote` - a line on the phone -
    /// while I read Documents/memcheck.txt from the Mac. A runner that bailed out in its first second looked
    /// exactly like one still loading models, and that cost nine minutes on #factsCompare and (differently) the
    /// silent nil that hid the missing planner adapter for days. One helper, used by all of them.
    @discardableResult
    func runFailed(_ tag: String, _ why: String) -> Bool {
        Judge.logMem("\(tag) abort: \(why)")
        planNote = "\(tag): \(why)"
        return false
    }

    /// M77: how fast a facts writer runs ON THIS PHONE - the one number that decides whether the index-time
    /// architecture ships (RESULTS 61c-69: same quality as judging every photo, ~65 judge calls per search instead
    /// of ~1,200, but only if writing the facts once is affordable). 20 photos, the 448 px rendition the app
    /// already makes, the writer's own prompt and sampling. NUMBERS ONLY: the facts describe his photos, so no text
    /// of any answer is logged or written anywhere.
    func runFactsSpeed(modelID: String, count: Int = 20) async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        await loadStores()
        stage = .ready
        let entries = await index.entries
        // the same 20 photos for both models, chosen by a fixed rule so the comparison is like for like
        let ids = entries.filter { !$0.value.isVideo }.keys.sorted().prefix(count)
        let m = Judge(modelID: modelID)
        do { try await m.load { _ in } } catch { runFailed("FACTSSPEED", "\(modelID) did not load: \(error)"); return }
        Judge.logMem("FACTSSPEED start \(modelID)")
        var rows: [String] = []
        for (i, id) in ids.enumerated() {
            guard let img = await PhotoLibrary.ciImage(id, side: 448) else { continue }
            do {
                let r = try await m.timeFacts(image: img, prompt: "Search facts for this photo, one line.")
                let tps = r.decode > 0 ? Double(r.tokens) / r.decode : 0
                let row = String(format: "%d prefill=%.2f tokens=%d decode=%.2f tok/s=%.1f thermal=%@ peakGB=%.2f",
                                 i + 1, r.prefill, r.tokens, r.decode, tps,
                                 ThermalLevel(ProcessInfo.processInfo.thermalState).name,
                                 Judge.peakGB)
                rows.append(row)
                Judge.logMem("FACTSSPEED \(modelID) \(row)")
            } catch { Judge.logMem("FACTSSPEED \(modelID) photo \(i + 1) failed: \(error)") }
        }
        await m.unload()
        if let f = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("facts_speed.txt") {
            try? (modelID + "\n" + rows.joined(separator: "\n")).write(to: f, atomically: true, encoding: .utf8)
        }
        planNote = "M77: timed \(rows.count) photos on \(modelID)."
        keepIndexingAfterDeveloperRun()
    }

    /// M72(a): how his face set breaks into EVENTS, with no judge call at all. The new person stop rule walks
    /// events rather than photos (lookWalkOrder, cap 3 per event), so how many events there are and how big they
    /// get decides whether 480 calls can see the whole library or only a corner of it. Counts only.
    func runEventCensus() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        await loadStores()
        stage = .ready
        _ = await pastSearches.load()
        let saved = await pastSearches.saved
        guard let me = saved.first(where: { $0.query.caseInsensitiveCompare("Photos of me") == .orderedSame }) else {
            runFailed("EVENTCENSUS", "run \"photos of me\" once first"); return
        }
        let entries = await index.entries
        let ids = me.ids
        let ev = events(ids.map { entries[$0]?.taken })
        var size = [Int: Int]()
        for e in ev { size[e, default: 0] += 1 }
        let counts = size.values.sorted(by: >)
        let total = ids.count, nEvents = counts.count
        let median = counts.isEmpty ? 0 : counts[counts.count / 2]
        // what the walk can actually reach: at most `lookCap` photos per event, 480 calls
        let reachable = counts.reduce(0) { $0 + min($1, lookCap) }
        let eventsIn480 = { () -> Int in var c = 0, n = 0
            for s in counts.reversed() { let t = min(s, lookCap); if n + t > lookFirstCalls { break }; n += t; c += 1 }
            return c }()
        let line = "EVENTCENSUS photos=\(total) events=\(nEvents) median=\(median) "
            + "largest=\(counts.first ?? 0) singletons=\(counts.filter { $0 == 1 }.count) "
            + "reachableAtCap\(lookCap)=\(reachable) eventsWithin\(lookFirstCalls)calls=\(eventsIn480)"
        Judge.logMem(line)
        planNote = line
        keepIndexingAfterDeveloperRun()
    }

    /// M56: HOW FAR THE SELFIE CUT CAN COME DOWN. The 25/25 at today's cut says nothing about the photos just
    /// below it - that sample only contains photos ABOVE it - so the question is answered by showing Reza the
    /// BAND a looser cut would newly add and asking about those alone (cluster's procedure, M56).
    /// Logs the cuts for 0.95 / 0.90 / 0.85 precision with their recalls on the tagged photos, then puts the
    /// band between the 0.90 cut and today's 0.95 cut on screen as its own album to sample.
    func runSelfieBand(lower: Double = 0.90, upper: Double = 0.95) async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        if embedder == nil { embedder = try? Embedder() }
        if faceEngine == nil { faceEngine = try? FaceEngine() }
        await loadStores()
        stage = .ready
        guard let det = SearchEngine.selfieDetector else { runFailed("SELFIEBAND", "no detector"); return }
        let entries = await index.entries
        var tagged = [String: IndexEntry]()
        for (id, e) in entries where !e.isVideo && e.isScreenshot != true {
            if e.camera == "front" || e.camera == "back" { tagged[id] = e }
        }
        let ids = Array(tagged.keys), front = ids.map { tagged[$0]?.camera == "front" }
        let scores = await index.selfieScoresFor(det, ids: ids)
        let labelled = scores.compactMap { (k, v) -> (score: Float, isFront: Bool)? in
            front.indices.contains(k) ? (score: v, isFront: front[k]) : nil
        }
        var line = "SELFIEBAND cuts"
        var cuts = [Double: Float]()
        for p in Array(Set([0.95, 0.90, 0.85, lower, upper])).sorted(by: >) {
            if let c = selfieCut(labelled, precision: p) {
                cuts[p] = c.cut
                line += String(format: " p%.2f=%.3f(recall %.3f)", p, Double(c.cut), c.recall)
            } else { line += String(format: " p%.2f=none", p) }
        }
        Judge.logMem(line)
        guard let hi = cuts[upper], let lo = cuts[lower], lo < hi else {
            runFailed("SELFIEBAND", "no band between those two cuts"); return
        }
        // THE BAND MUST BE INSIDE HIS FACE SET, which is the thing I got wrong first time and Reza caught before
        // spending a single tap on it: "are they selfies of ME or just selfies in general". The shipped answer is
        // face-matched, so a band computed over the whole library would have him judging other people's selfies -
        // the wrong question. So: run the SHIPPED selfie search at each cut and diff the two answers.
        // HIS FACE SET COMES FROM THE SAVED SEARCHES, not from running the search again. My first attempt called
        // search("selfies") twice from inside a task, and search() cancels whatever task is running - which was the
        // caller itself, so it cancelled itself and the band came back empty (band=0 wide=0 tight=0, 22:53).
        // "Photos of me" IS the face-matched set and "Selfies" is the current answer; both are already stored.
        _ = await pastSearches.load()
        let saved = await pastSearches.saved
        func savedIds(_ name: String) -> Set<String> {
            Set(saved.first { $0.query.caseInsensitiveCompare(name) == .orderedSame }?.ids ?? [])
        }
        let faceSet = savedIds("Photos of me")
        guard !faceSet.isEmpty else {
            runFailed("SELFIEBAND", "run \"photos of me\" once first - the band needs his face set"); return
        }
        // BOTH SETS COMPUTED HERE, not read from a saved search. The saved "Selfies" was contaminated by my own
        // broken attempt, which ran the search with the LOOSE cut forced and overwrote the entry, so the diff was
        // the loose set against itself (band=2 at 22:58). Saved state that an experiment can overwrite is not a
        // baseline.
        let faceIds = Array(faceSet)
        let faceScores = await index.selfieScoresFor(det, ids: faceIds)
        func answer(_ cut: Float) -> Set<String> {
            var out = Set<String>()
            for (k, v) in faceScores where faceIds.indices.contains(k) {
                let id = faceIds[k]
                guard let e = entries[id], !e.isVideo, e.isScreenshot != true, e.camera != "back" else { continue }
                if e.camera == "front" || v >= cut { out.insert(id) }
            }
            return out
        }
        let tight = answer(hi), wide = answer(lo)
        let band = wide.subtracting(tight)
        Judge.logMem("SELFIEBAND band=\(band.count) tight=\(tight.count) wide=\(wide.count) faceSet=\(faceSet.count)")
        var r = AlbumResult(name: "Selfies to double-check")
        r.found = Array(band).sorted { (entries[$0]?.taken ?? 0) > (entries[$1]?.taken ?? 0) }
        r.inScope = band.count; r.judged = 0; r.done = true
        r.note = String(format: "Photos the %.2f-precision cut would ADD over the %.2f one. "
                        + "Answering 25 of these decides whether we keep it.", lower, upper)
        results = [r]
        planNote = "M56: \(band.count) photos in the band. Tap Check a sample and answer 25."
        keepIndexingAfterDeveloperRun()
    }

    /// M57(4): put the cluster's 35 PUBLIC photos on screen in manifest order and ask "Is this a food photo?".
    /// They are Flickr/Commons photos their labellers called not-food or unsure while the judge accepted them, so
    /// Reza's answers say whether his idea of food is broader than the labels the 0.57 was measured against.
    /// Nothing of his is involved; only the file name and his yes/no are recorded.
    func showFoodNotion() async {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("food_notion_35")
        let files = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .filter { $0.hasSuffix(".jpg") }.sorted()
        guard !files.isEmpty else { runFailed("FOODNOTION", "food_notion_35 not on the device"); return }
        // manifest order, when the manifest came with them
        var order = files
        if let m = try? String(contentsOf: dir.appendingPathComponent("manifest.tsv"), encoding: .utf8) {
            let named = m.split(separator: "\n").dropFirst().compactMap { line -> String? in
                let c = line.split(separator: "\t"); return c.count > 1 ? String(c[1]) : nil
            }
            let have = Set(files)
            order = named.filter { have.contains($0) } + files.filter { !named.contains($0) }
        }
        foodNotion = order.map { dir.appendingPathComponent($0) }
        stage = .ready
        planNote = "35 public photos, in the cluster's order: tap Check these."
        Judge.logMem("FOODNOTION ready files=\(order.count)")
        keepIndexingAfterDeveloperRun()
    }

    /// M57(2): what a saved answer is MADE OF - screenshots vs camera photos vs videos - for the whole set and for
    /// the photos Reza hand-checked. The cluster asked because his food answer was 25/25 where theirs was 0.57, and
    /// one candidate explanation is that a lot of his "food photos" are screenshots of other people's food posts,
    /// which are about as unambiguous as a food photo gets. Counts only; no ids, no images, nothing leaves the phone.
    func runComposition(_ query: String) async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        await loadStores()
        _ = await pastSearches.load()
        let saved = await pastSearches.saved
        guard let row = saved.first(where: { $0.query.caseInsensitiveCompare(query) == .orderedSame }) else {
            planNote = "M57: no saved search called \(query)"; return
        }
        let audited = await auditStore.answers(query)
        let judged = Set(audited.yes).union(audited.no)
        let entries = await index.entries
        func describe(_ ids: [String], _ label: String) {
            var shots = 0, videos = 0, photos = 0
            for id in ids {
                guard let e = entries[id] else { continue }
                if e.isVideo { videos += 1 } else if e.isScreenshot == true { shots += 1 } else { photos += 1 }
            }
            let n = max(1, ids.count)
            Judge.logMem("COMPOSITION \(label) n=\(ids.count) screenshots=\(shots) "
                         + "(\(Int(100.0 * Double(shots) / Double(n)))%) videos=\(videos) camera=\(photos)")
        }
        describe(row.ids, "\(query)/all")
        describe(row.ids.filter { judged.contains($0) }, "\(query)/audited")
        planNote = "M57 composition logged for \(query)."
        keepIndexingAfterDeveloperRun()
    }

    /// M50 step 2: score the photos whose camera tag we DO know and report how well the cluster's detector does
    /// on THIS library, before it is allowed to answer anything. The cluster fitted it on 1,280 px Commons
    /// thumbnails; the phone stores 448 px renditions (~0.995 cosine), and Reza's library is not Commons' mix, so
    /// the threshold is re-picked here. Numbers only - no ids, no vectors, nothing leaves the phone.
    func runSelfieValidate() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        await loadStores()
        stage = .ready
        guard let det = SearchEngine.selfieDetector else { planNote = "M50: selfie_head.json missing"; return }
        let entries = await index.entries
        // only photos whose camera is known, and no screenshots (PhotoKit's flag: never a selfie)
        let tagged = entries.filter { !$0.value.isVideo && $0.value.isScreenshot != true
            && ($0.value.camera == "front" || $0.value.camera == "back") }
        let ids = Array(tagged.keys)
        let isFront = ids.map { tagged[$0]?.camera == "front" }
        let t0 = Date()
        let (units, unitItem, _) = await index.units(ids: ids)
        // one row per photo: a photo has a single whole-image vector, so take its first unit
        var scoreOf = [Int: Float]()
        units.withRows(0, units.count) { buf in
            let dim = units.dim
            for u in 0..<units.count where scoreOf[unitItem[u]] == nil {
                let row = UnsafeBufferPointer(rebasing: buf[(u * dim)..<((u + 1) * dim)])
                scoreOf[unitItem[u]] = det.logit(row)
            }
        }
        let labelled = scoreOf.compactMap { (k, v) -> (score: Float, isFront: Bool)? in
            isFront.indices.contains(k) ? (score: v, isFront: isFront[k]) : nil
        }
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        let c = checkSelfieDetector(labelled)
        let line = "SELFIEVALIDATE pos=\(c.positives) neg=\(c.negatives) auc=\(String(format: "%.4f", c.auc)) "
            + "recall@0.5=\(String(format: "%.3f", c.recallAtHalf)) fpr@0.5=\(String(format: "%.4f", c.falsePositiveRateAtHalf)) "
            + "t95=\(c.threshold95.map { String(format: "%.3f", $0) } ?? "none") "
            + "recall@t95=\(String(format: "%.3f", c.recallAt95)) scored=\(labelled.count) in \(ms) ms"
        Judge.logMem(line)
        if let f = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("selfie_validate.txt") {
            try? line.write(to: f, atomically: true, encoding: .utf8)
        }
        planNote = line
        keepIndexingAfterDeveloperRun()
    }


    /// The one-shot query in Documents/next_query.txt, consumed (deleted) on read so it can never run twice.
    static func fileQuery() -> String? {
        guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        let f = dir.appendingPathComponent("next_query.txt")
        guard let text = try? String(contentsOf: f, encoding: .utf8) else { return nil }
        try? FileManager.default.removeItem(at: f)
        let q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return q.isEmpty ? nil : q
    }

    /// Developer-only (launch argument `-runQuery "<text>"`, docs/MAC_INBOX.md M5/M6): run ONE search without
    /// anyone typing, so a timed run can be driven from the Mac and repeated identically on another judge.
    /// Like -selfCheck this must not sit behind start()'s indexing await: the index is already on disk, and the
    /// re-embed that await waits for takes ~13 min. It deliberately does NOT re-index first.
    func runDebugQuery(_ q: String) async {
        // LOG THE QUERY FIRST, before anything slow. 10-11 21:27: I launched `-runQuery "selfies"`, waited ten
        // minutes, and the device log showed the app judging "photos of a dog" - the argument from an EARLIER
        // launch. devicectl remembers launch arguments per bundle id (the same mechanism behind the ten-hour idle,
        // F5), so a run can silently measure the wrong thing. This line makes the argument that actually arrived
        // readable within a second of the app starting, so I can never again time a query that is not running.
        Judge.logMem("ONEQUERY start: \(q)")
        // The result of the LAST run must not be mistaken for this one's if this run dies.
        if let old = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("query_result.json") { try? FileManager.default.removeItem(at: old) }
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
        // WRITE THE NUMBERS TO DISK, not to the screen. A screenshot of a result grid is Reza's own photos -
        // the wrong thing to be capturing - and a stall and a finished search look the same in a picture
        // anyway (10-10). One row per album, rewritten after the search, pulled with `devicectl copy from`.
        let t0 = Date()
        search(q)
        while busy, !Task.isCancelled { try? await Task.sleep(for: .milliseconds(500)) }
        let secs = Int(Date().timeIntervalSince(t0))
        var out: [[String: Any]] = []
        for r in results {
            out.append(["query": q, "album": r.name, "found": r.found.count, "judged": r.judged,
                        "inScope": r.inScope, "seconds": secs, "note": r.note, "engine": effectiveEngine])
        }
        if results.isEmpty {
            out.append(["query": q, "album": "", "found": 0, "judged": 0, "inScope": 0,
                        "seconds": secs, "note": planNote, "engine": effectiveEngine])
        }
        if let file = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("query_result.json"),
           let d = try? JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted]) {
            try? d.write(to: file)
        }
        Judge.logMem("ONEQUERY done (\(secs) s): \(q)")
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
        let previous = searchTask
        previous?.cancel(); searchGen += 1
        let today = Day(iso: ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate]))!
        let hist = followUp ? history : [], cur = followUp ? currentPlan : nil
        busy = true; lastQuery = text; results = []; planNote = ""
        searchTask = Task {
            await previous?.value       // let a search that is still looking stop at its current photo first
            var phase = "planning"      // which half threw: Apple's planner and its judge fail differently
            do {
                // THE MODEL SWAP MOVED BELOW, INTO THE BRANCH THAT ACTUALLY PLANS (M100(2)). It used to run
                // before every search, including ones whose plan we already have: execute() unloads the planner to
                // make room for the photo judge, so on the SECOND and later searches the planner is gone and
                // planning dies with "Judge Code=1" (measured 10-10 14:24: query 1 found 209 dogs in 42 min, then
                // queries 2-10 each failed in 0 s). Only one of the two ~2.9 GB models fits at a time
                // (2.83 + 2.88 vs ~5.8 GB free), so it is a swap, not a both-resident fix. On a cache hit there is
                // nothing to plan, so the swap - and its two model loads - is skipped as well.
                // TIME THE PLANNER. 10-11 22:07: a selfies run that needs NO judge call still produced nothing for
                // eight minutes after the planner loaded, and the log could not say whether planning, the scope
                // filter or the UI was the slow part. The planner is a 4B text model on the same hot GPU, so it
                // is a cost in its own right and must be visible.
                // ALONGSIDE THE PLANNER, NOT IN FRONT OF IT. Measured on his library: this guess takes 6.0 s -
                // one text embedding is cheap, but mapping and scanning all 188,338 stored vectors is not - so
                // awaiting it here added 6 s to every search and the final answer arrived LATER than before. The
                // planner is GPU work and this is BLAS on the CPU, so they overlap: the guess lands at ~6 s, the
                // plan at ~9 s, and the total is unchanged.
                // Three guards, because this now writes to the screen from outside the search's own order:
                // the generation (a newer search must never be painted over), cancellation, and results.isEmpty
                // (if the plan already landed and the engine has published, the guess is stale and is dropped).
                let gen = self.searchGen
                if !followUp {
                    Task { @MainActor in
                        let fg0 = Date()
                        guard let first = await self.provisionalAnswer(text) else { return }
                        guard gen == self.searchGen, !Task.isCancelled, self.results.isEmpty else {
                            Judge.logMem("FIRSTGUESS dropped: the planned answer got there first")
                            return
                        }
                        self.results = [first]
                        Judge.logMem("FIRSTGUESS \(first.found.count) shown in "
                                     + "\(Int(Date().timeIntervalSince(fg0) * 1000)) ms, 0 judge calls")
                    }
                }
                let planT0 = Date()
                let planKey = AppModel.planKey(text, today: today)
                let plan: Plan
                if !followUp, let hit = AppModel.cachedPlan(planKey) {
                    plan = hit
                    Judge.logMem("PLANCACHE hit in \(Int(Date().timeIntervalSince(planT0) * 1000)) ms: "
                                 + "albums=\(plan.albums.count), planner not loaded")
                } else {
                    if !(await judge.isLoaded), !qwenOutOfMemory {
                        await self.visionJudge.unload()
                        try await self.judge.load { _ in }
                    }
                    Judge.logMem("PLAN start: \(text)")
                    plan = try await activePlanner.plan(text, history: hist, current: cur, today: today)
                    Judge.logMem("PLAN done (\(Int(Date().timeIntervalSince(planT0))) s): albums=\(plan.albums.count) "
                                 + "q=\(plan.albums.first?.judgeQuestion ?? "none") camera=\(plan.albums.first?.camera ?? "any")")
                    if !followUp { AppModel.cachePlan(plan, for: planKey) }
                }
                self.currentPlan = plan; self.history = hist + [text]; self.planNote = AppModel.plainNotes(plan.notes)
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
            await self.rememberResults(text)
            await self.keepLooking()
        }
    }

    // THE PLAN CACHE (M100(2)). Planning the same words twice costs 9-12 s twice, and it is the same 4B model on
    // the same words producing the same plan - it is greedy, so there is nothing to re-roll.
    // THE DAY IS PART OF THE KEY on purpose: "photos from last week" means something different tomorrow, and
    // checking whether a particular plan happens to be date-dependent is a thing to get subtly wrong. A repeat on
    // a later day simply plans again.
    // A FOLLOW-UP IS NEVER CACHED OR SERVED FROM THE CACHE: its plan depends on the conversation so far, so the
    // same words mean different things in different places.
    // The file holds the person's own queries on his own phone, next to PastSearches which already does; nothing
    // is logged or sent anywhere, and the log lines about it carry no query text.
    struct CachedPlan: Codable { let plan: Plan; let at: Date }
    static let planCacheMax = 300

    static func planKey(_ text: String, today: Day) -> String {
        let t = text.lowercased().split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        return "\(today.y)-\(today.m)-\(today.d)|\(t)"
    }

    private static var planCacheURL: URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("plan_cache.json")
    }

    static func cachedPlan(_ key: String) -> Plan? {
        guard let u = planCacheURL, let d = try? Data(contentsOf: u),
              let all = try? JSONDecoder().decode([String: CachedPlan].self, from: d) else { return nil }
        return all[key]?.plan
    }

    static func cachePlan(_ plan: Plan, for key: String) {
        guard let u = planCacheURL else { return }
        var all = [String: CachedPlan]()
        if let d = try? Data(contentsOf: u),
           let old = try? JSONDecoder().decode([String: CachedPlan].self, from: d) { all = old }
        all[key] = CachedPlan(plan: plan, at: Date())
        if all.count > planCacheMax {
            for k in all.sorted { $0.value.at < $1.value.at }.prefix(all.count - planCacheMax).map(\.key) {
                all.removeValue(forKey: k)
            }
        }
        if let d = try? JSONEncoder().encode(all) { try? d.write(to: u) }
    }

    /// THE FIRST SCREEN BEFORE THE PLANNER HAS SAID ANYTHING (M100(2)). Measured on his library 11-06: every
    /// search spent 9-12 s of its 17-20 s inside the planner (a 4B text model, plus the swap that unloads the photo
    /// judge to make room for it), while scoring all 148,451 photos took 1.9 s. So for most of a search the screen
    /// had nothing on it for the one reason the person cannot see.
    /// This ranks the library by the WORDS THEY TYPED - one text embedding and a dot product per photo, no planner,
    /// no judge - and puts a screenful up at once. It is the same ranking the engine's own vector-first stage uses,
    /// so when the planned answer arrives the list is refined rather than reshuffled at random.
    /// It is labelled as a guess, `done` is false and `stillLooking` is true, so nothing on screen claims this is
    /// the answer to the question they asked - only to the words they used.
    func provisionalAnswer(_ text: String) async -> AlbumResult? {
        guard let emb = embedder, let look = try? emb.vector(of: text) else { return nil }
        let entries = await index.entries
        guard !entries.isEmpty else { return nil }
        let ids = entries.map { $0.key }
        let (units, unitItem, _) = await index.units(ids: ids)
        guard units.count > 0 else { return nil }
        let cos = lookScores(units: units, unitItem: unitItem, nItems: ids.count, looks: [look], avoid: [])
        let top = cos.indices.sorted { cos[$0] > cos[$1] }.prefix(60).map { ids[$0] }
        guard !top.isEmpty else { return nil }
        var r = AlbumResult(name: "")
        r.provisional = true
        r.found = top
        // inScope STAYS 0 ON PURPOSE. This pass ranks everything in the index, videos included (188,338 items),
        // while the planned answer for "photos of a dog" scopes to 148,451 photos - so quoting a scope here would
        // put a number on screen that the real answer contradicts a few seconds later. The note is the only claim
        // the guess makes, and progressLine shows nothing while judged == 0.
        r.judged = 0
        r.stillLooking = true
        r.note = "A first guess from the words you typed. No photo has been opened, and the search is still "
               + "working out what you mean."
        return r
    }

    /// Keep what a search found, so it can be reopened - and a sample of it checked - without running it again.
    func rememberResults(_ query: String) async {
        for r in results where !r.found.isEmpty {
            await pastSearches.add(SavedSearch(query: results.count > 1 ? "\(query) — \(r.name)" : query,
                                               ids: r.found, judged: r.judged, inScope: r.inScope,
                                               note: r.note, at: Date()))
        }
    }

    /// The current plan again, without planning: after a face or a pet's photos were picked, or "Look at everything".
    /// NOT OFFERED, ON PURPOSE (M97(2)). Verifying only the photos on screen is built in SearchEngine
    /// (run(verifyShownOnly:)) and it works, but it is not a button yet: RESULTS 71 decides, per request category,
    /// whether the judge-free answer's precision is under .90 there, and only those categories get it. Until that
    /// number exists, offering an hour of judging to remove wrong photos from an answer measured at ~.97 precision
    /// would be spending the person's evening to confirm what we already believe - and, worse (M98(1)), leading with
    /// it steers people to the SMALLER error and leaves them believing the album has been checked.
    /// WHEN IT IS ENABLED, ITS PRICE IS: one judge call per photo shown, PLUS one per follow-up leaf per photo
    /// (exclude / filter), not just the photo count (M98(3)).
    static func plainTime(_ s: Double) -> String {
        if s < 90 { return "\(Int((s / 5).rounded()) * 5) seconds" }
        if s < 90 * 60 { return "\(Int((s / 60).rounded())) minutes" }
        return String(format: "%.1f hours", s / 3600)
    }

    /// WHAT THE CAREFUL CHECK WOULD COST across the albums on screen, or nil when there is nothing to offer.
    /// The seconds are a floor turned into plain words: the cascade commits to at least this many judge calls, each
    /// measured at ~2.4 s on this phone, and an extension can add more - so the button says "about" and the note
    /// below it says to plug in when the number is big.
    /// M100(3)(b): true when every album with a judge-free answer had no judge-kind leaf. R59d measures a
    /// kind-less leaf as ANSWERED by the millisecond scorer - within -.03 F1 of judging everything - so for
    /// "photos of a dog" the careful check is the same computation. Saying nothing would leave the person assuming
    /// a careful check exists and we just did not offer it.
    /// What "Look at everything" really costs: every photo in scope opened by the judge, at the measured seconds
    /// per call. M100(3)(b) wants this said out loud next to the button, because for an object question it is the
    /// ONLY thing beyond the quick check.
    var lookAtEverythingPrice: String? {
        let n = results.map(\.inScope).reduce(0, +)
        guard n > 0 else { return nil }
        return AppModel.plainTime(Double(n) * SearchEngine.secondsPerJudgeCall)
    }

    var carefulSameAnswer: Bool {
        carefulCost == nil && results.contains { $0.noCarefulCheck }
    }

    var carefulCost: (calls: Int, seconds: Double, words: String)? {
        let calls = results.compactMap(\.checkCalls).reduce(0, +)
        guard calls > 0 else { return nil }
        let s = results.compactMap(\.checkSeconds).reduce(0, +)
        return (calls, s, AppModel.plainTime(s))
    }

    func rerun(exhaustive: Bool = false, careful: Bool = false, verifyShownOnly: Bool = false) {
        guard let plan = currentPlan, embedder != nil else { return }
        let previous = searchTask
        previous?.cancel(); searchGen += 1
        busy = true; results = []
        let msg = history.last ?? lastQuery, earlier = Array(history.dropLast())
        searchTask = Task {
            await previous?.value
            do {
                let saved = await self.subjects.saved
                self.currentAsks = namedSubjects(plan, message: msg, history: earlier, saved: saved)
                try await self.execute(plan, exhaustive: exhaustive, careful: careful,
                                       verifyShownOnly: verifyShownOnly)
            } catch { self.planNote = "Could not run this search: \(error)" }
            await self.noteAppleRefusals()
            await self.refreshBursts()
            self.busy = false
            await self.keepLooking()
        }
    }

    /// RESULTS 44b (Reza 10-10: no cap that drops accuracy). Every album's first answer is on screen; now keep looking
    /// through exhaustive mode's rounds, judging only while the phone is cool enough (SearchEngine.run, paced), until
    /// everything is checked, the phone gets too hot, or the person taps Stop / starts another search.
    private func keepLooking() async {
        let looks = pendingLooks; pendingLooks = []
        guard !looks.isEmpty, !Task.isCancelled else { return }
        lookingOn = true
        for look in looks {
            if Task.isCancelled { break }
            do { try await look() } catch { planNote = "Stopped looking: \(error.localizedDescription)" }
        }
        lookingOn = false
        await refreshBursts()
    }

    /// The Stop button while a search keeps looking: it ends at the photo being judged; what was found stays.
    func stopLooking() { searchTask?.cancel() }

    private func execute(_ plan0: Plan, exhaustive: Bool, careful: Bool = false,
                         verifyShownOnly: Bool = false) async throws {
        guard let embedder = embedder else { return }
        pendingLooks = []
        let gen = searchGen
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
            // "only from Paris" -> GPS place filter; a place no photo has ("beach") -> a visual condition; a date
            // without a year whose latest one holds no photo here -> the latest earlier year that has photos
            let (album, yearsMoved) = earlierYearIfEmpty(libItems, placeOrLook(libItems, filterToPlace(libItems, album0)))
            let yearNote = yearsMoved == 0 ? "" : "No photo from '\(album.timePhrase ?? "")' in the latest one, so this "
                + "searched the latest year that has photos: \(album.dateFrom ?? "") to \(album.dateTo ?? "now") "
                + "(\(yearsMoved) year(s) earlier)."
            // THE GUESS BECOMES THIS CARD, it does not sit next to it. Two things would break otherwise: the
            // person would see a stale "first guess" card beside the real answer, and every self.results[k] below
            // would be off by one, because the engine writes results by ALBUM index.
            // The guessed photos stay visible under the album's own name until the engine publishes, so the screen
            // never goes from a screenful back to nothing.
            if k == 0, self.results.count == 1, self.results[0].provisional {
                var carried = AlbumResult(name: album.name)
                carried.found = self.results[0].found
                carried.stillLooking = true
                carried.note = "Still the first guess from the words you typed. The search is working on this now."
                self.results[0] = carried
            } else {
                self.results.append(AlbumResult(name: album.name))
            }
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
                let negatives = await people.negatives(for: person, owner: owner)    // faces marked "not them" [F8]
                let ps = try await engine.runPerson(album, refs: refs, others: others, negatives: negatives) { r in Task { @MainActor in
                    if k < self.results.count { self.results[k].judged = r.judged; self.results[k].inScope = r.inScope } } }
                personScores[k] = ps
                var r = AlbumResult(name: album.name); r.inScope = ps.ids.count; r.judged = ps.pYes.count; r.done = true
                r.note = [yearNote, uncheckedFacesNote(ps.unchecked) ?? ""].filter { !$0.isEmpty }.joined(separator: " ")
                if album.judgeQuestion == nil { r.found = ps.ids } else {
                    // how someone LOOKS stays relative to their own photos (top half); a FACT ("wearing a pink
                    // helmet", "at the beach", "a selfie") keeps the judge's yes photos only (FindPicsCore
                    // personAlbumFound = engine.stream_album, RESULTS 49c)
                    r.found = personAlbumFound(pYes: ps.pYes, question: album.judgeQuestion, accept: SearchEngine.accept)
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
                let (scope, momentNote0) = try await engine.momentScope(album)
                let momentNote = [yearNote, momentNote0].filter { !$0.isEmpty }.joined(separator: " ")
                if let sc = scope, sc.isEmpty {
                    self.results[k].note = momentNote; self.results[k].done = true; continue
                }
                var inner = album
                if scope != nil { inner.dateFrom = nil; inner.dateTo = nil; inner.timePhrase = nil; inner.place = nil }
                let (withOK, unknown, withUnchecked) = await engine.withPeople(album, people: people, owner: owner)
                let onUpdate: @Sendable (AlbumResult) -> Void = { r in Task { @MainActor in
                    guard gen == self.searchGen, k < self.results.count else { return }
                    var r = r
                    if let ok = withOK {
                        let notChecked = r.found.filter { !ok.contains($0) && withUnchecked.contains($0) }.count
                        r.found = r.found.filter { ok.contains($0) }
                        if let n = uncheckedFacesNote(notChecked) { r.note += (r.note.isEmpty ? "" : " ") + n }
                    }
                    if !momentNote.isEmpty { r.note = momentNote + " " + r.note }
                    if !unknown.isEmpty { r.note += " Not known yet: \(unknown.joined(separator: ", ")) (pick their face once to include them)." }
                    self.results[k] = r } }
                try await engine.run(inner, exhaustive: exhaustive, restrictTo: scope, paced: true,
                                     careful: careful, verifyShownOnly: verifyShownOnly, update: onUpdate)
                if !exhaustive {
                    // RESULTS 44b: the first answer is in; keep looking once every album has one (keepLooking)
                    pendingLooks.append { [weak self] in
                        guard let self, gen == self.searchGen, k < self.results.count, self.results[k].stillLooking else { return }
                        try await engine.run(inner, exhaustive: true, restrictTo: scope, paced: true,
                                             continuing: self.results[k], update: onUpdate)
                    }
                }
            }
        }
        // two opposite looks of the same person ("heavier" vs "fit"): split by the two groups the scores form
        let pk = personScores.keys.sorted()
        // only LOOKS are paired: two facts ("wearing glasses" / "not wearing glasses") each keep the judge's yes photos
        if pk.count == 2, plan.albums[pk[0]].person == plan.albums[pk[1]].person,
           isLookQuestion(plan.albums[pk[0]].judgeQuestion), isLookQuestion(plan.albums[pk[1]].judgeQuestion),
           let a = personScores[pk[0]], let b = personScores[pk[1]] {
            let entries = await index.entries
            let ids = Array(Set(a.pYes.keys).intersection(b.pYes.keys))
            let ev = events(ids.map { entries[$0]?.taken })
            let eventOf = Dictionary(uniqueKeysWithValues: zip(ids, ev.map { String($0) }))
            // two clear groups -> the split; otherwise the rank margin (engine.make_exclusive fallback)
            // lookPairAlbums, not splitPair: the walk now stops at a first answer, so these scores are PARTIAL -
            // a few hundred photos of his 41,307 - and lookPairAlbums is the cluster's rule for exactly that case
            // (it falls back to the old split once everything is judged; measured identical on 30/30 saved runs).
            if let split = Optional(lookPairAlbums(ids: ids, eventOf: eventOf, pA: a.pYes, pB: b.pYes)) {
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

    /// The "Who is X?" sheet (FacePickerSheet). `fromPeople`: opened from the People screen (no search re-run after).
    func openFacePicker(for person: String, fromPeople: Bool = false) async {
        await ensureGroups()
        let s = await people.suggestion(for: person, owner: owner, index: index)
        faceGroupsShown = await people.groups
        faceGen = await people.groupsGen
        faceSuggestion = s.suggested; faceOthers = s.others; askingFor = person
        faceNote = await people.reaskNote(person, owner: owner) ?? ""
        pickerFromPeople = fromPeople
        askWhichFace = true
    }

    func closeFacePicker() { askWhichFace = false; faceNote = "" }

    private var askingKey: String { isOwnerWord(askingFor) ? owner : askingFor }

    /// A saved person changed: close the picker; re-run the search that asked (not when opened from People).
    private func facePicked() {
        askWhichFace = false; faceNote = ""; peopleVersion += 1
        if !pickerFromPeople { rerun() }
    }

    func typicalFaces(_ g: FaceGroup) async -> [FaceSource] { await people.typicalFaces(g, gen: faceGen) }

    func reviewFaces(_ g: FaceGroup) async -> GroupReview? {
        await people.review(g, gen: faceGen, person: askingFor, owner: owner)
    }

    /// Confirm a reviewed group minus the faces the person left out (positions in g.faces; [F8]).
    func chooseFace(_ g: FaceGroup, excluding: Set<Int>) {
        let key = askingKey, gen = faceGen
        Task {
            if await people.name(group: g, excluding: excluding, gen: gen, as: key) { facePicked() }
            else { faceNote = "The face groups changed while new photos were being read. Close this and open it again." }
        }
    }

    /// People screen: remove faces from a saved person (they become "not this person").
    func removeFaces(key: String, positions: Set<Int>) async -> Bool {
        let ok = await people.removeFaces(key: key, positions: positions)
        peopleVersion += 1
        return ok
    }

    /// People screen: forget a saved person (people.json only; photos are never touched).
    func forgetPerson(key: String) async {
        await people.forget(key: key)
        peopleVersion += 1
    }

    /// "Add a photo of them": faces found in 1-3 picked photos (the one face they share, FindPicsCore.pickRefFaces).
    func addFacePhotos(_ ids: [String]) {
        guard let fe = faceEngine else { faceNote = "The face model is not loaded."; return }
        let key = askingKey
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
            await people.name(refs: refs, sources: sources, as: key)
            facePicked()
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
