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
    @Published var engine = UserDefaults.standard.string(forKey: "engine") ?? "qwen" {
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
    @Published var indexProgress: IndexProgress?  // new photos / iCloud downloads in progress (banner)
    @Published var bursts: [UUID: [Int]] = [:]     // per album: burst group of each found photo (display only)

    /// Near-identical shots of one moment -> one stack ("+N similar"); every photo stays in the album.
    func refreshBursts() async {
        let entries = await index.entries
        for r in results where r.done {
            let v = r.found.map { entries[$0]?.vector ?? [] }, t = r.found.map { entries[$0]?.taken }
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
    private var pendingChange = LibraryChange()
    private var changeDebounce: Task<Void, Never>?

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

    func start() async {
        #if DEBUG
        if ProcessInfo.processInfo.arguments.contains("-demoUI") { await startDemoUI(); return }
        #endif
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
        let a = await people.load(), b = await subjects.load(), c = await index.load()
        storesLoaded = a && b && c
        indexStatus = await index.summary() ?? ""
        return storesLoaded
    }

    // MARK: keeping the index current

    /// What indexing needs (no judge, no download consent: the image / face models are bundled).
    private func prepareIndexing() async -> Bool {
        guard PhotoLibrary.hasAccess else { return false }
        if embedder == nil { embedder = try? Embedder() }
        if faceEngine == nil { faceEngine = try? FaceEngine() }
        guard await loadStores() else { return false }
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

    private func showProgress(_ p: IndexProgress) {
        if !initialIndexDone, !p.downloading, case .indexing = stage { stage = .indexing(p.done, p.total); return }
        if p.downloading, case .indexing = stage { stage = .ready }      // local pass done: searchable now
        indexProgress = p
    }

    /// The whole library against the index: deleted assets out, new ones in, then iCloud downloads if allowed.
    private func indexLibrary(purpose: FetchPurpose, retryFailed: Bool) async {
        guard await prepareIndexing(), let emb = embedder else { return }
        let assets = PhotoLibrary.allAssets()
        let indexed = Array(await index.entries.keys)
        await index.remove(removedFromLibrary(indexed: indexed, library: Set(assets.map(\.id))))
        await index.update(assets: assets, embedder: emb, faceEngine: faceEngine, purpose: purpose, retryFailed: retryFailed,
                           progress: progressHandler())
        await afterIndexChange()
    }

    private func afterIndexChange() async {
        initialIndexDone = true
        groupsStale = true                 // face groups are rebuilt when the face picker next needs them
        indexStatus = await index.summary() ?? ""
        indexProgress = nil
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
            await self.index.update(assets: c.inserted, embedder: emb, faceEngine: self.faceEngine, purpose: .indexForeground,
                                    retryFailed: false, progress: self.progressHandler())
            await self.afterIndexChange()
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
        let t = enqueueIndexing { await self.indexLibrary(purpose: .indexBackground, retryFailed: true) }
        cancel.set(t)
        await t.value
        return !cancel.isCancelled
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

    /// A new request, or a follow-up that edits the current search ("only the ones outdoors").
    func search(_ text: String, exhaustive: Bool = false, followUp: Bool = false) {
        guard embedder != nil else { return }
        searchTask?.cancel()
        let today = Day(iso: ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate]))!
        let hist = followUp ? history : [], cur = followUp ? currentPlan : nil
        busy = true; lastQuery = text; results = []; planNote = ""
        searchTask = Task {
            do {
                let plan = try await activePlanner.plan(text, history: hist, current: cur, today: today)
                self.currentPlan = plan; self.history = hist + [text]; self.planNote = plan.notes
                let saved = await self.subjects.saved
                self.currentAsks = namedSubjects(plan, message: text, history: hist, saved: saved)
                try await self.execute(plan, exhaustive: exhaustive)
            } catch { self.planNote = "Could not run this search: \(error)" }
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
            await self.refreshBursts()
            self.busy = false
        }
    }

    private func execute(_ plan0: Plan, exhaustive: Bool) async throws {
        guard let embedder = embedder else { return }
        // effectiveEngine, not engine: when the weights do not fit, the menu may still say qwen3vl/vote but Apple's
        // model is what runs, and loading ~2.5 GB here would be exactly the kill we are avoiding.
        let e = self.effectiveEngine
        if e == "qwen3vl" || e == "vote" { try await self.visionJudge.load { _ in } }   // first use downloads ~2.5 GB
        let engine = SearchEngine(index: index, embedder: embedder, judge: activeJudge)
        // a named pet / thing ("my dog Max"): its album becomes a subject search (FindPicsCore.subjectPlan)
        let asks = currentAsks
        let plan = subjectPlan(plan0, asks: asks)
        var personScores: [Int: PersonScored] = [:]
        var asked = false                       // one question per run (face picker or pet photos)
        let libItems = await index.libraryItems(order: Array(await index.entries.keys))
        for (k, album0) in plan.albums.enumerated() {
            // "only from Paris" -> GPS place filter; a place no photo has ("beach") -> a visual condition
            let album = placeOrLook(libItems, filterToPlace(libItems, album0))
            self.results.append(AlbumResult(name: album.name))
            if let ask = asks.first(where: { $0.albums.contains(k) }) {
                try await runSubjectAlbum(k, album, ask: ask, engine: engine, asked: &asked)
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
                if album.judgeQuestion == nil { r.found = ps.ids } else {
                    let rel = withinPersonRank(ps.pYes)
                    // a fact about the photo ("a selfie", "at the beach"): a clear yes counts even outside the
                    // person's top half; how someone LOOKS stays relative to their own photos (engine.LOOK_WORDS)
                    let fact = album.judgeQuestion?.range(of: #"(?i)\b(look|looks|looking|appear|appears|appearing|seem|seems|seeming)\b"#,
                                                          options: .regularExpression) == nil
                    r.found = ps.ids.filter { (rel[$0] ?? 0) > 0.5 || (fact && (ps.pYes[$0] ?? 0) >= SearchEngine.accept) }
                        .sorted { (ps.pYes[$0] ?? 0) > (ps.pYes[$1] ?? 0) }
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
                let (withOK, unknown) = await engine.withPeople(album, people: people, owner: owner)
                try await engine.run(inner, exhaustive: exhaustive, restrictTo: scope) { r in Task { @MainActor in
                    guard k < self.results.count else { return }
                    var r = r
                    if let ok = withOK { r.found = r.found.filter { ok.contains($0) } }
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
        let (withOK, unknown) = await engine.withPeople(album, people: people, owner: owner)
        var r = try await engine.runSubject(examples: examples, name: ask.judgeName, kind: ask.kind, album: inner, restrictTo: scope) { r in
            Task { @MainActor in
                guard k < self.results.count else { return }
                var r = r
                if let ok = withOK { r.found = r.found.filter { ok.contains($0) } }
                self.results[k] = r
            }
        }
        if let ok = withOK { r.found = r.found.filter { ok.contains($0) } }
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
        faceSuggestion = s.suggested; faceOthers = s.others; askingFor = person; faceNote = ""
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
            for id in ids {
                var ui: UIImage? = nil
                if case .full(let u) = await PhotoLibrary.read(id, side: 1280, purpose: .judge) { ui = u }
                guard let cg = ui?.cgImage, let fs = try? fe.faces(in: cg) else { continue }
                photos.append(fs.map { (px: $0.px, emb: $0.embedding) })
            }
            let refs = pickRefFaces(photos: photos)
            guard !refs.isEmpty else {
                faceNote = "No face found in those photos. Pick photos where the face is large and clear."
                return
            }
            await people.name(refs: refs, as: askingKey)
            askWhichFace = false
            rerun()
        }
    }
}

struct RootView: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        switch model.stage {
        case .start: ProgressView("Starting…").task { model.launch() }
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
        case .indexing(let d, let t): VStack { ProgressView(value: Double(d), total: Double(max(t, 1))); Text("Reading your library once: \(d) of \(t). Keep the app open and plugged in.") }.padding()
        case .failed(let e): Text("Something went wrong: \(e)").padding()
        case .ready: SearchView()
        }
    }
}
