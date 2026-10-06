// find pics: type what you are looking for, like the Photos search bar; everything runs on this phone.
import FindPicsCore
import SwiftUI

@main
struct FindPicsApp: App {
    @StateObject var model = AppModel()
    var body: some Scene { WindowGroup { RootView().environmentObject(model) } }
}

@MainActor
final class AppModel: ObservableObject {
    enum Stage: Equatable { case start, noAccess, askDownload, downloading(Double), indexing(Int, Int), ready, failed(String) }
    /// App Store guideline 4.2.3(ii): say how big the download is and ask before fetching the models.
    static let downloadGB = 3.1          // mlx-community/Qwen3.5-4B-4bit (3.06 GB) + tokenizer files
    @AppStorage("modelDownloadAccepted") var downloadAccepted = false
    @Published var stage: Stage = .start
    @Published var results: [AlbumResult] = []
    @Published var busy = false
    @Published var lastQuery = ""
    @Published var planNote = ""
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
    var embedder: Embedder?
    var faceEngine: FaceEngine?
    @Published var askWhichFace = false
    @Published var faceGroupsShown: [FaceGroup] = []
    var owner = "me"
    var history: [String] = []
    var currentPlan: Plan?
    var searchTask: Task<Void, Never>?

    func start() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        do {
            embedder = try Embedder()
            faceEngine = try? FaceEngine()
            await people.load()
            guard downloadAccepted else { stage = .askDownload; return }
            try await judge.load { p in Task { @MainActor in self.stage = .downloading(p) } }
            await index.load()
            let assets = PhotoLibrary.allAssets()
            stage = .indexing(0, assets.count)
            await index.build(assets: assets, embedder: embedder!, faceEngine: faceEngine) { d, t in Task { @MainActor in self.stage = .indexing(d, t) } }
            await people.refreshGroups(index: index)
            stage = .ready
        } catch { stage = .failed(String(describing: error)) }
    }

    /// A new request, or a follow-up that edits the current search ("only the ones outdoors").
    func search(_ text: String, exhaustive: Bool = false, followUp: Bool = false) {
        guard let embedder = embedder else { return }
        searchTask?.cancel()
        let today = Day(iso: ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate]))!
        let hist = followUp ? history : [], cur = followUp ? currentPlan : nil
        busy = true; lastQuery = text; results = []; planNote = ""
        searchTask = Task {
            do {
                let plan = try await Planner(judge: judge).plan(text, history: hist, current: cur, today: today)
                self.currentPlan = plan; self.history = hist + [text]; self.planNote = plan.notes
                let engine = SearchEngine(index: index, embedder: embedder, judge: judge)
                var personScores: [Int: PersonScored] = [:]
                for (k, album) in plan.albums.enumerated() {
                    self.results.append(AlbumResult(name: album.name))
                    if let person = album.person, !person.isEmpty {
                        guard let refs = await people.refs(for: person, owner: owner) else {
                            self.faceGroupsShown = await people.groups; self.askWhichFace = true
                            self.results[k].note = "Who is \(person)? Pick their face once (the faces I see most often)."; self.results[k].done = true
                            continue
                        }
                        let others = await people.otherGroups(excluding: refs)
                        let ps = try await engine.runPerson(album, refs: refs, others: others) { r in Task { @MainActor in
                            if k < self.results.count { self.results[k].judged = r.judged; self.results[k].inScope = r.inScope } } }
                        personScores[k] = ps
                        var r = AlbumResult(name: album.name); r.inScope = ps.ids.count; r.judged = ps.pYes.count; r.done = true
                        if album.judgeQuestion == nil { r.found = ps.ids } else {
                            let rel = withinPersonRank(ps.pYes)
                            r.found = ps.ids.filter { (rel[$0] ?? 0) > 0.5 }.sorted { (ps.pYes[$0] ?? 0) > (ps.pYes[$1] ?? 0) }
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
                    if let split = splitPair(pA: a.pYes, pB: b.pYes, eventOf: eventOf) {
                        for (slot, k) in pk.enumerated() {
                            self.results[k].found = split.filter { $0.value == slot }.map { $0.key }
                                .sorted { (personScores[k]!.pYes[$0] ?? 0) > (personScores[k]!.pYes[$1] ?? 0) }
                            self.results[k].note = "Paired with '\(plan.albums[pk[1 - slot]].name)': each photo goes to the album its scores clearly belong to; \(split.values.filter { $0 == nil }.count) photo(s) are in neither."
                        }
                    }
                }
            } catch { self.planNote = "Could not run this search: \(error)" }
            await self.refreshBursts()
            self.busy = false
        }
    }
}

struct RootView: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        switch model.stage {
        case .start: ProgressView("Starting…").task { await model.start() }
        case .noAccess: Text("find pics needs access to your photos to search them. Settings > Privacy > Photos > find pics.").padding()
        case .askDownload:
            VStack(spacing: 16) {
                Text("find pics needs to download its on-phone AI once: about \(String(format: "%.1f", AppModel.downloadGB)) GB.")
                Text("Use Wi-Fi if you can. After this download, your photos are searched entirely on this phone and nothing is uploaded.")
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
