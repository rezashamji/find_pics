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
    enum Stage: Equatable { case start, noAccess, downloading(Double), indexing(Int, Int), ready, failed(String) }
    @Published var stage: Stage = .start
    @Published var results: [AlbumResult] = []
    @Published var busy = false
    @Published var lastQuery = ""
    @Published var planNote = ""

    let index = PhotoIndex()
    let judge = Judge()
    var embedder: Embedder?
    var history: [String] = []
    var currentPlan: Plan?
    var searchTask: Task<Void, Never>?

    func start() async {
        guard await PhotoLibrary.requestAccess() else { stage = .noAccess; return }
        do {
            embedder = try Embedder()
            try await judge.load { p in Task { @MainActor in self.stage = .downloading(p) } }
            await index.load()
            let assets = PhotoLibrary.allAssets()
            stage = .indexing(0, assets.count)
            await index.build(assets: assets, embedder: embedder!) { d, t in Task { @MainActor in self.stage = .indexing(d, t) } }
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
                for (k, album) in plan.albums.enumerated() {
                    self.results.append(AlbumResult(name: album.name))
                    try await engine.run(album, exhaustive: exhaustive) { r in Task { @MainActor in
                        if k < self.results.count { self.results[k] = r } } }
                }
            } catch { self.planNote = "Could not run this search: \(error)" }
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
        case .downloading(let p): VStack { ProgressView(value: p); Text("One-time download of the on-phone AI (\(Int(p * 100))%). After this, nothing leaves your phone.") }.padding()
        case .indexing(let d, let t): VStack { ProgressView(value: Double(d), total: Double(max(t, 1))); Text("Reading your library once: \(d) of \(t). Keep the app open and plugged in.") }.padding()
        case .failed(let e): Text("Something went wrong: \(e)").padding()
        case .ready: SearchView()
        }
    }
}
