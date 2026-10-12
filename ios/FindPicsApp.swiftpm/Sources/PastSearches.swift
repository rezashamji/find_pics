import SwiftUI
import FindPicsCore

/// What a finished search found, kept so it can be reopened without running it again.
///
/// WHY (10-11): a search over 187k photos can take minutes to hours on the phone, and until now its results lived
/// only in memory - closing the app threw them away. That makes the one thing the product most needs impossible in
/// practice: "Check a sample" (M43) asks Reza to hand-check 25 random results, and he is not going to re-run a
/// 40-minute search to do it. It also starts to answer the part of what he wants that is not searching at all:
/// results he can come back to and turn into albums.
struct SavedSearch: Codable, Identifiable, Equatable {
    var id: String { query + "|" + String(at.timeIntervalSince1970) }
    var query: String
    var ids: [String]
    var judged: Int, inScope: Int
    var note: String
    var at: Date

    /// Back into the shape the result card draws.
    var asResult: AlbumResult {
        var r = AlbumResult(name: query)
        r.found = ids; r.judged = judged; r.inScope = inScope; r.note = note; r.done = true
        return r
    }
}

actor PastSearchStore {
    private(set) var saved: [SavedSearch] = []
    /// Newest first, and bounded: ids are cheap (a UUID string each) but 187k of them is not.
    private let keep = 40
    private let file: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("past_searches.json")
    private var loaded = false

    @discardableResult func load() -> Bool {
        if loaded { return true }
        if FileManager.default.fileExists(atPath: file.path) {
            guard let d = try? Data(contentsOf: file),
                  let s = try? JSONDecoder().decode([SavedSearch].self, from: d) else { return false }
            saved = s
        }
        loaded = true
        return true
    }

    func add(_ s: SavedSearch) {
        guard load(), !s.ids.isEmpty else { return }
        // One entry per query text: the newest run of "food photos" replaces the older one, so the list stays
        // readable and an audit always points at the current answer.
        saved.removeAll { $0.query.caseInsensitiveCompare(s.query) == .orderedSame }
        saved.insert(s, at: 0)
        if saved.count > keep { saved = Array(saved.prefix(keep)) }
        if let d = try? JSONEncoder().encode(saved) {
            try? d.write(to: file, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        }
    }
}

struct PastSearchesView: View {
    @ObservedObject var model: AppModel
    @State private var rows: [SavedSearch] = []

    var body: some View {
        List {
            if rows.isEmpty {
                Text("Searches you have run will be kept here, so you can look at what they found - and check a "
                     + "sample of it - without running them again.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            ForEach(rows) { s in
                NavigationLink { SavedSearchView(saved: s, model: model) } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(s.query).font(.body)
                        Text("\(s.ids.count.formatted()) photos · \(s.at.formatted(date: .abbreviated, time: .shortened))")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .navigationTitle("Past searches")
        .task {
            _ = await model.pastSearches.load()
            rows = await model.pastSearches.saved
        }
    }
}

struct SavedSearchView: View {
    let saved: SavedSearch
    @ObservedObject var model: AppModel
    @State private var opened = Set<String>()
    @State private var showing: String?
    @State private var savedNote = ""

    var body: some View {
        ScrollView {
            Text("Found \(saved.ids.count.formatted()) on \(saved.at.formatted(date: .abbreviated, time: .shortened)).")
                .font(.caption).foregroundStyle(.secondary).padding(.horizontal)
            AlbumCard(r: saved.asResult, groups: [], opened: $opened, showing: $showing, saved: $savedNote)
            if !savedNote.isEmpty { Text(savedNote).font(.footnote).padding() }
            Button("Run it again") { model.search(saved.query) }.padding()
        }
        .navigationTitle(saved.query)
        .sheet(item: Binding(get: { showing.map { Shown(id: $0) } }, set: { showing = $0?.id })) { s in FullPhoto(id: s.id) }
    }
}
