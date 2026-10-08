@preconcurrency import FindPicsCore
import PhotosUI
import SwiftUI

struct SearchView: View {
    @EnvironmentObject var model: AppModel
    @State var text = ""
    @State var showing: String?
    @State var saved = ""
    @State var opened = Set<String>()     // expanded "+N similar" stacks
    @State var subjectOpen = false              // "find a specific pet / thing / place" from example photos
    @State var subjectPicks: [PhotosPickerItem] = []
    @State var subjectName = ""
    @State var subjectKind = ""

    var body: some View {
        NavigationStack {
            ScrollView {
                if let p = model.indexProgress {
                    Text(p.faces ? faceUpgradeLine(checked: p.done, total: p.total)
                                   + " photos with faces read at full size (Wi-Fi; people searches use a photo once its faces are read)."
                         : p.downloading ? "Reading the rest from iCloud (\(max(0, p.total - p.done).formatted()) left; Wi-Fi; nothing is uploaded). Everything already read is searchable now; the rest continues while you use find pics, or while the phone charges (don't swipe find pics away)."
                                       : "Adding new photos to the search (\(max(0, p.total - p.done).formatted()) left).")
                        .font(.caption).foregroundStyle(.secondary).padding(.horizontal)
                    // a bar + measured rate / time left: a slow pass must not look like a stuck app (Reza 10-08). The
                    // bar and headline count the WHOLE library, so they never go down across launches (MAC 08:47)
                    if !p.faces, model.libraryCount > 0 {
                        let s = searchableSoFar(libraryCount: model.libraryCount, passTotal: p.total, passDone: p.done)
                        Text("\(s.formatted()) of \(model.libraryCount.formatted()) photos and videos searchable")
                            .font(.caption.bold()).padding(.horizontal)
                        ProgressView(value: Double(s), total: Double(model.libraryCount)).padding(.horizontal)
                    } else {
                        ProgressView(value: Double(min(p.done, p.total)), total: Double(max(p.total, 1))).padding(.horizontal)
                    }
                    Text(model.progressETA ?? "Working… (estimating time left)")
                        .font(.caption2).foregroundStyle(.secondary).padding(.horizontal)
                }
                // the frames pass, secondary to the line above: every video is already searchable by its cover frame
                if let v = model.videoFrames, let line = videoFramesLine(sampled: v.sampled, videos: v.videos) {
                    Text(line).font(.caption2).foregroundStyle(.secondary).padding(.horizontal)
                }
                if !model.faceReindexNote.isEmpty {
                    Text(model.faceReindexNote).font(.caption).foregroundStyle(.secondary).padding(.horizontal)
                }
                if !model.indexStatus.isEmpty {
                    Text(model.indexStatus).font(.caption).foregroundStyle(.orange).padding(.horizontal)
                }
                if !model.engineNote.isEmpty {
                    Text(model.engineNote).font(.caption).foregroundStyle(.orange).padding(.horizontal)
                }
                if !model.planNote.isEmpty { Text(model.planNote).font(.footnote).foregroundStyle(.secondary).padding(.horizontal) }
                ForEach(model.results) { r in
                    AlbumCard(r: r, groups: model.bursts[r.id] ?? [], opened: $opened, showing: $showing, saved: $saved)
                }
                if !model.busy, !model.results.isEmpty, model.results.contains(where: { $0.judged < $0.inScope }) {
                    Button("Look at everything (slower, more complete)") { model.rerun(exhaustive: true) }.padding()
                }
                if !saved.isEmpty { Text(saved).font(.footnote).padding() }
                Text("Your photos never leave this phone.").font(.caption2).foregroundStyle(.secondary).padding()
            }
            .navigationTitle("find pics")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Menu("Model: \(model.engine)") {   // side-by-side test: same searches, different judge/planner
                        Button("Qwen (downloaded)") { model.engine = "qwen" }
                        Button("Qwen3-VL photo judge") { model.engine = "qwen3vl" }
                        Button("Two-model vote (Qwen3-VL + Qwen3.5)") { model.engine = "vote" }
                        Button("Apple, 1-10 rating") { model.engine = "apple-rating" }
                        Button("Apple, 0-100 rating") { model.engine = "apple-rating100" }
                        Button("Apple, yes/no") { model.engine = "apple-yesno" }
                    }.font(.caption)
                }
                ToolbarItem(placement: .bottomBar) {
                    Button("Find a specific pet or thing…") { subjectOpen = true }.font(.footnote)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink("Self-check") { SelfCheckView(embedder: model.embedder, faces: model.faceEngine) }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink("About") { AboutView() }
                }
            }
            .searchable(text: $text, prompt: "e.g. me looking heavier vs me looking fit")
            .onSubmit(of: .search) {
                let follow = !model.results.isEmpty && looksLikeFollowUp(text)
                model.search(text, followUp: follow); text = ""
            }
            .overlay { if model.busy && model.results.isEmpty { ProgressView("Understanding your request…") } }
            .sheet(item: Binding(get: { showing.map { Shown(id: $0) } }, set: { showing = $0?.id })) { s in FullPhoto(id: s.id) }
            .sheet(isPresented: $subjectOpen) {
                NavigationStack {
                    Form {
                        PhotosPicker("Pick 1-3 photos of it", selection: $subjectPicks, maxSelectionCount: 3, matching: .images,
                                     photoLibrary: .shared())
                        Text("\(subjectPicks.count) photo(s) picked").font(.caption)
                        TextField("Its name (e.g. Max)", text: $subjectName)
                        TextField("What it is (e.g. dog, bike, house)", text: $subjectKind)
                        Button("Find it") {
                            let ids = subjectPicks.compactMap { $0.itemIdentifier }
                            subjectOpen = false
                            model.searchSubject(ids: ids, name: subjectName.isEmpty ? "it" : subjectName,
                                                kind: subjectKind.isEmpty ? "thing" : subjectKind)
                        }.disabled(subjectPicks.isEmpty)
                    }.navigationTitle("A specific pet or thing")
                }
            }
            .sheet(isPresented: $model.askWhichFace) {
                FacePicker(person: model.askingFor, groups: model.faceGroupsShown, suggested: model.faceSuggestion,
                           others: model.faceOthers, note: model.faceNote,
                           photos: { g in await model.people.photos(of: g) },
                           choose: { g in model.chooseFace(g) },
                           addPhotos: { ids in model.addFacePhotos(ids) })
            }
            .sheet(isPresented: $model.askSubject) {
                if let ask = model.pendingSubject {
                    SubjectAskSheet(ask: ask) { ids in model.provideSubjectPhotos(ids) }
                }
            }
        }
    }

    /// "only the ones outdoors", "also videos", "drop the blurry ones": edits the current search.
    func looksLikeFollowUp(_ t: String) -> Bool {
        t.lowercased().range(of: #"^(only|also|just|but|and|without|no |not |drop|remove|exclude|actually|make it|now)\b"#, options: .regularExpression) != nil
    }
}

struct Shown: Identifiable { let id: String }

/// "Show me Max: pick 1-3 clear photos of Max." A named pet / thing (FindPicsCore.namedSubjects); the picked photos are
/// remembered on this phone and the same search runs again as a subject search.
struct SubjectAskSheet: View {
    let ask: SubjectAsk
    let use: ([String]) -> Void
    @State var picks: [PhotosPickerItem] = []
    var body: some View {
        NavigationStack {
            Form {
                Text(ask.prompt).font(.headline)
                Text("find pics then looks for this \(ask.kind) in your library, side by side with these photos, on this phone.")
                    .font(.footnote).foregroundStyle(.secondary)
                PhotosPicker("Pick 1-3 photos", selection: $picks, maxSelectionCount: 3, matching: .images, photoLibrary: .shared())
                Text("\(picks.count) photo(s) picked").font(.caption)
                Button("Find \(ask.display)") { use(picks.compactMap { $0.itemIdentifier }) }.disabled(picks.isEmpty)
            }.navigationTitle(ask.display)
        }
    }
}

struct Thumb: View {
    let id: String
    @State var img: UIImage?
    var body: some View {
        Color.gray.opacity(0.15).aspectRatio(1, contentMode: .fit)
            .overlay { if let i = img { Image(uiImage: i).resizable().scaledToFill() } }
            .clipped().cornerRadius(6)
            .task { img = await PhotoLibrary.image(id, side: 300) }
    }
}

struct FullPhoto: View {
    let id: String
    @State var img: UIImage?
    var body: some View {
        Group { if let i = img { Image(uiImage: i).resizable().scaledToFit() } else { ProgressView() } }
            .task { img = await PhotoLibrary.image(id, side: 1600, network: true) }   // iCloud original: download to show it
    }
}


/// One album: header, progress, note, and the photo grid with "+N similar" stacks (split out of SearchView so the
/// compiler type-checks small pieces).
struct AlbumCard: View {
    let r: AlbumResult
    let groups: [Int]                 // burst group of each found photo (empty = no stacks yet)
    @Binding var opened: Set<String>
    @Binding var showing: String?
    @Binding var saved: String

    /// Photos to show: the first photo of every stack, plus all photos of the stacks the person opened.
    struct Tile: Identifiable { let k: Int; let id: String; let n: Int }

    var visible: [Tile] {
        let g = groups, stacked = g.count == r.found.count
        var out: [Tile] = []
        for (k, id) in r.found.enumerated() {
            if !stacked { out.append(Tile(k: k, id: id, n: 1)); continue }
            let first = !g[..<k].contains(g[k])
            if first || opened.contains(key(k)) { out.append(Tile(k: k, id: id, n: first ? g.filter { $0 == g[k] }.count : 1)) }
        }
        return out
    }

    func key(_ k: Int) -> String { r.id.uuidString + "\(groups[k])" }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            header
            Text(r.done ? "Checked \(r.judged) of \(r.inScope) photos" : "Searching… checked \(r.judged) of \(r.inScope)")
                .font(.caption).foregroundStyle(.secondary)
            if !r.note.isEmpty { Text(r.note).font(.caption).foregroundStyle(.secondary) }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 100), spacing: 4)], spacing: 4) {
                ForEach(visible) { v in tile(v) }
            }
        }.padding(.horizontal)
    }

    var header: some View {
        HStack {
            Text("\(r.name) — \(r.found.count)").font(.headline)
            Spacer()
            if r.done && !r.found.isEmpty {
                Button("Save as album") { save() }.font(.footnote)
            }
        }
    }

    func tile(_ v: Tile) -> some View {
        Thumb(id: v.id).onTapGesture { showing = v.id }
            .overlay(alignment: .bottomLeading) {
                if v.n > 1 {
                    Button(opened.contains(key(v.k)) ? "hide" : "+\(v.n - 1) similar") {
                        if opened.contains(key(v.k)) { opened.remove(key(v.k)) } else { opened.insert(key(v.k)) }
                    }.font(.caption2).padding(4).background(.black.opacity(0.6)).foregroundStyle(.white).cornerRadius(6).padding(4)
                }
            }
    }

    func save() {
        let name = r.name, ids = r.found
        Task { @MainActor in
            if PhotoLibrary.isLimited {   // limited access: albums cannot be created
                saved = "To save albums, allow Full Access: Settings > find pics > Photos."
                return
            }
            do { try await PhotoLibrary.saveAlbum(named: name, ids: ids); saved = "Saved '\(name)'" }
            catch { saved = "Could not save '\(name)': \(error.localizedDescription)" }
        }
    }
}
