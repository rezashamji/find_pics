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
                if model.unreadable > 0 {
                    Text("\(model.unreadable) photos could not be read on this phone (often originals kept only in iCloud); searches cannot see them yet.")
                        .font(.caption).foregroundStyle(.orange).padding(.horizontal)
                }
                if !model.planNote.isEmpty { Text(model.planNote).font(.footnote).foregroundStyle(.secondary).padding(.horizontal) }
                ForEach(model.results) { r in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("\(r.name) — \(r.found.count)").font(.headline)
                            Spacer()
                            if r.done && !r.found.isEmpty {
                                Button("Save as album") {
                                    Task {
                                        if PhotoLibrary.isLimited {   // limited access: albums cannot be created
                                            saved = "To save albums, allow Full Access: Settings > find pics > Photos."
                                            return
                                        }
                                        do { try await PhotoLibrary.saveAlbum(named: r.name, ids: r.found); saved = "Saved '\(r.name)'" }
                                        catch { saved = "Could not save '\(r.name)': \(error.localizedDescription)" }
                                    }
                                }.font(.footnote)
                            }
                        }
                        Text(r.done ? "Checked \(r.judged) of \(r.inScope) photos" : "Searching… checked \(r.judged) of \(r.inScope)")
                            .font(.caption).foregroundStyle(.secondary)
                        if !r.note.isEmpty { Text(r.note).font(.caption).foregroundStyle(.secondary) }
                        let g = model.bursts[r.id] ?? []
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 100), spacing: 4)], spacing: 4) {
                            ForEach(Array(r.found.enumerated()).filter { k, _ in g.count != r.found.count || opened.contains(r.id.uuidString + "\(g[k])") || !g[..<k].contains(g[k]) }, id: \.element) { k, id in
                                let n = g.count == r.found.count ? g.filter { $0 == g[k] }.count : 1
                                Thumb(id: id).onTapGesture { showing = id }
                                    .overlay(alignment: .bottomLeading) {
                                        if n > 1 && !g[..<k].contains(g[k]) {
                                            Button(opened.contains(r.id.uuidString + "\(g[k])") ? "hide" : "+\(n - 1) similar") {
                                                let key = r.id.uuidString + "\(g[k])"
                                                if opened.contains(key) { opened.remove(key) } else { opened.insert(key) }
                                            }.font(.caption2).padding(4).background(.black.opacity(0.6)).foregroundStyle(.white).cornerRadius(6).padding(4)
                                        }
                                    }
                            }
                        }
                    }.padding(.horizontal)
                }
                if !model.busy, !model.results.isEmpty, model.results.contains(where: { $0.judged < $0.inScope }) {
                    Button("Look at everything (slower, more complete)") { model.search(model.lastQuery, exhaustive: true) }.padding()
                }
                if !saved.isEmpty { Text(saved).font(.footnote).padding() }
                Text("Your photos never leave this phone.").font(.caption2).foregroundStyle(.secondary).padding()
            }
            .navigationTitle("find pics")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Menu("Model: \(model.engine)") {   // side-by-side test: same searches, different judge/planner
                        Button("Qwen (downloaded)") { model.engine = "qwen" }
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
                FacePicker(groups: model.faceGroupsShown, photos: { g in await model.people.photos(of: g) }) { g in
                    Task { await model.people.name(group: g, as: model.owner); model.askWhichFace = false; model.search(model.lastQuery) }
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
            .task { img = await PhotoLibrary.image(id, side: 1600) }
    }
}
