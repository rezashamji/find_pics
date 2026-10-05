import SwiftUI

struct SearchView: View {
    @EnvironmentObject var model: AppModel
    @State var text = ""
    @State var showing: String?
    @State var saved = ""
    @State var opened = Set<String>()     // expanded "+N similar" stacks

    var body: some View {
        NavigationStack {
            ScrollView {
                if !model.planNote.isEmpty { Text(model.planNote).font(.footnote).foregroundStyle(.secondary).padding(.horizontal) }
                ForEach(model.results) { r in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("\(r.name) — \(r.found.count)").font(.headline)
                            Spacer()
                            if r.done && !r.found.isEmpty {
                                Button("Save as album") {
                                    Task { try? await PhotoLibrary.saveAlbum(named: r.name, ids: r.found); saved = "Saved '\(r.name)'" }
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
            .toolbar { NavigationLink("Self-check") { SelfCheckView(embedder: model.embedder, faces: model.faceEngine) } }
            .searchable(text: $text, prompt: "e.g. me looking heavier vs me looking fit")
            .onSubmit(of: .search) {
                let follow = !model.results.isEmpty && looksLikeFollowUp(text)
                model.search(text, followUp: follow); text = ""
            }
            .overlay { if model.busy && model.results.isEmpty { ProgressView("Understanding your request…") } }
            .sheet(item: Binding(get: { showing.map { Shown(id: $0) } }, set: { showing = $0?.id })) { s in FullPhoto(id: s.id) }
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
