import SwiftUI

struct SearchView: View {
    @EnvironmentObject var model: AppModel
    @State var text = ""
    @State var showing: String?
    @State var saved = ""

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
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 100), spacing: 4)], spacing: 4) {
                            ForEach(r.found, id: \.self) { id in Thumb(id: id).onTapGesture { showing = id } }
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
            .searchable(text: $text, prompt: "e.g. me looking heavier vs me looking fit")
            .onSubmit(of: .search) {
                let follow = !model.results.isEmpty && looksLikeFollowUp(text)
                model.search(text, followUp: follow); text = ""
            }
            .overlay { if model.busy && model.results.isEmpty { ProgressView("Understanding your request…") } }
            .sheet(item: Binding(get: { showing.map { Shown(id: $0) } }, set: { showing = $0?.id })) { s in FullPhoto(id: s.id) }
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
