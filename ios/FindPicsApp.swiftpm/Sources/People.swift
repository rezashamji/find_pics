// "Who is X?": the faces the library shows most often (FindPicsCore.faceGroups, same as the server's people sheet).
// The app SUGGESTS one (FindPicsCore.suggestGroup: for "me" the group most present in front-camera selfies, else the
// largest group not named yet) and asks "Is this you?"; Yes / another group / "Add a photo of them" (1-3 photos, faces
// detected on the phone). Saved as named fingerprints in the app's own container only, with the face model that made
// them and WHERE those faces are, so a face-model change re-derives them from the same faces (or asks again; never a
// silent mix of two models' vectors). iOS gives apps no access to Apple's People names, so nothing here reads Apple's
// People album.
// [F8] A group is an automatic cluster and can mix people: nothing is confirmed from a row of thumbnails any more. A row
// opens GroupFacesView (every face of the group, least alike first); the person taps the wrong ones out and confirms
// the rest. Excluded faces are saved as that person's negatives (FindPicsCore/PeopleReview.swift: never re-added by
// expandRefs, never counted). "Pick 1-3 photos" is at the top, as the most accurate way. PeopleView (toolbar "People")
// shows who is set, their faces, and lets the person remove faces, pick again or forget them.
@preconcurrency import FindPicsCore
import Foundation
import PhotosUI
import SwiftUI

struct NamedPerson: Codable {
    let name: String
    var refs: [[Float]]
    /// Face model that made `refs` (FaceProfile id); nil = saved before this was recorded, i.e. buffalo_l.
    var model: String? = nil
    /// Where the reference faces are (photo / frame / box): lets a face-model change re-derive `refs` from the same faces.
    var sources: [FaceSource]? = nil
    /// The face model changed and too few of these faces were found again: ask "Is this you?" once more.
    var reask: Bool? = nil
    /// Why `reask` (the picker's note); nil = the face model changed.
    var reaskNote: String? = nil
    /// Faces the person said are NOT them (excluded in the picker or removed later), same face model as `refs`.
    var negatives: [[Float]]? = nil
    /// Where each negative is (pairs with `negatives`), so a face-model change re-derives them too.
    var negativeSources: [FaceSource]? = nil

    var pick: PersonPick {
        PersonPick(refs: refs, sources: sources ?? [], negatives: negatives ?? [], negativeSources: negativeSources ?? [])
    }
    mutating func apply(_ p: PersonPick) {
        refs = p.refs; sources = p.sources; negatives = p.negatives; negativeSources = p.negativeSources
        model = FaceProfile.shipped.id; reask = nil; reaskNote = nil
    }

    /// Usable now: refs exist and come from the shipped face model.
    var current: Bool { !refs.isEmpty && shippedModel }
    /// Vectors (refs and negatives) are the shipped face model's; true also while asked again (refs emptied, negatives kept).
    var shippedModel: Bool { (model ?? legacyFaceModel) == FaceProfile.shipped.id }
    /// The earlier pick whose "not this person" faces carry over to a new pick (nil: another face model's vectors).
    var carryOver: PersonPick? { shippedModel ? pick : nil }
}

actor PeopleStore {
    private(set) var named: [String: NamedPerson] = [:]
    private(set) var groups: [FaceGroup] = []
    private var faceEmb = StoredRows()                 // the index store's mapped face rows (not copied)
    private var faceItem: [String] = []
    private var faceBox: [StoredFace] = []        // box + frame time of each face (for FaceSource)
    private var faceSide: [Double] = []           // long side of the read each face came from (FaceSource.side)
    private var itemIds: [String] = []            // FaceGroup.items numbers -> photo ids
    /// Bumped whenever the groups are rebuilt: a picker opened on older groups must not name faces by stale numbers.
    private(set) var groupsGen = 0
    private let file: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("people.json")

    private var loaded = false
    /// False when people.json exists but cannot be read yet (locked since restart): then nothing is saved over it.
    @discardableResult func load() -> Bool {
        if loaded { return true }
        if FileManager.default.fileExists(atPath: file.path) {
            guard let d = try? Data(contentsOf: file), let n = try? JSONDecoder().decode([NamedPerson].self, from: d) else { return false }
            named = Dictionary(n.map { ($0.name.lowercased(), $0) }, uniquingKeysWith: { a, _ in a })
        }
        loaded = true
        return true
    }
    func save() { guard loaded else { return }; if let d = try? JSONEncoder().encode(Array(named.values)) { try? d.write(to: file, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication]) } }

    func refreshGroups(index: PhotoIndex) async {
        let f = await index.allFaces()
        let sides = await index.faceSidesByItem()
        faceEmb = f.emb; faceItem = f.item; faceBox = f.box
        faceSide = f.item.map { sides[$0] ?? 0 }
        itemIds = Array(Set(f.item)).sorted()
        let itemNumber = Dictionary(uniqueKeysWithValues: itemIds.enumerated().map { ($1, $0) })
        // reliable vectors first: faces read at faceReadSide; all faces only if those form no group (FindPicsCore)
        groups = faceGroupsPreferChecked(faces: f.emb, faceItem: f.item.map { itemNumber[$0]! }, facePx: f.px, det: f.det,
                                         checked: faceSide.map { $0 >= faceReadSide }, accept: SearchEngine.faceProfile.group,
                                         faceTaken: f.taken)
        groupsGen += 1
    }

    /// The group is one of the current groups (its face numbers point into the current face arrays).
    private func isCurrent(_ g: FaceGroup, gen: Int) -> Bool { gen == groupsGen && groups.contains(g) }

    private func source(_ k: Int) -> FaceSource {
        FaceSource(id: faceItem[k], t: faceBox[k].frameT, box: faceBox[k].box, imageW: faceBox[k].imageW, imageH: faceBox[k].imageH,
                   side: faceSide[k])
    }

    /// Every face of a group for review: where each is, the order to show them (least alike first), and the ones this
    /// person already said are not them. nil: the groups were rebuilt since the picker opened (reopen it).
    func review(_ g: FaceGroup, gen: Int, person: String, owner: String) -> GroupReview? {
        guard isCurrent(g, gen: gen) else { return nil }
        let vecs = g.faces.map { faceEmb[$0] }
        let neg = named[key(person, owner: owner)]?.carryOver?.negatives ?? []
        return GroupReview(sources: g.faces.map { source($0) }, order: leastAlikeFirst(vecs), excluded: alreadyExcluded(vectors: vecs, negatives: neg))
    }

    /// Confirm a group minus the faces the person excluded (positions in g.faces): only the kept faces become references,
    /// the excluded ones negatives (FindPicsCore.pickFromGroup). `name` is what the request said ("me" -> the owner's
    /// name, the key refs(for:) looks up). False: groups rebuilt meanwhile, or nothing kept; nothing saved.
    @discardableResult func name(group g: FaceGroup, excluding: Set<Int>, gen: Int, as name: String) -> Bool {
        guard isCurrent(g, gen: gen) else { return false }
        let earlier = named[name.lowercased()]?.carryOver
        guard let p = pickFromGroup(vectors: g.faces.map { faceEmb[$0] }, sources: g.faces.map { source($0) }, excluded: excluding,
                                    earlier: earlier) else { return false }
        var q = NamedPerson(name: name, refs: [])
        q.apply(p)
        named[name.lowercased()] = q
        save()
        return true
    }

    /// "Add a photo of them": references from faces found in the photos the person picked (`sources`: where each is).
    /// Earlier "not them" faces still hold.
    func name(refs: [[Float]], sources: [FaceSource], as name: String) {
        let earlier = named[name.lowercased()]?.carryOver
        guard let p = pickFromPhotos(refs: refs, sources: sources, earlier: earlier) else { return }
        var q = NamedPerson(name: name, refs: [])
        q.apply(p)
        named[name.lowercased()] = q; save()
    }

    // MARK: People screen

    /// Who is set, for the People screen (sorted by name).
    func list() -> [PersonSummary] {
        named.map { k, p in
            let editable = p.current && (p.sources?.count ?? -1) == p.refs.count
            return PersonSummary(key: k, name: p.name, faces: p.sources ?? [], order: editable ? leastAlikeFirst(p.refs) : [],
                                 refCount: p.refs.count, excludedCount: p.negatives?.count ?? 0, editable: editable,
                                 current: p.current, reask: p.reask == true)
        }.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    /// Remove some of a saved person's faces (positions in their references): they become negatives. False: nothing
    /// would be left, or the faces cannot be told apart (old pick): nothing changed.
    @discardableResult func removeFaces(key k: String, positions: Set<Int>) -> Bool {
        guard var p = named[k], p.current, let q = removeFromPick(p.pick, remove: positions) else { return false }
        p.apply(q); named[k] = p; save()
        return true
    }

    /// Forget a person (app data only: people.json in the app's own container; no photo is touched).
    func forget(key k: String) { named[k] = nil; save() }

    /// The person's references from the shipped face model: "me" maps to whoever was named as the owner. nil when not
    /// named yet, or named with another face model and not re-derived (yet): old vectors are never handed out.
    func refs(for person: String, owner: String) -> [[Float]]? {
        guard let p = named[key(person, owner: owner)], p.current else { return nil }
        return p.refs
    }

    /// Faces the person said are not them (same face model as their refs; [] when none or not current).
    func negatives(for person: String, owner: String) -> [[Float]] {
        guard let p = named[key(person, owner: owner)], p.current else { return [] }
        return p.negatives ?? []
    }

    /// Someone saved with another face model, not re-derived yet (re-derivation runs after the re-embedding pass).
    func awaitingRederive() -> Bool { named.values.contains { !$0.current && $0.reask != true } }

    /// Named before, but the face model changed and their faces were not found again: the picker says why it asks.
    func needsReask(_ person: String, owner: String) -> Bool { named[key(person, owner: owner)]?.reask == true }

    /// The picker's note for a person who must be confirmed again (nil: not asked again).
    func reaskNote(_ person: String, owner: String) -> String? {
        guard let p = named[key(person, owner: owner)], p.reask == true else { return nil }
        return p.reaskNote ?? "find pics now uses a new face-recognition model and could not find your earlier pick again. Please confirm once more."
    }

    // MARK: face-model change

    /// Before any face is re-embedded: remember WHERE each saved person's reference faces are. Their vectors are copies
    /// of index faces of the old model (naming a group copies them), so they are found exactly. Idempotent.
    func recordSources(index: PhotoIndex) async {
        let todo = named.filter { !$0.value.current && $0.value.sources == nil && !$0.value.refs.isEmpty }
        guard !todo.isEmpty else { return }
        var byModel: [String: LibraryFaces] = [:]
        for (k, p) in todo {
            let m = p.model ?? legacyFaceModel
            if byModel[m] == nil { byModel[m] = await index.allFaces(model: m) }
            let f = byModel[m]!
            let at = locateRefs(p.refs, in: f.emb)
            var q = p
            q.sources = at.compactMap { $0 }.map { i in
                FaceSource(id: f.item[i], t: f.box[i].frameT, box: f.box[i].box, imageW: f.box[i].imageW, imageH: f.box[i].imageH)
            }
            named[k] = q
        }
        save()
    }

    /// After a re-embedding pass: each saved person of another face model gets the NEW vectors of the same faces
    /// (found in the index by photo, frame time and box; a picked photo not in the index is read and its faces
    /// detected again). Kept when most faces were found (FindPicsCore.keepAfterRederive), otherwise marked to be
    /// asked again. Returns the names that must be asked again.
    @discardableResult func rederive(index: PhotoIndex, faceEngine: FaceEngine?) async -> [String] {
        let todo = named.filter { !$0.value.current && $0.value.reask != true }
        guard !todo.isEmpty else { return [] }
        let entries = await index.entries
        var reasked = [String]()
        for (k, p) in todo {
            let src = p.sources ?? []
            let (refs, kept) = await vectorsAgain(src, entries: entries, index: index, faceEngine: faceEngine)
            // negatives the same way; one not found again is dropped (it only ever removes, never asks again)
            let (neg, negKept) = await vectorsAgain(p.negativeSources ?? [], entries: entries, index: index, faceEngine: faceEngine)
            var q = p
            q.model = FaceProfile.shipped.id
            q.negatives = neg; q.negativeSources = negKept
            if keepAfterRederive(found: refs.count, total: max(src.count, p.refs.count)) {
                q.refs = refs; q.sources = kept; q.reask = nil
            } else {
                q.refs = []; q.sources = []; q.reask = true; reasked.append(p.name)
            }
            named[k] = q
        }
        save()
        return reasked
    }

    /// The shipped face model's vectors of the faces at `src` (found in the index by photo, frame time and box; a picked
    /// photo not in the index is read and its faces detected again). Faces not found are left out.
    private func vectorsAgain(_ src: [FaceSource], entries: [String: IndexEntry], index: PhotoIndex,
                              faceEngine: FaceEngine?) async -> ([[Float]], [FaceSource]) {
        var refs = [[Float]](), kept = [FaceSource]()
        for s in src {
            if let e = entries[s.id], e.hasFaces {
                guard e.facesCurrent else { continue }               // still old vectors (photo not re-read yet)
                let fs = await index.detectedFaces(s.id, frameT: s.t)
                if let i = matchSourceFace(s, candidates: fs.map { (box: $0.box, imageW: $0.imageW, imageH: $0.imageH) }) {
                    refs.append(fs[i].embedding); kept.append(s.withSide(s.t == nil ? e.faceSideEffective : faceReadSide))
                }
            } else if s.t == nil, let fe = faceEngine,
                      case .full(let ui) = await PhotoLibrary.read(s.id, side: 1280, purpose: .indexForeground),
                      let cg = ui.cgImage, let fs = try? fe.faces(in: cg),
                      let i = matchSourceFace(s, candidates: fs.map { (box: $0.box, imageW: $0.imageW, imageH: $0.imageH) }) {
                refs.append(fs[i].embedding); kept.append(s)            // a picked photo that is not in the index
            }
        }
        return (refs, kept)
    }

    // MARK: face upgrade

    /// After a face-upgrade run: saved people whose reference faces are in photos it re-read at faceReadSide get the
    /// new vectors of the same faces (same box, FindPicsCore.refsAfterUpgrade); too few found again -> asked again with
    /// refsUpgradeReaskNote, like after a face-model change. Returns the names that must be asked again.
    @discardableResult func refreshAfterUpgrade(index: PhotoIndex, upgraded: Set<String>) async -> [String] {
        guard !upgraded.isEmpty, load() else { return [] }
        let todo = named.filter { p in
            p.value.current && p.value.reask != true
                && ((p.value.sources ?? []) + (p.value.negativeSources ?? [])).contains { $0.t == nil && upgraded.contains($0.id) }
        }
        guard !todo.isEmpty else { return [] }
        let now = await index.photoFaces(Set(todo.values.flatMap { (($0.sources ?? []) + ($0.negativeSources ?? [])).map(\.id) }))
        var reasked = [String]()
        for (k, p) in todo {
            guard let cur = named[k], cur.refs == p.refs else { continue }    // renamed meanwhile: leave the new pick
            var q = p
            // negatives: the same faces' new vectors; a lost one is dropped (negatives never make anyone be asked again)
            let ns = p.negativeSources ?? []
            if let n = refsAfterUpgrade(refs: p.negatives ?? [], sources: ns, now: ns.map { now[$0.id] }) {
                q.negatives = n.refs; q.negativeSources = n.sources
            }
            let src = p.sources ?? []
            if let r = refsAfterUpgrade(refs: p.refs, sources: src, now: src.map { now[$0.id] }) {
                if r.keep { q.refs = r.refs; q.sources = r.sources }
                else { q.refs = []; q.sources = []; q.reask = true; q.reaskNote = refsUpgradeReaskNote; reasked.append(p.name) }
            }
            named[k] = q
        }
        save()
        return reasked
    }

    func key(_ person: String, owner: String) -> String { isOwnerWord(person) ? owner.lowercased() : person.lowercased() }

    /// Which group to suggest for `person` (nil: every group is named already) and the others in listing order.
    func suggestion(for person: String, owner: String, index: PhotoIndex) async -> (suggested: Int?, others: [Int]) {
        let entries = await index.entries
        let cams = itemIds.map { entries[$0]?.camera ?? "" }
        let k = key(person, owner: owner)
        let others = named.filter { $0.key != k && $0.value.current }.map { $0.value.refs }   // groups that are someone else already
        let assigned = assignedGroups(groups: groups.map { $0.faces.map { faceEmb[$0] } }, named: others)
        let s = suggestGroup(groupItems: groups.map(\.items), itemCamera: cams, forOwner: isOwnerWord(person) || k == owner.lowercased(),
                             assigned: assigned)
        return (s, otherGroupsOrder(count: groups.count, suggested: s, assigned: assigned))
    }

    /// Faces of OTHER frequent people (the other groups), for the "closer to someone else -> not this person" rule.
    func otherGroups(excluding refs: [[Float]]) -> [[[Float]]] { groups.map { $0.faces.map { faceEmb[$0] } } }
}


/// Every face of a group, for review before confirming (positions are into FaceGroup.faces).
struct GroupReview: Sendable {
    let sources: [FaceSource]
    /// Positions to show, least like the rest of the group first (FindPicsCore.leastAlikeFirst).
    let order: [Int]
    /// Positions this person already said are not them (shown excluded).
    let excluded: Set<Int>
}

/// One saved person, for the People screen.
struct PersonSummary: Identifiable, Sendable {
    let key: String
    let name: String
    let faces: [FaceSource]
    /// Positions in `faces` to show, least like the rest first (an intruder near the top).
    let order: [Int]
    let refCount: Int
    let excludedCount: Int
    /// Faces pair up with the references, so single faces can be removed (otherwise: pick again or forget only).
    let editable: Bool
    let current: Bool
    let reask: Bool
    var id: String { key }
    var shownName: String { isOwnerWord(name) ? "Me" : name }
    /// In a sentence: "you" / "Mom".
    var you: String { isOwnerWord(name) ? "you" : name }
    /// After "not" / "Forget": "me" / "Mom".
    var me: String { isOwnerWord(name) ? "me" : name }
}

extension PeopleStore {
    /// The faces most like the rest of the group (its row in the picker): the end of the review order.
    func typicalFaces(_ g: FaceGroup, gen: Int, n: Int = 6) -> [FaceSource] {
        guard let r = review(g, gen: gen, person: "", owner: "") else { return [] }
        return r.order.suffix(n).reversed().map { r.sources[$0] }
    }
}

/// "Who is X?". Top: pick 1-3 photos of the person (the most accurate way). Below: the suggested face group and the
/// others. A group is an automatic guess that can mix people, so a row is never confirmed as it stands: tapping it
/// opens every face of the group (GroupFacesView), where wrong faces are tapped out before confirming.
struct FacePicker: View {
    let person: String
    let groups: [FaceGroup]
    let suggested: Int?
    let others: [Int]
    let note: String
    let typical: (FaceGroup) async -> [FaceSource]
    let review: (FaceGroup) async -> GroupReview?
    let choose: (FaceGroup, Set<Int>) -> Void
    let addPhotos: ([String]) -> Void
    @State var picks: [PhotosPickerItem] = []

    var owner: Bool { isOwnerWord(person) }
    var chooseTitle: String { owner ? "Choose photos of you" : "Choose photos of \(person)" }
    var groupsHeader: String {
        suggested == nil ? (owner ? "Which group is you?" : "Which group is \(person)?") : "Not that one? Other groups"
    }
    var title: String { owner ? "Which face is you?" : "Which face is \(person)?" }
    var body: some View {
        List {
            Section(footer: Text("The face groups below are automatic guesses and can mix up people who look alike.")) {
                Text(pickPhotosPrompt(person)).font(.headline)
                PhotosPicker(chooseTitle, selection: $picks, maxSelectionCount: 3,
                             matching: .images, photoLibrary: .shared())
                if !picks.isEmpty {
                    Button("Use \(picks.count) photo(s)") { addPhotos(picks.compactMap { $0.itemIdentifier }) }
                        .buttonStyle(.borderedProminent)
                }
                if !note.isEmpty { Text(note).font(.footnote).foregroundStyle(.orange) }
            }
            if let s = suggested, s < groups.count {
                Section(header: Text("Or check the group find pics suggests")) {
                    Text(whoQuestion(person)).font(.headline)
                    link(groups[s])
                }
            }
            Section(header: Text(groupsHeader),
                    footer: Text("find pics remembers this on this phone only. Change it any time under People.")) {
                ForEach(others.filter { $0 < groups.count }, id: \.self) { g in link(groups[g]) }
            }
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
    }

    func link(_ g: FaceGroup) -> some View {
        NavigationLink {
            GroupFacesView(person: person, group: g, note: note, review: review, choose: choose)
        } label: {
            GroupRow(group: g, typical: typical)
        }
    }
}

/// A group in the picker: its most typical faces and how big it is.
struct GroupRow: View {
    let group: FaceGroup
    let typical: (FaceGroup) async -> [FaceSource]
    @State var faces: [FaceSource] = []
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                ForEach(Array(faces.enumerated()), id: \.offset) { _, f in FaceCrop(source: f).frame(width: 44, height: 44) }
            }
            Text("\(group.faces.count) faces in \(group.items.count) photos. Tap to check every face.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .task { faces = await typical(group) }
    }
}

/// Every face of one group, least like the rest first (an intruder shows up near the top). Tap a face to leave it out;
/// the button confirms only the faces kept, and the left-out ones are remembered as "not this person".
struct GroupFacesView: View {
    let person: String
    let group: FaceGroup
    let note: String
    let review: (FaceGroup) async -> GroupReview?
    let choose: (FaceGroup, Set<Int>) -> Void
    @State var data: GroupReview?
    @State var stale = false
    @State var excluded: Set<Int> = []

    var who: String { isOwnerWord(person) ? "you" : person }
    var body: some View {
        ScrollView {
            if let d = data {
                Text("Tap every face that is not \(who) to leave it out. Least alike first.")
                    .font(.footnote).foregroundStyle(.secondary).padding(.horizontal)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 76), spacing: 6)], spacing: 6) {
                    ForEach(d.order, id: \.self) { k in
                        FaceCrop(source: d.sources[k])
                            .opacity(excluded.contains(k) ? 0.35 : 1)
                            .overlay(alignment: .topTrailing) {
                                if excluded.contains(k) {
                                    Image(systemName: "xmark.circle.fill").foregroundStyle(.white, .red).padding(3)
                                }
                            }
                            .contentShape(Rectangle())
                            .onTapGesture { if excluded.contains(k) { excluded.remove(k) } else { excluded.insert(k) } }
                            .accessibilityLabel(excluded.contains(k) ? "Left out" : "Kept")
                            .accessibilityAddTraits(.isButton)
                    }
                }.padding(.horizontal)
            } else if stale {
                Text("The face groups changed while new photos were being read. Go back and open the list again.").padding()
            } else {
                ProgressView().padding()
            }
        }
        .safeAreaInset(edge: .bottom) {
            if let d = data {
                VStack(spacing: 6) {
                    if !note.isEmpty { Text(note).font(.footnote).foregroundStyle(.orange) }
                    Button(confirmGroupLabel(person, kept: d.sources.count - excluded.count, total: d.sources.count)) {
                        choose(group, excluded)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(excluded.count >= d.sources.count)
                }
                .padding(8).frame(maxWidth: .infinity).background(.bar)
            }
        }
        .navigationTitle(whoQuestion(person))
        .navigationBarTitleDisplayMode(.inline)
        .task {
            if let d = await review(group) { data = d; excluded = d.excluded } else { stale = true }
        }
    }
}

/// One face, cropped from its photo (or video frame) with the stored box; loaded when it scrolls into view.
struct FaceCrop: View {
    let source: FaceSource
    @State var img: UIImage?
    var body: some View {
        Color.gray.opacity(0.15).aspectRatio(1, contentMode: .fit)
            .overlay { if let i = img { Image(uiImage: i).resizable().scaledToFill() } }
            .clipped().cornerRadius(6)
            .task(id: source) { img = await FaceCrops.load(source) }
    }
}

enum FaceCrops {
    /// The face's square (FindPicsCore.faceCropRect) from a read just big enough for it (faceCropReadSide). The box is
    /// in the analysed image's pixels and is scaled to the image read now, as SearchEngine.redBoxCrop does.
    static func load(_ s: FaceSource) async -> UIImage? {
        let side = CGFloat(faceCropReadSide(box: s.box, imageW: s.imageW, imageH: s.imageH))
        let r = faceCropRect(box: s.box, imageW: s.imageW, imageH: s.imageH)
        func crop(_ cg: CGImage) -> CGImage? {
            let W = Double(cg.width), H = Double(cg.height)
            return cg.cropping(to: CGRect(x: r.x1 * W, y: r.y1 * H, width: (r.x2 - r.x1) * W, height: (r.y2 - r.y1) * H).integral)
        }
        if let t = s.t {                          // a video: the frame the face was seen in
            guard let ci = await VideoFrames.frame(s.id, at: t, side: side),
                  let cg = CIContext().createCGImage(ci, from: ci.extent), let c = crop(cg) else { return nil }
            return UIImage(cgImage: c)
        }
        guard let ui = await PhotoLibrary.image(s.id, side: side), let cg = ui.cgImage, let c = crop(cg) else { return nil }
        return UIImage(cgImage: c, scale: 1, orientation: ui.imageOrientation)
    }
}

/// The "Who is X?" picker as a sheet (from a search, or from the People screen).
struct FacePickerSheet: View {
    @ObservedObject var model: AppModel
    var body: some View {
        NavigationStack {
            FacePicker(person: model.askingFor, groups: model.faceGroupsShown, suggested: model.faceSuggestion,
                       others: model.faceOthers, note: model.faceNote,
                       typical: { g in await model.typicalFaces(g) },
                       review: { g in await model.reviewFaces(g) },
                       choose: { g, ex in model.chooseFace(g, excluding: ex) },
                       addPhotos: { ids in model.addFacePhotos(ids) })
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Not now") { model.closeFacePicker() } } }
        }
    }
}

/// People: who find pics knows ("me", "Mom"), their faces, and a way to fix a wrong pick.
struct PeopleView: View {
    @ObservedObject var model: AppModel
    @State var people: [PersonSummary] = []
    @State var loaded = false
    var body: some View {
        List {
            Section(footer: Text("Saved on this phone only. People searches use these faces.")) {
                if loaded && people.isEmpty { Text("Nobody yet. Search for \"me ...\" or a name, or set yourself up now.") }
                ForEach(people) { p in
                    NavigationLink { PersonEditor(model: model, key: p.key) } label: {
                        HStack {
                            // the most typical face (end of the least-alike-first order)
                            if let f = p.order.last.map({ p.faces[$0] }) ?? p.faces.first { FaceCrop(source: f).frame(width: 44, height: 44) }
                            VStack(alignment: .leading) {
                                Text(p.shownName)
                                Text(subtitle(p)).font(.caption).foregroundStyle(p.reask ? Color.orange : Color.secondary)
                            }
                        }
                    }
                }
            }
            if !people.contains(where: { isOwnerWord($0.name) || $0.key == model.owner.lowercased() }) {
                Button("Set up \"me\"") { Task { await model.openFacePicker(for: "me", fromPeople: true) } }
            }
        }
        .navigationTitle("People")
        .task(id: model.peopleVersion) { people = await model.people.list(); loaded = true }
    }
}

extension PeopleView {
    func subtitle(_ p: PersonSummary) -> String {
        if p.reask { return "Needs to be confirmed again" }
        return "\(p.refCount) face(s)" + (p.excludedCount > 0 ? ", \(p.excludedCount) marked not \(p.me)" : "")
    }
}

/// One saved person: every face find pics uses for them. Tap faces to remove them (they become "not this person"),
/// pick again from the start, or forget the person.
struct PersonEditor: View {
    @ObservedObject var model: AppModel
    let key: String
    @State var person: PersonSummary?
    @State var marked: Set<Int> = []
    @State var confirmForget = false
    @Environment(\.dismiss) var dismiss

    var body: some View {
        ScrollView {
            if let p = person {
                VStack(alignment: .leading, spacing: 8) {
                    if p.reask {
                        Text("find pics could not find this pick again after an update. Pick again.").font(.footnote).foregroundStyle(.orange)
                    } else if p.editable {
                        Text("These are the \(p.refCount) faces used to find \(p.you), least alike first. Tap any that are wrong, then remove them.")
                            .font(.footnote).foregroundStyle(.secondary)
                    } else if !p.current {
                        Text("Waiting for the new face model to finish; this person can be edited then.").font(.footnote).foregroundStyle(.secondary)
                    } else {
                        Text("This pick was saved by an older version and its faces cannot be shown one by one. Pick again to check them.")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                    if p.editable {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 76), spacing: 6)], spacing: 6) {
                            ForEach(p.order, id: \.self) { k in
                                FaceCrop(source: p.faces[k])
                                    .opacity(marked.contains(k) ? 0.35 : 1)
                                    .overlay(alignment: .topTrailing) {
                                        if marked.contains(k) { Image(systemName: "xmark.circle.fill").foregroundStyle(.white, .red).padding(3) }
                                    }
                                    .contentShape(Rectangle())
                                    .onTapGesture { if marked.contains(k) { marked.remove(k) } else { marked.insert(k) } }
                                    .accessibilityAddTraits(.isButton)
                            }
                        }
                    }
                }.padding(.horizontal)
            } else {
                ProgressView().padding()
            }
        }
        .safeAreaInset(edge: .bottom) {
            if let p = person {
                VStack(spacing: 8) {
                    if p.editable && !marked.isEmpty {
                        Button("Remove \(marked.count) face(s)") {
                            let m = marked
                            Task { if await model.removeFaces(key: key, positions: m) { marked = [] } }
                        }
                        .buttonStyle(.borderedProminent).tint(.red)
                        .disabled(marked.count >= p.refCount)
                    }
                    HStack {
                        Button("Pick again") { Task { await model.openFacePicker(for: p.name, fromPeople: true) } }
                        Spacer()
                        Button("Forget \(p.me)", role: .destructive) { confirmForget = true }
                    }
                }
                .padding(8).background(.bar)
            }
        }
        .navigationTitle(person?.shownName ?? "")
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog("find pics will forget these faces. Your photos are not touched.", isPresented: $confirmForget,
                            titleVisibility: .visible) {
            Button("Forget", role: .destructive) { Task { await model.forgetPerson(key: key); dismiss() } }
        }
        .task(id: model.peopleVersion) {
            person = await model.people.list().first { $0.key == key }
            marked = []
        }
    }
}
