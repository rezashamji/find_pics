// "Who is X?": the faces the library shows most often (FindPicsCore.faceGroups, same as the server's people sheet).
// The app SUGGESTS one (FindPicsCore.suggestGroup: for "me" the group most present in front-camera selfies, else the
// largest group not named yet) and asks "Is this you?"; Yes / another group / "Add a photo of them" (1-3 photos, faces
// detected on the phone). Saved as named fingerprints in the app's own container only, with the face model that made
// them and WHERE those faces are, so a face-model change re-derives them from the same faces (or asks again; never a
// silent mix of two models' vectors). iOS gives apps no access to Apple's People names, so nothing here reads Apple's
// People album.
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

    /// Usable now: refs exist and come from the shipped face model.
    var current: Bool { !refs.isEmpty && (model ?? legacyFaceModel) == FaceProfile.shipped.id }
}

actor PeopleStore {
    private(set) var named: [String: NamedPerson] = [:]
    private(set) var groups: [FaceGroup] = []
    private var faceEmb: [[Float]] = []
    private var faceItem: [String] = []
    private var faceBox: [DetectedFace] = []      // box + frame time of each face (for FaceSource)
    private var faceSide: [Double] = []           // long side of the read each face came from (FaceSource.side)
    private var itemIds: [String] = []            // FaceGroup.items numbers -> photo ids
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
                                         checked: faceSide.map { $0 >= faceReadSide }, accept: SearchEngine.faceProfile.group)
    }

    /// Photo ids showing a group's faces (for the picker), best first.
    func photos(of g: FaceGroup) -> [String] { Array(Set(g.faces.map { faceItem[$0] })) }

    /// `name` is what the request said ("me" -> stored under the owner's name, the key refs(for:) looks up).
    func name(group g: FaceGroup, as name: String) {
        let src = g.faces.map { k in
            FaceSource(id: faceItem[k], t: faceBox[k].frameT, box: faceBox[k].box, imageW: faceBox[k].imageW, imageH: faceBox[k].imageH,
                       side: faceSide[k])
        }
        named[name.lowercased()] = NamedPerson(name: name, refs: g.faces.map { faceEmb[$0] }, model: FaceProfile.shipped.id, sources: src)
        save()
    }

    /// "Add a photo of them": references from faces found in the photos the person picked (`sources`: where each is).
    func name(refs: [[Float]], sources: [FaceSource], as name: String) {
        guard !refs.isEmpty else { return }
        named[name.lowercased()] = NamedPerson(name: name, refs: refs, model: FaceProfile.shipped.id, sources: sources); save()
    }

    /// The person's references from the shipped face model: "me" maps to whoever was named as the owner. nil when not
    /// named yet, or named with another face model and not re-derived (yet): old vectors are never handed out.
    func refs(for person: String, owner: String) -> [[Float]]? {
        guard let p = named[key(person, owner: owner)], p.current else { return nil }
        return p.refs
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
        var byModel: [String: (emb: [[Float]], item: [String], px: [Float], det: [Float], box: [DetectedFace])] = [:]
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
            var refs = [[Float]](), kept = [FaceSource]()
            for s in src {
                if let e = entries[s.id], e.hasFaces {
                    guard e.facesCurrent else { continue }               // still old vectors (photo not re-read yet)
                    let fs: [DetectedFace] = s.t.map { t in e.frames?.first(where: { $0.t == t })?.faces ?? [] } ?? (e.faces ?? [])
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
            var q = p
            q.model = FaceProfile.shipped.id
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

    // MARK: face upgrade

    /// After a face-upgrade run: saved people whose reference faces are in photos it re-read at faceReadSide get the
    /// new vectors of the same faces (same box, FindPicsCore.refsAfterUpgrade); too few found again -> asked again with
    /// refsUpgradeReaskNote, like after a face-model change. Returns the names that must be asked again.
    @discardableResult func refreshAfterUpgrade(index: PhotoIndex, upgraded: Set<String>) async -> [String] {
        guard !upgraded.isEmpty, load() else { return [] }
        let todo = named.filter { p in
            p.value.current && p.value.reask != true && (p.value.sources ?? []).contains { $0.t == nil && upgraded.contains($0.id) }
        }
        guard !todo.isEmpty else { return [] }
        let now = await index.photoFaces(Set(todo.values.flatMap { ($0.sources ?? []).map(\.id) }))
        var reasked = [String]()
        for (k, p) in todo {
            guard let cur = named[k], cur.refs == p.refs else { continue }    // renamed meanwhile: leave the new pick
            let src = p.sources ?? []
            guard let r = refsAfterUpgrade(refs: p.refs, sources: src, now: src.map { now[$0.id] }) else { continue }
            var q = p
            if r.keep { q.refs = r.refs; q.sources = r.sources }
            else { q.refs = []; q.sources = []; q.reask = true; q.reaskNote = refsUpgradeReaskNote; reasked.append(p.name) }
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

/// "Is this you?" with the suggested group, Yes, the other groups, and "Add a photo of them".
struct FacePicker: View {
    let person: String
    let groups: [FaceGroup]
    let suggested: Int?
    let others: [Int]
    let note: String
    let photos: (FaceGroup) async -> [String]
    let choose: (FaceGroup) -> Void
    let addPhotos: ([String]) -> Void
    @State var picks: [PhotosPickerItem] = []

    var owner: Bool { isOwnerWord(person) }
    var body: some View {
        List {
            if let s = suggested, s < groups.count {
                Section {
                    Text(whoQuestion(person)).font(.headline)
                    GroupRow(group: groups[s], photos: photos)
                    Button(owner ? "Yes, that's me" : "Yes, that's \(person)") { choose(groups[s]) }.buttonStyle(.borderedProminent)
                }
            }
            Section(header: Text(suggested == nil ? (owner ? "Which one is you?" : "Which one is \(person)?") : "No? Tap the right row")) {
                ForEach(others.filter { $0 < groups.count }, id: \.self) { g in
                    GroupRow(group: groups[g], photos: photos).contentShape(Rectangle()).onTapGesture { choose(groups[g]) }
                }
            }
            Section(footer: Text("find pics remembers this on this phone only.")) {
                PhotosPicker(owner ? "Add a photo of you" : "Add a photo of them", selection: $picks, maxSelectionCount: 3,
                             matching: .images, photoLibrary: .shared())
                if !picks.isEmpty {
                    Button("Use \(picks.count) photo(s)") { addPhotos(picks.compactMap { $0.itemIdentifier }) }
                }
                if !note.isEmpty { Text(note).font(.footnote).foregroundStyle(.orange) }
            }
        }
    }
}

struct GroupRow: View {
    let group: FaceGroup
    let photos: (FaceGroup) async -> [String]
    @State var ids: [String] = []
    var body: some View {
        ScrollView(.horizontal) { HStack { ForEach(ids.prefix(6), id: \.self) { Thumb(id: $0).frame(width: 70, height: 70) } } }
            .task { ids = await photos(group) }
    }
}
