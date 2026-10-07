// "Who is X?": the faces the library shows most often (FindPicsCore.faceGroups, same as the server's people sheet).
// The app SUGGESTS one (FindPicsCore.suggestGroup: for "me" the group most present in front-camera selfies, else the
// largest group not named yet) and asks "Is this you?"; Yes / another group / "Add a photo of them" (1-3 photos, faces
// detected on the phone). Saved as named fingerprints in the app's own container only. iOS gives apps no access to
// Apple's People names, so nothing here reads Apple's People album.
@preconcurrency import FindPicsCore
import Foundation
import PhotosUI
import SwiftUI

struct NamedPerson: Codable { let name: String; let refs: [[Float]] }

actor PeopleStore {
    private(set) var named: [String: NamedPerson] = [:]
    private(set) var groups: [FaceGroup] = []
    private var faceEmb: [[Float]] = []
    private var faceItem: [String] = []
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
        faceEmb = f.emb; faceItem = f.item
        itemIds = Array(Set(f.item)).sorted()
        let itemNumber = Dictionary(uniqueKeysWithValues: itemIds.enumerated().map { ($1, $0) })
        groups = faceGroups(faces: f.emb, faceItem: f.item.map { itemNumber[$0]! }, facePx: f.px, det: f.det)
    }

    /// Photo ids showing a group's faces (for the picker), best first.
    func photos(of g: FaceGroup) -> [String] { Array(Set(g.faces.map { faceItem[$0] })) }

    /// `name` is what the request said ("me" -> stored under the owner's name, the key refs(for:) looks up).
    func name(group g: FaceGroup, as name: String) {
        named[name.lowercased()] = NamedPerson(name: name, refs: g.faces.map { faceEmb[$0] }); save()
    }

    /// "Add a photo of them": references from faces found in the photos the person picked.
    func name(refs: [[Float]], as name: String) {
        guard !refs.isEmpty else { return }
        named[name.lowercased()] = NamedPerson(name: name, refs: refs); save()
    }

    /// The person's references: "me" maps to whoever was named as the owner.
    func refs(for person: String, owner: String) -> [[Float]]? {
        named[key(person, owner: owner)]?.refs
    }

    func key(_ person: String, owner: String) -> String { isOwnerWord(person) ? owner.lowercased() : person.lowercased() }

    /// Which group to suggest for `person` (nil: every group is named already) and the others in listing order.
    func suggestion(for person: String, owner: String, index: PhotoIndex) async -> (suggested: Int?, others: [Int]) {
        let entries = await index.entries
        let cams = itemIds.map { entries[$0]?.camera ?? "" }
        let k = key(person, owner: owner)
        let others = named.filter { $0.key != k }.map { $0.value.refs }       // groups that are someone else already
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
