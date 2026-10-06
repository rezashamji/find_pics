// "Which face is you": the faces the library shows most often (FindPicsCore.faceGroups, same as the server's people
// sheet); the person taps theirs once and it is saved as named fingerprints, in the app's own container only.
@preconcurrency import FindPicsCore
import Foundation
import SwiftUI

struct NamedPerson: Codable { let name: String; let refs: [[Float]] }

actor PeopleStore {
    private(set) var named: [String: NamedPerson] = [:]
    private(set) var groups: [FaceGroup] = []
    private var faceEmb: [[Float]] = []
    private var faceItem: [String] = []
    private let file: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("people.json")

    func load() {
        if let d = try? Data(contentsOf: file), let n = try? JSONDecoder().decode([NamedPerson].self, from: d) {
            named = Dictionary(uniqueKeysWithValues: n.map { ($0.name.lowercased(), $0) })
        }
    }
    func save() { if let d = try? JSONEncoder().encode(Array(named.values)) { try? d.write(to: file, options: .completeFileProtection) } }

    func refreshGroups(index: PhotoIndex) async {
        let f = await index.allFaces()
        faceEmb = f.emb; faceItem = f.item
        let itemNumber = Dictionary(uniqueKeysWithValues: Array(Set(f.item)).enumerated().map { ($1, $0) })
        groups = faceGroups(faces: f.emb, faceItem: f.item.map { itemNumber[$0]! }, facePx: f.px, det: f.det)
    }

    /// Photo ids showing a group's faces (for the picker), best first.
    func photos(of g: FaceGroup) -> [String] { Array(Set(g.faces.map { faceItem[$0] })) }

    func name(group g: FaceGroup, as name: String) {
        named[name.lowercased()] = NamedPerson(name: name, refs: g.faces.map { faceEmb[$0] }); save()
    }

    /// The person's references: "me" maps to whoever was named as the owner.
    func refs(for person: String, owner: String) -> [[Float]]? {
        let key = ["me", "i", "myself"].contains(person.lowercased()) ? owner.lowercased() : person.lowercased()
        return named[key]?.refs
    }

    /// Faces of OTHER frequent people (the other groups), for the "closer to someone else -> not this person" rule.
    func otherGroups(excluding refs: [[Float]]) -> [[[Float]]] { groups.map { $0.faces.map { faceEmb[$0] } } }
}

struct FacePicker: View {
    let groups: [FaceGroup]
    let photos: (FaceGroup) async -> [String]
    let choose: (FaceGroup) -> Void
    var body: some View {
        List {
            Text("Which one is you? Tap your row once; find pics remembers it (on this phone only).").font(.footnote)
            ForEach(Array(groups.enumerated()), id: \.offset) { k, g in
                GroupRow(group: g, photos: photos).onTapGesture { choose(g) }
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
