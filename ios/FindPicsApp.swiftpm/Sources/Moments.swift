// "the week I went to X", "after the cake was cut, before the speeches", "me with Jay": the moment (anchor) is found
// first with its own question, its time window (FindPicsCore.windowRows, identical to the server) limits where the
// album searches; "until" cuts the window at the first photo of the end moment; "with" people must also be in the photo.
// Port of converse._album_stream's anchor / until / with_people steps.
@preconcurrency import FindPicsCore
import Foundation
import os

extension SearchEngine {
    /// Photos the album may search, or nil = everything. Empty set = the moment was not found (say so, search nothing).
    func momentScope(_ album: Album) async throws -> (scope: Set<String>?, note: String) {
        guard let anchor = album.anchor, let aq = anchor.judgeQuestion else { return (nil, "") }
        var a = Album(name: album.name + " (moment)")
        a.looks = anchor.looks; a.judgeQuestion = aq; a.dateFrom = album.dateFrom; a.dateTo = album.dateTo
        a.place = album.place; a.media = album.media
        let foundBox = OSAllocatedUnfairLock<[String]>(initialState: [])   // `update` is @Sendable: no captured var
        try await run(a, exhaustive: false) { r in if r.done { foundBox.withLock { $0 = r.found } } }
        let found = foundBox.withLock { $0 }
        if found.isEmpty {
            return ([], "Could not find the moment this album is anchored to (\"\(aq)\"): no photo passed that question. Describe it differently, or search everywhere.")
        }
        let entries = await index.entries
        let ids = Array(entries.keys)
        let pos = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
        let taken = ids.map { entries[$0]?.taken }, place = ids.map { entries[$0]?.place ?? "" }
        var rows = windowRows(taken: taken, place: place, anchors: found.compactMap { pos[$0] }, window: album.window)
        var note = "Found the moment in \(found.count) photo(s); searching \(rows.count) photo(s) around it."
        if let until = album.until, let uq = until.judgeQuestion {
            var u = Album(name: album.name + " (end)"); u.looks = until.looks; u.judgeQuestion = uq
            u.dateFrom = album.dateFrom; u.dateTo = album.dateTo; u.place = album.place
            let endsBox = OSAllocatedUnfairLock<[String]>(initialState: [])
            try await run(u, exhaustive: false) { r in if r.done { endsBox.withLock { $0 = r.found } } }
            let ends = endsBox.withLock { $0 }
            if let start = found.compactMap({ entries[$0]?.taken }).min(),
               let cut = ends.compactMap({ entries[$0]?.taken }).filter({ $0 > start }).min() {
                rows = rows.filter { (taken[$0] ?? .infinity) < cut }
                note += " Stopped at the first photo of the end moment."
            }
        }
        return (Set(rows.map { ids[$0] }), note)
    }

    /// "me with Jay": photos where each other named person's face also matches.
    func withPeople(_ album: Album, people: PeopleStore, owner: String) async -> (ok: Set<String>?, unknown: [String]) {
        guard !album.withPeople.isEmpty else { return (nil, []) }
        let f = await index.allFaces()
        let ids = Array(Set(f.item)).sorted(); let num = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
        var ok: Set<String>? = nil, unknown = [String]()
        for w in album.withPeople {
            guard let refs = await people.refs(for: w, owner: owner) else { unknown.append(w); continue }
            let (s, _) = itemPersonScores(faces: f.emb, faceItem: f.item.map { num[$0]! }, nItems: ids.count, refs: refs)
            let these = Set(ids.indices.filter { s[$0] >= SearchEngine.personAccept }.map { ids[$0] })
            ok = ok.map { $0.intersection(these) } ?? these
        }
        return (ok, unknown)
    }
}
