// One album search on the phone, streamed: fast stage (image vectors, all photos, milliseconds) -> the judge on the
// best-ranked photos first, in rounds; results update after each round. Exhaustive = keep going until every in-scope
// photo has been judged. Same yes cut as the server (P(yes) >= 0.7).
import FindPicsCore
import Foundation

struct AlbumResult: Identifiable {
    let id = UUID()
    var name: String
    var found: [String] = []                // asset ids, best first
    var judged = 0, inScope = 0
    var note = ""
    var done = false
}

struct SearchEngine {
    let index: PhotoIndex
    let embedder: Embedder
    let judge: Judge
    static let accept = 0.7

    /// Calls `update` after every round. `exhaustive`: judge every in-scope photo; else stop after the first rounds
    /// stop finding new matches (the fast answer).
    func run(_ album: Album, exhaustive: Bool, update: @escaping @Sendable (AlbumResult) -> Void) async throws {
        var res = AlbumResult(name: album.name)
        if let person = album.person, !person.isEmpty {
            res.note = "Searching for a person by face is not on the phone yet (\(person)); showing the rest of the search."
        }
        let entries = await index.entries
        let ids = Array(entries.keys)
        let items = await index.libraryItems(order: ids)
        let mask = scopeMask(items, album)
        let scoped = zip(ids, mask).filter { $0.1 }.map { $0.0 }
        res.inScope = scoped.count
        let question = album.judgeQuestion
        if question == nil {                          // no visual condition: everything in scope (dates / media / clock)
            res.found = scoped.sorted { (entries[$0]?.taken ?? 0) > (entries[$1]?.taken ?? 0) }; res.done = true; update(res); return
        }
        let looks = try (album.looks.isEmpty ? [question!] : album.looks).map { try embedder.vector(of: $0) }
        let avoid = try album.avoid.map { try embedder.vector(of: $0) }
        let units = scoped.map { entries[$0]!.vector }
        let scores = lookScores(units: units, unitItem: Array(0..<units.count), nItems: units.count, looks: looks, avoid: avoid)
        let order = scores.indices.sorted { scores[$0] > scores[$1] }.map { scoped[$0] }
        var pos = 0, chunk = 50, quietRounds = 0
        var hits: [(String, Double)] = []
        while pos < order.count {
            if Task.isCancelled { break }
            let batch = order[pos..<min(pos + chunk, order.count)]
            var newHits = 0
            for id in batch {
                guard let img = await PhotoLibrary.ciImage(id, side: 896) else { continue }
                var p = try await judge.pYes(img, question: question!)
                if p >= SearchEngine.accept, let ex = album.excludeQuestion, try await judge.pYes(img, question: ex) >= SearchEngine.accept { p = 0 }
                if p >= SearchEngine.accept, let fq = album.filterQuestion, try await judge.pYes(img, question: fq) < SearchEngine.accept { p = 0 }
                if p >= SearchEngine.accept { hits.append((id, p)); newHits += 1 }
                res.judged += 1
            }
            pos += batch.count
            res.found = hits.sorted { $0.1 > $1.1 }.map { $0.0 }
            update(res)
            quietRounds = newHits == 0 ? quietRounds + 1 : 0
            if !exhaustive && quietRounds >= 2 { break }      // fast answer: two empty rounds past the matches
            chunk = min(chunk * 2, 400)
        }
        res.done = true
        if !exhaustive && pos < order.count {
            res.note += (res.note.isEmpty ? "" : " ") + "Fast answer: the judge checked the \(res.judged) most likely of \(res.inScope) photos. Tap \"Look at everything\" to check the rest."
        }
        update(res)
    }
}
