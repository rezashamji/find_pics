// One album search on the phone, streamed: fast stage (image vectors, all photos, milliseconds) -> the judge on the
// best-ranked photos first, in rounds; results update after each round. Exhaustive = keep going until every in-scope
// photo has been judged. Same yes cut as the server (P(yes) >= 0.7).
import CoreImage
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
    let judge: any PhotoJudge
    static let accept = 0.7

    /// Calls `update` after every round. `exhaustive`: judge every in-scope photo; else stop after the first rounds
    /// stop finding new matches (the fast answer).
    func run(_ album: Album, exhaustive: Bool, restrictTo: Set<String>? = nil, update: @escaping @Sendable (AlbumResult) -> Void) async throws {
        var res = AlbumResult(name: album.name)
        if let person = album.person, !person.isEmpty {
            res.note = "Searching for a person by face is not on the phone yet (\(person)); showing the rest of the search."
        }
        let entries = await index.entries
        let ids = Array(entries.keys).filter { restrictTo?.contains($0) ?? true }
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
        let (units, unitItem, unitT) = await index.units(ids: scoped)
        let scores = lookScores(units: units, unitItem: unitItem, nItems: scoped.count, looks: looks, avoid: avoid)
        var bestT = [String: Double](), bestS = [String: Float]()   // videos: the judge sees the frame that matched best
        for (u, k) in unitItem.enumerated() {
            guard let t = unitT[u] else { continue }
            var sc: Float = 0
            for l in looks { var d: Float = 0; for (a, b) in zip(units[u], l) { d += a * b }; sc += d / Float(looks.count) }
            if sc > (bestS[scoped[k]] ?? -.infinity) { bestS[scoped[k]] = sc; bestT[scoped[k]] = t }
        }
        let order = scores.indices.sorted { scores[$0] > scores[$1] }.map { scoped[$0] }
        // rounds with an honest bound (FindPicsCore.streamRounds, replay-tested: 0 overclaims in 170 rounds). The phone
        // judge is slow (~1 photo/s), so the first round is smaller than the server's.
        var params = StreamParams(); params.headSize = 150; params.headChunk = 50; params.headMax = 1500; params.tailBudget = 150
        try await streamRounds(n: order.count, params: params, seed: album.name.utf8.reduce(UInt64(1469598103934665603)) { ($0 ^ UInt64($1)) &* 1099511628211 }, judge: { pos in
            var out = [Double]()
            for q in pos {
                let id = order[q]
                func ask(_ q: String) async throws -> Double {
                    if let c = await judge.cached(id + "|" + q) { return c }
                    let frameImg: CIImage? = bestT[id] != nil ? await VideoFrames.frame(id, at: bestT[id]!, side: 896) : nil
                    guard let img = frameImg ?? (await PhotoLibrary.ciImage(id, side: 896)) else { return 0 }
                    let v = try await judge.pYes(img, question: q); await judge.remember(id + "|" + q, v); return v
                }
                var pr = try await ask(question!)
                if pr >= SearchEngine.accept, let ex = album.excludeQuestion, try await ask(ex) >= SearchEngine.accept { pr = 0 }
                if pr >= SearchEngine.accept, let fq = album.filterQuestion, try await ask(fq) < SearchEngine.accept { pr = 0 }
                out.append(pr)
                res.judged += 1
                if res.judged % 25 == 0 { update(res) }
            }
            return out
        }, onRound: { r in
            res.found = r.found.map { order[$0] }
            let c = r.certificate
            res.note = c.nTail == 0 ? "The judge checked every photo in scope."
                : "At least \(Int((c.recallLower * 100).rounded(.down)))% of matches found (95% confidence, relative to the AI judge); about \(Int(c.missedUpper)) could still be hiding among \(c.nTail) unchecked photos."
            res.done = r.last || !exhaustive
            update(res)
            return exhaustive && !Task.isCancelled
        })
        res.done = true
        update(res)
    }
}
