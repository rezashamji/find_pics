// One album search on the phone, streamed: fast stage (image vectors, all photos, milliseconds) -> the judge on the
// best-ranked photos first, in rounds; results update after each round. Exhaustive = keep going until every in-scope
// photo has been judged. Same yes cut as the server (P(yes) >= 0.7).
import CoreImage
@preconcurrency import FindPicsCore
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
    /// The judge the 'Is this a photo of X?' 0.99 cutoff was measured on (FindPicsCore.judgeCutoff, RESULTS 35).
    var strictSubjects: Bool { (judge as? Judge)?.id == Judge.visionJudgeCandidateID || judge is EnsembleJudge }

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
        // "checked 0 of 0" with a full index told us nothing (Reza, 10-10 12:15). When the scope is empty, say
        // WHICH step emptied it: an index with no entries is a different bug from a filter that rejected
        // everything, and a screenshot cannot tell them apart.
        if scoped.isEmpty {
            res.note = "Nothing to search: \(entries.count) items indexed, \(items.count) matched to the library, "
                     + "\(mask.filter { $0 }.count) left after the date/media filter."
            res.done = true
            update(res)
            return
        }
        let question = album.judgeQuestion
        if question == nil {                          // no visual condition: everything in scope (dates / media / clock)
            res.found = scoped.sorted { (entries[$0]?.taken ?? 0) > (entries[$1]?.taken ?? 0) }; res.done = true; update(res); return
        }
        let looks = try (album.looks.isEmpty ? [question!] : album.looks).map { try embedder.vector(of: $0) }
        let avoid = try album.avoid.map { try embedder.vector(of: $0) }
        let (units, unitItem, unitT) = await index.units(ids: scoped)
        // videos in scope indexed from their cover frame only (the frames pass has not reached them: FindPicsCore/LazyVideo.swift)
        let coverOnlyInScope = scoped.filter { entries[$0]?.videoFramesPending == true }.count
        let scores = lookScores(units: units, unitItem: unitItem, nItems: scoped.count, looks: looks, avoid: avoid)
        var bestT = [String: Double](), bestS = [String: Float]()   // videos: the judge sees the frame that matched best
        for (u, k) in unitItem.enumerated() {
            guard let t = unitT[u] else { continue }
            var sc: Float = 0
            for l in looks { var d: Float = 0; for (a, b) in zip(units[u], l) { d += a * b }; sc += d / Float(looks.count) }
            if sc > (bestS[scoped[k]] ?? -.infinity) { bestS[scoped[k]] = sc; bestT[scoped[k]] = t }
        }
        let order = scores.indices.sorted { scores[$0] > scores[$1] }.map { scoped[$0] }
        // rounds with an honest bound (FindPicsCore.streamRounds, replay-tested: 0 overclaims). The phone judge is slow
        // (~0.67 photos/s sustained, MAC 10-10 13:56), so rounds are TIME budgets (FindPicsCore.phoneFastParams,
        // RESULTS 43): a new bound every doubling of judge calls from 120 (~3 min), the first after 150 calls instead
        // of after a head of up to 2,000 photos. Head rule unchanged (RESULTS 41): fast mode is done once < 3% of the
        // last 100 head photos are yes (or 2,000), plus one more round by itself if its random check found >= 3.
        var missing = Set<String>()        // photos the judge could not see at full resolution (iCloud, no download now)
        let cut = judgeCutoff(question: question!, strictSubjects: strictSubjects)
        let params = phoneFastParams(accept: cut)
        let fastParams = params
        try await streamRounds(n: order.count, params: params, seed: album.name.utf8.reduce(UInt64(1469598103934665603)) { ($0 ^ UInt64($1)) &* 1099511628211 }, judge: { pos in
            var out = [Double]()
            for q in pos {
                let id = order[q]
                func ask(_ q: String) async throws -> Double {
                    if let c = await judge.cached(id + "|" + q) { return c }
                    let frameImg: CIImage? = bestT[id] != nil ? await VideoFrames.frame(id, at: bestT[id]!, side: 896) : nil
                    var photoImg = frameImg                  // (no `await` inside `??`: its right side is a sync autoclosure)
                    if photoImg == nil { photoImg = await PhotoLibrary.ciImage(id, side: 896) }
                    guard let img = photoImg else { missing.insert(id); return 0 }   // never judged on a smaller copy
                    let v = try await judge.pYes(img, question: q); await judge.remember(id + "|" + q, v); return v
                }
                var pr = try await ask(question!)
                if pr >= cut, let ex = album.excludeQuestion, try await ask(ex) >= SearchEngine.accept { pr = 0 }
                if pr >= cut, let fq = album.filterQuestion, try await ask(fq) < SearchEngine.accept { pr = 0 }
                out.append(pr)
                res.judged += 1
                // show each match the moment the judge finds it: on the phone (~0.67 photos/s, MAC 10-10 13:56) a round can
                // run for tens of minutes, and results used to appear only at the round's end. onRound still replaces the
                // list with the round's own (same matches, ranked), so nothing found here is lost or shown twice.
                if pr >= cut, !res.found.contains(id) {
                    if res.found.isEmpty { Judge.logMem("SEARCH first match after \(res.judged) judged: \(album.name)") }
                    res.found.append(id); update(res)
                }
                else if res.judged % 25 == 0 {
                    // SAY WHEN THE PHONE IS THROTTLING, instead of silently crawling. MEASURED 10-10: judging
                    // 896 px photos heats an iPhone 18 Pro from ~35/min to ~12-15/min within 15 minutes, and
                    // after ~68 min iOS posted "Charging On Hold" and locked the phone, which suspends the app.
                    // A search that quietly gets 3x slower looks broken; iOS tells us why, so pass it on.
                    switch ProcessInfo.processInfo.thermalState {
                    case .serious:
                        res.note = "Your phone is warm, so this is running slower than usual."
                    case .critical:
                        res.note = "Your phone is too warm to keep searching quickly. It will speed up once it cools down."
                    default:
                        if res.note.hasPrefix("Your phone is") { res.note = "" }
                    }
                    update(res)
                }
            }
            return out
        }, onRound: { r in
            res.found = r.found.map { order[$0] }
            let c = r.certificate
            res.note = completenessNote(c, inScope: order.count)   // plain words (FindPicsCore; Reza 10-07)
            if !missing.isEmpty {
                res.note += " \(missing.count) photo(s) could not be checked: their originals are in iCloud and could not be downloaded now."
            }
            if let n = coverFrameOnlyNote(media: album.media, pendingInScope: coverOnlyInScope,
                                          foundVideos: res.found.filter { entries[$0]?.isVideo == true }.count) {
                res.note += " " + n
            }
            let more = exhaustive || fastModeWantsMore(r, params: fastParams)
            res.done = r.last || !more
            // M37: the phone timing of a fast search (first match / first bound / done) is read from these lines
            Judge.logMem("SEARCH round \(r.k) judged \(r.judged) of \(order.count) found \(r.found.count) "
                         + "bound +\(Int(c.missedUpper.rounded(.up))) of \(c.nTail) unchecked, done \(res.done): \(album.name)")
            update(res)
            return more && !Task.isCancelled
        })
        res.done = true
        update(res)
    }
}
