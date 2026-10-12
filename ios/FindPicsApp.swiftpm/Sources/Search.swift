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
    var done = false                        // the first answer is complete (it may still be looking: stillLooking)
    /// After the first answer the search keeps looking, paced by the phone's temperature (RESULTS 44b).
    var stillLooking = false
    /// What the search is doing right now, in plain words ("Pausing to let your phone cool."); shown apart from `note`.
    var status = ""
}

/// The phone reached ProcessInfo.ThermalState.critical while the search kept looking (RESULTS 44b).
struct ThermalStop: Error {}

extension ThermalLevel {
    init(_ s: ProcessInfo.ThermalState) {
        switch s {
        case .nominal: self = .nominal
        case .fair: self = .fair
        case .serious: self = .serious
        case .critical: self = .critical
        @unknown default: self = .serious
        }
    }
}

/// A photo as the judge sees it, handed from the prefetch task to the judge loop (CIImage is immutable once made).
struct JudgeImage: @unchecked Sendable { let image: CIImage }

struct SearchEngine {
    let index: PhotoIndex
    let embedder: Embedder
    let judge: any PhotoJudge
    static let accept = 0.7
    /// The judge the 'Is this a photo of X?' 0.99 cutoff was measured on (FindPicsCore.judgeCutoff, RESULTS 35).
    var strictSubjects: Bool { (judge as? Judge)?.id == Judge.visionJudgeCandidateID || judge is EnsembleJudge }

    /// Calls `update` after every round. `exhaustive`: judge every in-scope photo; else stop after the first rounds
    /// stop finding new matches (the fast answer).
    /// `paced`: once the first answer is complete (FindPicsCore.firstAnswerDone), judge only while the phone is cool
    /// enough (FindPicsCore.thermalPace: pause at .serious, stop at .critical). A fast run with `paced` ends at its first
    /// answer marked stillLooking; the app then calls run again with `continuing:` = that result, which runs exhaustive
    /// mode's rounds (the same seed and permutation, so the same rounds the fast run showed, then more; every answer
    /// already given comes from the judge's cache) and keeps looking (RESULTS 44b; Reza 10-10: no cap on accuracy).
    func run(_ album: Album, exhaustive: Bool, restrictTo: Set<String>? = nil, paced: Bool = false,
             continuing: AlbumResult? = nil, update: @escaping @Sendable (AlbumResult) -> Void) async throws {
        let exhaustive = exhaustive || continuing != nil
        var res = AlbumResult(name: album.name)
        if var c = continuing { c.judged = 0; c.stillLooking = true; c.status = SearchEngine.lookingStatus; res = c }
        if continuing == nil, let person = album.person, !person.isEmpty {
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
            Judge.logMem("NOJUDGE \(album.name): \(scoped.count) in scope, 0 judge calls")
            res.found = scoped.sorted { (entries[$0]?.taken ?? 0) > (entries[$1]?.taken ?? 0) }; res.done = true; update(res); return
        }
        let looks = try (album.looks.isEmpty ? [question!] : album.looks).map { try embedder.vector(of: $0) }
        let avoid = try album.avoid.map { try embedder.vector(of: $0) }
        let (units, unitItem, unitT) = await index.units(ids: scoped)
        // videos in scope indexed from their cover frame only (the frames pass has not reached them: FindPicsCore/LazyVideo.swift)
        let coverOnlyInScope = scoped.filter { entries[$0]?.videoFramesPending == true }.count
        let scores = lookScores(units: units, unitItem: unitItem, nItems: scoped.count, looks: looks, avoid: avoid)
        // M40 / RESULTS 45: SHOW THE LIKELY MATCHES BEFORE SPENDING A SINGLE JUDGE CALL. Ranking by the stored
        // PE-Core vectors costs milliseconds; the judge costs 2-4 s a photo, which is why "food photos" took 149
        // minutes on Reza's phone while Apple Photos answers in under a second from precomputed labels. The
        // cluster measured (746 blind eye labels, 216 searches, leave-one-library-out) that the 20th-best fast
        // score says, before any judge call, whether the vector ranking's first screen is already as right as
        // the judge's: 172/216 correct vs 148/216 for "always judge", and the per-photo threshold is 0.90 right
        // by eye. So: put those photos on screen now, and let the judge carry on behind them.
        let decision = firstScreenDecision(album: album, scores: scores)
        let likely = likelyAtOnce(album: album, scores: scores)
        Judge.logMem("VECTORFIRST \(album.name) source=\(decision.source.rawValue) reason=\(decision.reason) "
                     + "likely=\(likely.count)")
        if !likely.isEmpty {
            res.found = likely.map { scoped[$0] }
            res.note = decision.source == .vectors
                ? "Likely matches, shown at once. Still checking them."
                : "Likely matches, shown at once. Still checking them and the rest."
            update(res)
        }
        var bestT = [String: Double](), bestS = [String: Float]()   // videos: the judge sees the frame that matched best
        for (u, k) in unitItem.enumerated() {
            guard let t = unitT[u] else { continue }
            var sc: Float = 0
            for l in looks { var d: Float = 0; for (a, b) in zip(units[u], l) { d += a * b }; sc += d / Float(looks.count) }
            if sc > (bestS[scoped[k]] ?? -.infinity) { bestS[scoped[k]] = sc; bestT[scoped[k]] = t }
        }
        let order = scores.indices.sorted { scores[$0] > scores[$1] }.map { scoped[$0] }
        // PREFETCH (RESULTS 44, item b): the next photos are read (PhotoKit; for most of Reza's library an iCloud
        // download of the 896 px copy) while the GPU judges the current one; streamRounds says which come next.
        let frameAt = bestT
        let prefetch = Prefetcher<JudgeImage> { id in await SearchEngine.judgeImage(id, frameAt: frameAt[id]) }
        // rounds with an honest bound (FindPicsCore.streamRounds, replay-tested: 0 overclaims). The phone judge is slow
        // (~35 photos/min cool, 12-19 hot, halted by iOS after ~68 min: MAC 10-10 15:50), so rounds are TIME budgets
        // (FindPicsCore.phoneFastParams, RESULTS 43): a new bound every doubling of judge calls from 120, the first after
        // 150 calls instead of after a head of up to 2,000 photos. Head rule unchanged (RESULTS 41): fast mode is done
        // once < 3% of the last 100 head photos are yes (or 2,000), plus one more round by itself if its random check
        // found >= 3; and at the latest after the round that reaches 480 calls (RESULTS 44: ~20-32 min). That is the
        // FIRST ANSWER, not the end: with `paced` the app keeps looking afterwards at the phone's thermal pace (44b).
        var missing = Set<String>()        // photos the judge could not see at full resolution (iCloud, no download now)
        let cut = judgeCutoff(question: question!, strictSubjects: strictSubjects)
        let params = phoneFastParams(accept: cut)
        let fastParams = params
        let exQ = album.excludeQuestion, filterQ = album.filterQuestion
        // M39: where a judged photo's time goes (seconds, this search): waiting for its read vs the model calls
        var readWait = 0.0, modelTime = 0.0, modelCalls = 0
        // THERMAL PACING after the first answer (RESULTS 44b; FindPicsCore/ThermalPace.swift). Checked before every
        // model call; answers already cached never wait.
        var pacer = ThermalPacer(poll: 20)
        var pastFirstAnswer = false, lastLevel: ThermalLevel? = nil
        let started = Date()
        func paceGate() async throws {
            if Task.isCancelled { throw CancellationError() }          // Stop, or a new search: end at this photo
            guard paced && pastFirstAnswer else { return }
            while true {
                let level = ThermalLevel(ProcessInfo.processInfo.thermalState), now = Date().timeIntervalSince(started)
                let step = pacer.step(level, now: now)
                if level != lastLevel {
                    lastLevel = level
                    Judge.logMem("THERMAL \(level) at \(Int(now)) s, \(res.judged) judged, pauses \(pacer.pauses) "
                                 + "(\(Int(pacer.pausedSeconds + pacer.pausedNow(now))) s paused): \(album.name)")
                }
                switch step {
                case .judge:
                    if res.status != SearchEngine.lookingStatus { res.status = SearchEngine.lookingStatus; update(res) }
                    return
                case .stop:
                    throw ThermalStop()
                case .wait(let s):
                    if res.status != SearchEngine.coolingStatus { res.status = SearchEngine.coolingStatus; update(res) }
                    try await Task.sleep(nanoseconds: UInt64(s * 1_000_000_000))     // throws on Stop / a new search
                }
            }
        }
        do {
        try await streamRounds(n: order.count, params: params, seed: album.name.utf8.reduce(UInt64(1469598103934665603)) { ($0 ^ UInt64($1)) &* 1099511628211 },
                               lookahead: SearchEngine.lookahead, upcoming: { pos in
            // the photo about to be judged + the next two; photos answered before (a re-run) need no read
            var ids = [String]()
            for q in pos {
                if await judge.cached(order[q] + "|" + question!) == nil { ids.append(order[q]) }
            }
            await prefetch.want(ids)
        }, judge: { pos in
            var out = [Double]()
            for q in pos {
                let id = order[q]
                // ONE read per photo for all its questions (it used to be re-read for each follow-up), and the same
                // image object for each, which is what lets the judge reuse its image prefix (RESULTS 44, item a)
                var img: CIImage?? = .none
                func photo() async -> CIImage? {
                    if case .some(let i) = img { return i }
                    let t0 = Date()
                    let i = await prefetch.take(id)?.image; img = .some(i)
                    readWait += Date().timeIntervalSince(t0); return i
                }
                func ask(_ q: String, keepImage: Bool) async throws -> Double {
                    if let c = await judge.cached(id + "|" + q) { return c }
                    try await paceGate()
                    guard let im = await photo() else { missing.insert(id); return 0 }   // never judged on a smaller copy
                    let t0 = Date()
                    let v = try await judge.pYes(im, question: q, keepImage: keepImage)
                    modelTime += Date().timeIntervalSince(t0); modelCalls += 1
                    await judge.remember(id + "|" + q, v); return v
                }
                var pr = try await ask(question!, keepImage: exQ != nil || filterQ != nil)
                if pr >= cut, let ex = exQ, try await ask(ex, keepImage: filterQ != nil) >= SearchEngine.accept { pr = 0 }
                if pr >= cut, let fq = filterQ, try await ask(fq, keepImage: false) < SearchEngine.accept { pr = 0 }
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
                    // (During the first answer only; after it, paceGate pauses instead.)
                    if !pastFirstAnswer {
                        switch ProcessInfo.processInfo.thermalState {
                        case .serious: res.status = "Your phone is warm, so this is running slower than usual."
                        case .critical: res.status = "Your phone is too warm to keep searching quickly. It will speed up once it cools down."
                        default: if res.status.hasPrefix("Your phone is") { res.status = "" }
                        }
                    }
                    update(res)
                }
            }
            return out
        }, onRound: { r in
            let roundFound = r.found.map { order[$0] }
            if continuing != nil {
                // replaying the rounds already shown: keep every match on screen (all are judge answers) while the
                // replay catches up, instead of flashing back to round 1's list
                let inRound = Set(roundFound)
                res.found = roundFound + res.found.filter { !inRound.contains($0) }
            } else { res.found = roundFound }
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
            if !pastFirstAnswer && firstAnswerDone(r, params: fastParams) {
                pastFirstAnswer = true
                Judge.logMem("SEARCH first answer complete after \(r.judged) judged, \(Int(Date().timeIntervalSince(started))) s: \(album.name)")
            }
            res.done = r.last || !more || continuing != nil
            // a paced fast run that ends at its first answer is continued by the app (AppModel.keepLooking)
            res.stillLooking = !r.last && (continuing != nil || (paced && !exhaustive && !more))
            if res.stillLooking && (res.status.isEmpty || res.status.hasPrefix("Your phone is")) {
                res.status = SearchEngine.lookingStatus
            }
            if r.last { res.stillLooking = false; res.status = "" }
            // M37: the phone timing of a fast search (first match / first bound / done) is read from these lines
            Judge.logMem("SEARCH round \(r.k) judged \(r.judged) of \(order.count) found \(r.found.count) "
                         + "bound +\(Int(c.missedUpper.rounded(.up))) of \(c.nTail) unchecked, done \(res.done): \(album.name)")
            let (h, m, d) = (await prefetch.hits, await prefetch.misses, await prefetch.dropped)
            Judge.logMem("PREFETCH served \(h) read ahead, \(m) read on the spot, \(d) read ahead for nothing; waited "
                         + String(format: "%.1f s for reads, %.1f s in %ld model calls (%.2f s each)", readWait, modelTime,
                                  modelCalls, modelTime / Double(max(modelCalls, 1))) + ": \(album.name)")
            update(res)
            return more && !Task.isCancelled
        })
        } catch is ThermalStop {
            res.stillLooking = false
            res.status = "Stopped looking: your phone got too hot. These are the matches found so far; searching again "
                       + "goes on from here (photos already checked are not checked again while find pics stays open)."
            Judge.logMem("THERMAL critical: stopped looking after \(res.judged) judged, pauses \(pacer.pauses) "
                         + "(\(Int(pacer.pausedSeconds)) s paused): \(album.name)")
        } catch is CancellationError {
            res.stillLooking = false
            if res.done { res.status = "Stopped looking. These are the matches found so far." }
        }
        res.done = true
        if !res.stillLooking && res.status == SearchEngine.lookingStatus { res.status = "" }
        update(res)
    }

    static let lookingStatus = "First answer above. Still looking, at a pace that keeps your phone cool."
    static let coolingStatus = "Pausing to let your phone cool; it goes on by itself."

    /// Photos read ahead of the judge (RESULTS 44); launch argument -noPrefetch = 0, the old one-at-a-time reads (A/B, M39).
    static let lookahead = ProcessInfo.processInfo.arguments.contains("-noPrefetch") ? 0 : 2

    /// The photo as the judge sees it: the video frame that matched best, else the full-resolution 896 px photo (never
    /// a smaller copy: nil when it cannot be read now). Runs ahead of the judge in the prefetch task.
    static func judgeImage(_ id: String, frameAt t: Double?) async -> JudgeImage? {
        var img: CIImage? = nil
        if let t { img = await VideoFrames.frame(id, at: t, side: 896) }
        if img == nil { img = await PhotoLibrary.ciImage(id, side: 896) }
        return img.map(JudgeImage.init)
    }
}
