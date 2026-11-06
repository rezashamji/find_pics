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
    /// THE OFFER, not a promise: the careful check's floor in judge calls and seconds (CascadeSession.judgeFloor).
    /// Set when the judge-free answer is up and the judge has NOT run. The screen turns it into a button that says
    /// what it will cost, because on a whole-library question it is hours (M95(a)).
    /// M100(3)(b): the plan has a judge-free answer and NO judge-kind leaf, so the cascade would return exactly
    /// what the quick check already returned. There is no careful check to sell, and the screen must say that in
    /// words rather than just showing no button.
    /// The first screen, ranked from the typed words before the planner has said anything (M100(2)). It is
    /// REPLACED IN PLACE by the planned album, never shown beside it - and the engine writes results[k] by album
    /// index, so a leftover card at index 0 would also shift every album's results by one.
    var provisional = false
    var noCarefulCheck = false
    var checkCalls: Int? = nil
    var checkSeconds: Double? = nil
}

/// The phone reached ProcessInfo.ThermalState.critical while the search kept looking (RESULTS 44b).
struct ThermalStop: Error {}
/// The dry pass makes no judge call by construction; this is thrown if one is ever asked for, so a wiring
/// mistake fails loudly instead of quietly judging photos during what the screen calls the instant answer.
struct MilliOnly: Error {}

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
    /// M53 / RESULTS 48: a small learned head blended with cosine decides the ORDER the judge checks photos in - it
    /// never decides a match. Same matches for 27% fewer judge calls on audit4 and 42% fewer on ev16, which on this
    /// phone is the difference between a 10-minute first answer and a 6-minute one, and it is free: 8.9 MB of
    /// weights and one pass over vectors we already store. (The head ALONE was tried as a judge replacement and
    /// rejected - M48-ANSWER: on a query it never trained on it is no better than cosine.)
    /// The cluster's selfie weights (M50), bundled with the app; fitted on public Commons photos only.
    static let selfieDetector: SelfieDetector? = {
        guard let u = Bundle.module.url(forResource: "selfie_head", withExtension: "json", subdirectory: "Models"),
              let d = try? Data(contentsOf: u),
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
              let w = j["w"] as? [Double], let b = j["b"] as? Double else { return nil }
        return SelfieDetector(w: w.map(Float.init), b: Float(b))
    }()

    static let learnedHead: HeadRanker? = Bundle.module
        .url(forResource: "learned_head", withExtension: "fphead", subdirectory: "Models")
        .flatMap { try? HeadRanker(contentsOf: $0) }
    /// The judge the per-subject cutoffs were measured on (FindPicsCore.judgeCutoff; RESULTS 35, re-tuned in
    /// RESULTS 47: objects 0.7, beach 0.999 under its new wording, sunset/other subjects 0.999, food 0.9999).
    var strictSubjects: Bool { (judge as? Judge)?.id == Judge.visionJudgeCandidateID || judge is EnsembleJudge }

    /// Calls `update` after every round. `exhaustive`: judge every in-scope photo; else stop after the first rounds
    /// stop finding new matches (the fast answer).
    /// `paced`: once the first answer is complete (FindPicsCore.firstAnswerDone), judge only while the phone is cool
    /// enough (FindPicsCore.thermalPace: pause at .serious, stop at .critical). A fast run with `paced` ends at its first
    /// answer marked stillLooking; the app then calls run again with `continuing:` = that result, which runs exhaustive
    /// mode's rounds (the same seed and permutation, so the same rounds the fast run showed, then more; every answer
    /// already given comes from the judge's cache) and keeps looking (RESULTS 44b; Reza 10-10: no cap on accuracy).
    /// SECONDS PER JUDGE CALL, measured on Reza's iPhone 18 Pro (MAC 11-05: 2.3-2.5 s at 896 px, slower when warm).
    /// Used only to price the careful check before it starts; the search itself never predicts, it reports.
    static let secondsPerJudgeCall = 2.4

    func run(_ album: Album, exhaustive: Bool, restrictTo: Set<String>? = nil, paced: Bool = false,
             careful: Bool = false, verifyShownOnly: Bool = false,
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
        // A BARE "selfies": the camera tag for the few photos that carry it, plus the detector for the rest (M50).
        // Zero judge calls. The threshold is calibrated on THIS library's own tagged photos, not shipped: on the
        // cluster's Commons data logit 0.5 had a 0.84% false-positive rate, and on Reza's phone the same cut has
        // 23% (MEASURED 10-12: AUC 0.963, 581 front / 1,671 back, 123 ms) - the phone stores 448 px renditions
        // where the fit used 1,280 px thumbnails. A shipped threshold would quietly fill his selfies with other
        // people's photos; a per-library one cannot.
        if question == nil, album.camera == "front", let det = SearchEngine.selfieDetector,
           let t = await index.selfieThreshold(det) {
            let tagged = Set(scoped)
            let cand = ids.filter { !tagged.contains($0) && entries[$0]?.isVideo != true
                                 && entries[$0]?.isScreenshot != true && entries[$0]?.camera != "back" }
            let extra = await index.selfieLikely(det, ids: cand, threshold: t)
            if !extra.isEmpty {
                res.found = scoped.sorted { (entries[$0]?.taken ?? 0) > (entries[$1]?.taken ?? 0) } + extra
                res.inScope = res.found.count
                res.note = "\(scoped.count.formatted()) your phone recorded with its front camera, plus "
                         + "\(extra.count.formatted()) more that look like selfies. Nothing was opened by the AI, "
                         + "so this was instant."
                res.done = true
                Judge.logMem("SELFIEANSWER tagged=\(scoped.count) likely=\(extra.count) t=\(String(format: "%.3f", t))")
                update(res)
                return
            }
        }
        if question == nil {                          // no visual condition: everything in scope (dates / media / clock)
            // A CAMERA CENSUS, because two counts disagreed: the selfies search said 267 in scope at 22:14 and the
            // geometry export counted 579 front-camera photos at 22:52. One of them is wrong about a user-visible
            // answer, so the log now says exactly what the filter saw.
            let camFront = items.filter { $0.camera == "front" }.count
            let camBack = items.filter { $0.camera == "back" }.count
            Judge.logMem("NOJUDGE \(album.name): \(scoped.count) in scope, 0 judge calls; items=\(items.count) "
                         + "front=\(camFront) back=\(camBack) unknown=\(items.count - camFront - camBack)")
            // SAY WHAT THIS ANSWER CANNOT SEE. The front-camera tag is EXIF, and EXIF needs the whole original:
            // for a library that lives in iCloud (170k of Reza's 187k) we never download one just to read it, so
            // those photos carry no camera at all. MEASURED 10-11: a bare "selfies" answers in 56 s with 0 judge
            // calls - and finds 267 photos, because only 267 originals are on the phone. Instant and badly
            // incomplete is not a win to report quietly.
            if album.camera == "front" {
                let unknown = items.filter { $0.camera.isEmpty }.count
                if unknown > scoped.count {
                    res.note = "\(scoped.count.formatted()) photos your phone recorded with its front camera. "
                             + "\(unknown.formatted()) others are stored in iCloud, so which camera took them cannot be "
                             + "read without downloading each one - any selfies among those are not here yet."
                }
            }
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
        // THE JUDGE-FREE ANSWER, BEFORE THE FIRST SCREEN IS CHOSEN: it is the fast answer, not an afterthought to
        // the judge. One P for every photo in scope (MilliScorer, policy V) through the plan the question decomposes
        // to - trip metadata, OR / AND parts, every leaf - and not one judge call anywhere in it.
        let places = PlaceIndex(items.map { $0.place })
        var localRow = [Int]()                        // each scoped photo's row in `places` (= its row in `items`)
        for (r, keep) in mask.enumerated() where keep { localRow.append(r) }
        let plan = QuestionTree.decompose(question!, places: places)
        let milli = await MilliScorer.make(plan: plan, scoped: scoped, index: index, embedder: embedder)
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
        // THE FAST ANSWER, WITH NO JUDGE CALL AT ALL (M85; Reza 11-05: "anything that takes many minutes is dumb").
        // A judge call costs 2.3-2.5 s on this phone, so ANY design that makes them is minutes: even the cascade's
        // measured ~65 calls is ~2.5 min. The query kinds that already feel like a product here - objects, "photos
        // of me", selfies - all make zero. So this answer makes zero too: one pass over vectors the phone already
        // stores. NOT instant, though, and the screen must not say it is: MEASURED on the Mac at 51 us a photo
        // (PolicyVCostTests, Accelerate, the same code), which is 7.6 s over his 148,428-photo scope and more on the
        // phone. Two orders of magnitude better than the judge's 2.5 minutes, and still not "at once" - so the word
        // is gone from the note, and the real cost is in the MILLI log line.
        // WHAT THE SCREEN MAY SAY (M94, and this IS the product): only what is COUNTED. The AUC .995 on food and the
        // ~79%-of-the-judge recall were both measured with EVERY photo described. Today ~1% are, so neither number
        // describes this answer, and a recall the app has not measured for THIS coverage and THIS kind of question
        // must never be on screen. No recall claim until RESULTS 71 measures it per category.
        // The judge still runs behind this and replaces these results as it confirms them, and "Look at everything"
        // is there for anyone who wants the exhaustive answer.
        var fastP = [Float]()
        var fastCall: CascadeCall? = nil
        if let m = milli {
            // the measured cascade's own dry pass (RESULTS 59d/69/70, 0 mismatches on 2,976 recorded configs), not a
            // new rule: it walks the same tree the judging pass will, and records each leaf's scope while it goes
            let call = CascadeCall(question: question!, crop: "", photos: Array(0..<scoped.count), local: localRow,
                                   faces: [Int](repeating: -1, count: scoped.count), places: places)
            let session = CascadeSession(hasFacts: { m.has($0) }, towervec: { m.score($0, $1) },
                                         vecB: { m.score($0, $1) },
                                         judge: { _, _, _ in throw MilliOnly() })   // a judge call here is a wiring bug
            fastP = session.dryAnswer(call)
            fastCall = call
            // THE HARNESS'S OWN CUT, NOT A NEW ONE (M100(4)). I invented 0.5 and labelled it PROVISIONAL. There was
            // nothing provisional to decide: for a kind-less (object) leaf CascadeSession returns the millisecond P
            // unchanged (CascadeRoute evaluate: `if kind.isEmpty { return pt }`) and the ENGINE's accept decides, and
            // accept = .7 is what RESULTS 69 / 71 measured. The invented cut showed 11,560 "food photos" on his
            // library - a number no measurement stood behind. For judge-kind leaves RESULTS 71 picks one global t*
            // on TRAIN, applied as a log-odds shift so that this same .7 keeps meaning P >= t*; until it lands those
            // leaves use the plain dry pass, which is also .7 here.
            let fastCut = Float(SearchEngine.accept)
            let fast = fastP.indices.filter { fastP[$0] >= fastCut }.sorted { fastP[$0] > fastP[$1] }.map { scoped[$0] }
            let floor = session.judgeFloor()
            res.inScope = scoped.count
            res.found = fast
            res.note = (fast.isEmpty
                ? "The quick check found none among the \(m.scorable.formatted()) of \(scoped.count.formatted()) "
                  + "photos in scope that it could look at."
                : "\(fast.count.formatted()) found by the quick check, which looked at \(m.scorable.formatted()) of "
                  + "the \(scoped.count.formatted()) photos in scope.")
                + " No photo was opened by the AI, which is why it was quick - and why it can be wrong."
            res.checkCalls = floor.calls
            res.checkSeconds = Double(floor.calls) * SearchEngine.secondsPerJudgeCall
            res.noCarefulCheck = floor.leaves == 0
            Judge.logMem("FASTANSWER \(album.name): \(fast.count) of \(m.scorable) scorable (\(scoped.count) in "
                         + "scope), \(m.describedCount) scored from written lines, cut=\(fastCut) (= engine accept), "
                         + "0 judge calls; careful check floor >= \(floor.calls) calls over \(floor.leaves) leaf question(s)")
            update(res)
        }
        // THE JUDGE NO LONGER RUNS BY ITSELF (M95(a); Reza 11-05: "anything that takes many minutes is dumb other than
        // the one time big model run thorugh"). It used to start the moment the fast answer was on screen and grind for
        // 20-32 minutes, heating the phone, for an answer the person may already have been happy with. Now it waits to
        // be asked, and the asking says what it costs.
        // Only when there IS a judge-free answer: with no image vectors or no vecB head there is nothing to show, so
        // that case still falls through to the measured stream below rather than showing an empty album.
        if !careful && !exhaustive && continuing == nil && milli != nil {
            if let n = coverFrameOnlyNote(media: album.media, pendingInScope: coverOnlyInScope,
                                          foundVideos: res.found.filter { entries[$0]?.isVideo == true }.count) {
                res.note += " " + n
            }
            res.done = true
            res.stillLooking = false
            update(res)
            return
        }
        var bestT = [String: Double](), bestS = [String: Float]()   // videos: the judge sees the frame that matched best
        for (u, k) in unitItem.enumerated() {
            guard let t = unitT[u] else { continue }
            var sc: Float = 0
            for l in looks { var d: Float = 0; for (a, b) in zip(units[u], l) { d += a * b }; sc += d / Float(looks.count) }
            if sc > (bestS[scoped[k]] ?? -.infinity) { bestS[scoped[k]] = sc; bestT[scoped[k]] = t }
        }
        // THE ORDER THE JUDGE WORKS THROUGH (M53). firstScreenDecision and likelyAtOnce above deliberately stay on
        // the RAW cosine scores: their thresholds are cosine values measured as such (RESULTS 45).
        let headT0 = Date()
        var rankKey = scores
        if let head = SearchEngine.learnedHead, let qv = try? embedder.vector(of: question!) {
            let hl = head.fold(question: qv).itemLogits(units: units, unitItem: unitItem, nItems: scoped.count)
            rankKey = blendScores(cosine: scores, headLogits: hl)
            Judge.logMem("HEADRANK n=\(scoped.count) ms=\(Int(Date().timeIntervalSince(headT0) * 1000))")
        }
        let order = rankKey.indices.sorted { rankKey[$0] > rankKey[$1] }.map { scoped[$0] }
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
        // THE >= .90 / <= .10 SHORTCUT IS GONE (M87(1), M92(1)). It looked like the facts path paying off - 48% of a
        // round answered without the judge - but nothing had measured that P >= .90 is precise or that P <= .10
        // misses nothing: AUC .995 says the RANKING is good, which is a different claim. Worse, it wrote 0.99 / 0.01
        // into the JUDGE's answers, so the first-answer bounds and the completeness statements counted photos the
        // judge never saw as judged. That is the one thing this app must not do. Where the millisecond scorer is
        // allowed to answer is now the cascade's own question (its per-leaf budget and threshold rule), and the
        // judge's answers hold only what the judge actually said.
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
        // "JUST CHECK THE ONES YOU SHOWED ME" - the other honest job, and the cheaper one (MAC-M95c). Seeing the two
        // numbers side by side on a real screen made it obvious: the quick answer says 1,509 found and the cascade
        // button offers to open 7,421, because the cascade is hunting for matches the quick answer MISSED. Someone who
        // just wants the wrong ones gone is asking for something else, and it costs a fraction as much.
        // The promises are different and both are sayable without a number we have not measured: this one can only
        // REMOVE photos, never find a missing one, and the button says exactly that.
        if careful, verifyShownOnly, !res.found.isEmpty {
            let shown = res.found
            var kept = [String]()
            var opened = 0, done = 0
            res.status = SearchEngine.lookingStatus
            update(res)
            do {
                for (k, id) in shown.enumerated() {
                    try await paceGate()
                    func ask(_ q: String) async throws -> Double {
                        if let c = await judge.cached(id + "|" + q) { return c }
                        await prefetch.want(Array(shown[k..<min(shown.count, k + 1 + SearchEngine.lookahead)]))
                        let t0 = Date()
                        guard let im = await prefetch.take(id)?.image else { missing.insert(id); return 0 }
                        readWait += Date().timeIntervalSince(t0)
                        let t1 = Date()
                        let v = try await judge.pYes(im, question: q, keepImage: false)
                        modelTime += Date().timeIntervalSince(t1); modelCalls += 1; opened += 1
                        await judge.remember(id + "|" + q, v); return v
                    }
                    var ok = try await ask(question!) >= cut
                    if ok, let ex = exQ, try await ask(ex) >= SearchEngine.accept { ok = false }
                    if ok, let fq = filterQ, try await ask(fq) < SearchEngine.accept { ok = false }
                    if ok { kept.append(id) }
                    done += 1; res.judged = done
                    if done % 5 == 0 || done == shown.count { res.found = kept; update(res) }
                }
            } catch is ThermalStop {
                res.status = "Stopped: your phone got too warm. What it checked before that is below."
            }
            res.found = kept + shown.dropFirst(done).filter { !kept.contains($0) }
            let removed = done - kept.count
            res.note = "The AI opened \(opened.formatted()) of the \(shown.count.formatted()) photos the quick check "
                     + "showed and removed \(removed.formatted()). This cannot find photos the quick check missed - "
                     + "only take out ones it got wrong."
            if !missing.isEmpty {
                res.note += " \(missing.count) photo(s) could not be checked: their originals are in iCloud and could "
                          + "not be downloaded now."
            }
            Judge.logMem("VERIFYSHOWN \(album.name): opened \(opened) of \(shown.count), removed \(removed), "
                         + "\(Int(modelTime)) s of model time")
            res.checkCalls = nil; res.checkSeconds = nil
            res.done = true; res.stillLooking = false
            if res.status == SearchEngine.lookingStatus { res.status = "" }
            update(res)
            return
        }
        // "CHECK THESE PROPERLY" (M95(a)): the MEASURED cascade drives the judge, instead of the 480-call stream that
        // the judge-free fast answer has made redundant. It judges each leaf's photos in scorer-rank order, a block of
        // max(50, 5% of that leaf's scope) at a time, and extends by one more block while over half the last block came
        // back a match - so the judge is spent where the matches are and stops where they run out. That shape is the one
        // with a number behind it: F1 within -.03 of judging everything for ~65 calls a search (RESULTS 59d/69/70), and
        // it is the cascade's OWN photo choice that was measured, which is why streamRounds does not get to pick here.
        if careful, let m = milli, let call = fastCall {
            var done = 0
            let judgeFree = res.found
            let cascadeJudge: ([JudgeItem], String, String) async throws -> [Float] = { items, q, _ in
                var out = [Float](repeating: 0, count: items.count)
                for (k, it) in items.enumerated() {
                    let id = scoped[it.photo]
                    if let c = await judge.cached(id + "|" + q) { out[k] = Float(c); continue }
                    // read the next few while this one is judged (RESULTS 44, item b)
                    let ahead = items[k..<min(items.count, k + 1 + SearchEngine.lookahead)].map { scoped[$0.photo] }
                    await prefetch.want(ahead)
                    try await paceGate()
                    let t0 = Date()
                    guard let im = await prefetch.take(id)?.image else { missing.insert(id); continue }
                    readWait += Date().timeIntervalSince(t0)
                    let t1 = Date()
                    let v = try await judge.pYes(im, question: q, keepImage: false)
                    modelTime += Date().timeIntervalSince(t1); modelCalls += 1
                    await judge.remember(id + "|" + q, v)
                    out[k] = Float(v)
                    done += 1
                    res.judged = done
                    if done % 5 == 0 || done == 1 { update(res) }
                }
                return out
            }
            let session = CascadeSession(hasFacts: { m.has($0) }, towervec: { m.score($0, $1) },
                                         vecB: { m.score($0, $1) }, judge: cascadeJudge)
            _ = session.dryAnswer(call)            // the leaf scopes the budget and threshold are derived from
            session.beginAnswers()
            res.status = SearchEngine.lookingStatus
            update(res)
            var p = [Float]()
            do { p = try await session.answer(call) }
            catch is ThermalStop {
                res.status = "Stopped: your phone got too warm. The matches found so far are below."
            }
            if !p.isEmpty {
                var pass = p.indices.filter { Double(p[$0]) >= cut }
                // the follow-ups stay a plain pass over the photos that got through, as they have always been: they
                // only ever see a handful of photos, and this is the shape their behaviour was measured in
                for (q, keepYes) in [(exQ, false), (filterQ, true)] {
                    guard let q, !pass.isEmpty else { continue }
                    var kept = [Int]()
                    for i in pass {
                        let id = scoped[i]
                        var v = await judge.cached(id + "|" + q)
                        if v == nil {
                            await prefetch.want([id])
                            try await paceGate()
                            if let im = await prefetch.take(id)?.image {
                                let t1 = Date()
                                v = try await judge.pYes(im, question: q, keepImage: false)
                                modelTime += Date().timeIntervalSince(t1); modelCalls += 1
                                await judge.remember(id + "|" + q, v!)
                                done += 1; res.judged = done
                            } else { missing.insert(id) }
                        }
                        let yes = (v ?? 0) >= SearchEngine.accept
                        if yes == keepYes { kept.append(i) }
                    }
                    pass = kept
                }
                res.found = pass.sorted { p[$0] > p[$1] }.map { scoped[$0] }
                let added = res.found.filter { !judgeFree.contains($0) }.count
                let dropped = judgeFree.filter { !res.found.contains($0) }.count
                res.note = "The AI opened \(done.formatted()) photo\(done == 1 ? "" : "s") itself, the ones the quick "
                         + "check ranked highest, and kept \(res.found.count.formatted()). It stopped extending where "
                         + "matches ran out - under half of the last block it checked were matches. "
                         + "Against the quick answer: \(added.formatted()) added, \(dropped.formatted()) removed."
                if !missing.isEmpty {
                    res.note += " \(missing.count) photo(s) could not be checked: their originals are in iCloud and "
                              + "could not be downloaded now."
                }
                // THE SHAPE, NOT THE WORDS (M96(2)): the cluster needs the plan's leaf kinds and scope sizes to put
                // this device number beside its own per-category ones. The leaf questions are Reza's own words about
                // his own photos and are never written to a log.
                let shape = session.leafShape().map { "\($0.kind):\($0.scope)" }.joined(separator: ",")
                Judge.logMem("CAREFUL \(album.name): leaves[\(shape)] calls=\(done) floor=\(res.checkCalls ?? -1) "
                             + "kept=\(res.found.count) added=\(added) removed=\(dropped) "
                             + "leafEvals=\(session.judgeLeafCalls) fallbackLeaves=\(session.fallbackLeafCalls) "
                             + "modelSecs=\(Int(modelTime))")
            }
            res.checkCalls = nil; res.checkSeconds = nil
            res.done = true; res.stillLooking = false
            if res.status == SearchEngine.lookingStatus { res.status = "" }
            update(res)
            return
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
    /// The facts score for every photo that has a written description, or nil when none do. One small head per
    /// photo over the 384 numbers the night pass stored, so the whole library costs milliseconds.
    /// Policy V (RESULTS 65): photos WITHOUT a line are not guessed at here - they go to the judge as before.
    static func judgeImage(_ id: String, frameAt t: Double?) async -> JudgeImage? {
        var img: CIImage? = nil
        if let t { img = await VideoFrames.frame(id, at: t, side: 896) }
        if img == nil { img = await PhotoLibrary.ciImage(id, side: 896) }
        return img.map(JudgeImage.init)
    }
}
