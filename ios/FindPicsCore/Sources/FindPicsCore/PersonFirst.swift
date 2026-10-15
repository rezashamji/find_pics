// Person searches get a FIRST ANSWER instead of judging every face-matched photo (RESULTS 62). Port of
// src/findpics/person_first.py; golden fixture Fixtures/person_first.json (eval/person_first/fixtures.py).
//
// Until 10-14 a person album with a condition judged EVERY photo the face filter accepted before showing anything:
// 41,541 photos for the owner on Reza's phone, ~55 h at ~4.8 s a photo (MAC M67). Which branch a question takes is
// isLookQuestion (RESULTS 49c).
//
// FACT ("me at the beach", "me wearing a hat": the judge's yes alone, P >= 0.7). Exactly an ordinary fast-mode search
// whose scope is the person's face-matched photos: order them with personFactOrder (the album's look vector, blended
// with the learned head like Search.run), then streamRounds(params: phoneFastParams(accept: 0.7)) - same rounds, same
// Clopper-Pearson bound, first answer at firstAnswerDone, then keep looking (paced) until everything is judged, where
// the album is what judging everything gives.
//
// LOOK ("me looking heavier vs fit", "tired": relative to the person's own photos). No bound exists for a relative
// album, but the full rule gives every photo its EVENT's median score (photos <= 3 h apart; splitPair), so a few judged
// photos sort a whole event. streamPersonLook judges photos in lookWalkOrder (a random order fixed before any answer,
// at most lookCap judged photos per event first, the rest after), asks every question of a photo before the next
// photo, and after every 20 judge calls (240 after the first answer at lookFirstCalls) reports the albums:
//   pair:   lookPairAlbums = lookPairPartial (event medians of the judged log-odds differences, the two-group split
//           fitted on the judged photos weighted by the photos they stand for) or, without two clear groups, the rank
//           margin on the judged photos (as the full rule falls back);
//   single: lookSinglePartial (a judged photo by its own P, an unjudged photo of a reached event by its event's median,
//           kept when it ranks above the middle of the person's photos, the middle estimated with the same weights).
// Photos of events not reached yet are "not sorted yet". With every photo judged both equal the full rules exactly.
import Foundation

/// A LOOK search's first answer: the same judge-call budget as fast mode's (phoneFastParams.fastStopCalls).
public let lookFirstCalls = 480
/// First pass of the walk: at most this many judged photos per event.
public let lookCap = 3
/// Screen updates: every 20 judge calls up to the first answer, every 240 after it, and at the end.
public let lookUpdateEvery = 20, lookUpdateEveryLater = 240

/// 64-bit FNV-1a over the UTF-8 bytes (person_first.fnv1a64).
public func fnv1a64(_ s: String) -> UInt64 {
    var h: UInt64 = 0xcbf29ce484222325
    for b in s.utf8 { h ^= UInt64(b); h = h &* 0x100000001b3 }
    return h
}

/// MurmurHash3's 64-bit finalizer (person_first.fmix64): FNV-1a alone leaves its top bits nearly blind to the LAST
/// bytes, so ids that differ only at the end sorted next to each other (RESULTS 62b).
public func fmix64(_ x: UInt64) -> UInt64 {
    var h = x
    h ^= h >> 33; h = h &* 0xff51afd7ed558ccd
    h ^= h >> 33; h = h &* 0xc4ceb9fe1a85ec53
    return h ^ (h >> 33)
}

/// The walk's random key of a photo: fmix64(fnv1a64("<seed>|<id>")) (person_first.walk_key).
public func walkKey(seed: Int, id: String) -> UInt64 { fmix64(fnv1a64("\(seed)|\(id)")) }

func eventKey(_ eventOf: [String: String]?, _ id: String) -> String { eventOf?[id] ?? "_\(id)" }

/// The LOOK judging order, fixed before any answer: positions into `ids`. Photos sorted by walkKey(seed:id:) (ties
/// by id); a photo is taken while its event has fewer than `cap` taken photos, the others follow in the same order
/// (cap nil or 0: the plain random order).
/// `bySize` (RESULTS 62b, opt-in): COVERAGE-FIRST - events largest first (ties: the key of the event's first photo);
/// pass 1 = one photo of every event, pass 2 = photos 2...cap of every event in the same order, then the rest in key order.
public func lookWalkOrder(ids: [String], eventOf: [String: String]?, seed: Int = 0, cap: Int? = lookCap,
                          bySize: Bool = false) -> [Int] {
    let keys = ids.map { walkKey(seed: seed, id: $0) }
    let perm = ids.indices.sorted { keys[$0] != keys[$1] ? keys[$0] < keys[$1] : ids[$0] < ids[$1] }
    if bySize {
        var per = [String: [Int]](), seen = [String]()
        for k in perm {
            let e = eventKey(eventOf, ids[k])
            if per[e] == nil { seen.append(e) }
            per[e, default: []].append(k)
        }
        let evs = seen.sorted { a, b in
            let ka = per[a]!, kb = per[b]!
            if ka.count != kb.count { return ka.count > kb.count }
            if keys[ka[0]] != keys[kb[0]] { return keys[ka[0]] < keys[kb[0]] }
            return ids[ka[0]] < ids[kb[0]]
        }
        let c2 = max(cap ?? 1, 1)
        let first = evs.map { per[$0]![0] }
        let second = evs.flatMap { Array(per[$0]!.dropFirst().prefix(c2 - 1)) }
        let taken = Set(first).union(second)
        return first + second + perm.filter { !taken.contains($0) }
    }
    guard let c = cap, c > 0 else { return perm }
    var cnt = [String: Int](), first = [Int](), rest = [Int]()
    for k in perm {
        let e = eventKey(eventOf, ids[k])
        if cnt[e, default: 0] < c { first.append(k); cnt[e, default: 0] += 1 } else { rest.append(k) }
    }
    return first + rest
}

func median(_ v: [Double]) -> Double {
    let s = v.sorted(), c = s.count
    return c % 2 == 1 ? s[c / 2] : (s[c / 2 - 1] + s[c / 2]) / 2
}

/// One LOOK album after some photos are judged (pYes: the judged ones): true / false for every SORTED photo (judged, or
/// in an event with a judged photo when `propagate`); photos absent from the result are not sorted yet.
public func lookSinglePartial(ids: [String], eventOf: [String: String]?, pYes: [String: Double], relCut: Double = 0.5,
                              propagate: Bool = true) -> [String: Bool] {
    let J = ids.filter { pYes[$0] != nil }.sorted()
    if J.isEmpty { return [:] }
    var size = [String: Int](), per = [String: [Double]]()
    for i in ids { size[eventKey(eventOf, i), default: 0] += 1 }
    for i in J { per[eventKey(eventOf, i), default: []].append(pYes[i]!) }
    let med = per.mapValues(median)
    let vals = J.map { pYes[$0]! }
    let w = J.map { Double(size[eventKey(eventOf, $0)]!) / Double(per[eventKey(eventOf, $0)]!.count) }
    let o = vals.indices.sorted { vals[$0] != vals[$1] ? vals[$0] < vals[$1] : $0 < $1 }   // stable, as numpy
    let sv = o.map { vals[$0] }
    var cw = [Double](), acc = 0.0
    for k in o { acc += w[k]; cw.append(acc) }
    let W = acc
    func rel(_ v: Double) -> Double {
        var lo = 0, hi = sv.count
        while lo < hi { let m = (lo + hi) / 2; if sv[m] <= v { lo = m + 1 } else { hi = m } }
        return (lo > 0 ? cw[lo - 1] : 0) / W
    }
    var out = [String: Bool]()
    for i in ids {
        if let p = pYes[i] { out[i] = rel(p) > relCut }
        else if propagate, let m = med[eventKey(eventOf, i)] { out[i] = rel(m) > relCut }
    }
    return out
}

/// Two opposite LOOK albums after some photos are judged for both: 0 (first album) / 1 (second) / nil (not clearly
/// either) for every SORTED photo; absent = not sorted yet. nil when fewer than pairMin photos are judged or the judged
/// photos do not form two clear groups (lookPairAlbums then uses the rank margin on the judged photos).
public func lookPairPartial(ids: [String], eventOf: [String: String]?, pA: [String: Double], pB: [String: Double],
                            propagate: Bool = true) -> [String: Int?]? {
    let J = ids.filter { pA[$0] != nil && pB[$0] != nil }.sorted()
    if J.count < pairMin { return nil }
    var size = [String: Int](), per = [String: [Double]]()
    for i in ids { size[eventKey(eventOf, i), default: 0] += 1 }
    for i in J { per[eventKey(eventOf, i), default: []].append(logit(pA[i]!) - logit(pB[i]!)) }
    let med = per.mapValues(median)
    let ev = J.map { eventKey(eventOf, $0) }
    let x = ev.map { med[$0]! }
    let w = ev.map { Double(size[$0]!) / Double(per[$0]!.count) }
    let (post, clear, mid) = twoGroups(x, groups: ev, weights: w)
    if !clear { return nil }
    var side = [String: Int?]()
    for (k, e) in ev.enumerated() {
        let s: Int? = (post[k] >= pairSure && x[k] > mid) ? 0 : ((post[k] <= 1 - pairSure && x[k] < mid) ? 1 : nil)
        side.updateValue(s, forKey: e)
    }
    var out = [String: Int?]()
    for i in ids {
        let e = eventKey(eventOf, i)
        if let s = side[e], propagate || (pA[i] != nil && pB[i] != nil) { out.updateValue(s, forKey: i) }
    }
    return out
}

/// What a paired LOOK search shows after some photos are judged for both questions (person_first.pair_albums).
public func lookPairAlbums(ids: [String], eventOf: [String: String]?, pA: [String: Double], pB: [String: Double],
                           propagate: Bool = true) -> [String: Int?] {
    if let a = lookPairPartial(ids: ids, eventOf: eventOf, pA: pA, pB: pB, propagate: propagate) { return a }
    let a = pA.filter { pB[$0.key] != nil }, b = pB.filter { pA[$0.key] != nil }
    return a.isEmpty ? [:] : rankMarginPair(pA: a, pB: b)
}

/// person_first.update_due: refresh the albums every lookUpdateEvery judge calls up to lookFirstCalls, every
/// lookUpdateEveryLater after, and when everything is judged. Only WHEN the screen changes, never what it shows.
public func lookUpdateDue(calls: Int, total: Int) -> Bool {
    if calls >= total { return true }
    return calls <= lookFirstCalls ? calls % lookUpdateEvery == 0 : (calls - lookFirstCalls) % lookUpdateEveryLater == 0
}

/// One screen update of a LOOK person search.
public struct LookUpdate: Sendable {
    /// Judge calls so far (photos x questions) and photos judged.
    public let calls: Int, judged: Int
    /// pair: id -> 0 / 1 / nil (neither); single: id -> 0 (in the album) / nil (not). Absent = not sorted yet.
    public let albums: [String: Int?]
    /// The judge's answers so far, one dictionary per question.
    public let p: [[String: Double]]
    public let total: Int
    /// calls >= lookFirstCalls or everything judged: the first answer is on screen.
    public let firstAnswer: Bool, last: Bool
    public var sorted: Int { albums.count }
}

/// A LOOK person search (person_first.stream_look). `questions`: 1 (one look) or 2 (two opposite looks of one person,
/// asked of each photo one after the other so the judge can reuse the photo's image prefix). `judge(id, q)` returns
/// P(yes) for question q on photo id. `onUpdate` returns false to stop (Stop, a new search, or the phone too hot).
public func streamPersonLook(ids: [String], eventOf: [String: String]?, questions: Int, seed: Int = 0,
                             cap: Int? = lookCap, bySize: Bool = false, judge: (String, Int) async throws -> Double,
                             onUpdate: (LookUpdate) async -> Bool) async rethrows {
    precondition(questions == 1 || questions == 2)
    let order = lookWalkOrder(ids: ids, eventOf: eventOf, seed: seed, cap: cap, bySize: bySize)
    var P = [[String: Double]](repeating: [:], count: questions)
    let total = questions * ids.count
    var calls = 0
    for (n, k) in order.enumerated() {
        let id = ids[k]
        for q in 0..<questions { P[q][id] = try await judge(id, q); calls += 1 }
        let done = n + 1
        if lookUpdateDue(calls: calls, total: total) || done == ids.count {
            let albums: [String: Int?]
            if questions == 2 {
                albums = lookPairAlbums(ids: ids, eventOf: eventOf, pA: P[0], pB: P[1])
            } else {
                var a = [String: Int?]()
                for (i, keep) in lookSinglePartial(ids: ids, eventOf: eventOf, pYes: P[0]) { a.updateValue(keep ? 0 : nil, forKey: i) }
                albums = a
            }
            let u = LookUpdate(calls: calls, judged: done, albums: albums, p: P, total: ids.count,
                               firstAnswer: calls >= lookFirstCalls || done == ids.count, last: done == ids.count)
            if !(await onUpdate(u)) || u.last { return }
        }
    }
}

/// FACT person search: positions of the person's photos, best first, by the look score (cosine, lookScores) or its
/// blend with the learned head's logits (blendScores, RESULTS 48). Ties keep the input order; NaN ranks last.
public func personFactOrder(cosine: [Float], headLogits: [Float]? = nil) -> [Int] {
    var key = headLogits.map { blendScores(cosine: cosine, headLogits: $0) } ?? cosine
    key = key.map { $0.isNaN ? -.infinity : $0 }
    return key.indices.sorted { key[$0] != key[$1] ? key[$0] > key[$1] : $0 < $1 }
}
