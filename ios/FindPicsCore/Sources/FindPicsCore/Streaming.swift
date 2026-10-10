// Judging in rounds with an honest completeness bound at every round: port of engine.stream_album (object/scene
// albums). Round 1: the judge checks the head (top of the fast ranking, extended in chunks while it keeps finding
// matches) + a uniform random tail sample fixed before any tail label; round k >= 2 doubles the head and samples
// twice as much of what is still unjudged. The error budget alpha is split in advance (round 1: alpha/2, each later
// round alpha/2/nLater), so every bound shown holds simultaneously, whenever the person stops.
import Foundation

public struct StreamParams: Sendable {
    public var headSize = 600, headChunk = 200, headMax = 6000
    public var headStopRate = 0.03, tailBudget = 1000, alpha = 0.05, accept = 0.7
    /// Round 1 stops extending the head when fewer than headStopRate of the last `headWindow` judged head photos are
    /// yes (nil = headChunk, the old rule). A window wider than the chunk is less noisy: on the phone (chunk 50) one
    /// unlucky 50 with a single yes ended round 1 for car at 1,250 photos while the yes rate there was ~7% (RESULTS 41).
    public var headWindow: Int? = nil
    /// Fast mode only (see fastModeWantsMore): run the next round without being asked while the round's random check
    /// of the unjudged rest found at least this many matches (0 = off), up to `autoRounds` rounds in all.
    public var autoRoundHits = 0, autoRounds = 1
    public var stream = true
    /// TIMED rounds (the phone, RESULTS 43; 0 = the round structure above, unchanged for the server and old replays).
    /// The judge is slow on a phone (~0.67 photos/s), so a round is a TIME budget: round k ends after
    /// roundCalls * 2^(k-1) judge calls (a new bound every doubling of time), or earlier when fast mode's head rule
    /// (headSize/headChunk/headWindow/headStopRate/headMax) is satisfied; rounds after that one double the calls so
    /// far. Random photos are judged DURING the head (one after every `interleave` head photos), so a bound exists
    /// from the first round on instead of after a head of up to 2,000 photos.
    public var roundCalls = 0
    /// One random photo after every `interleave` head photos (0 = none until a round ends).
    public var interleave = 0
    /// Rounds that end on time top the random sample up to this many (0 = only what interleaving drew); the round in
    /// which the head rule is satisfied always tops it up to tailBudget.
    public var roundTopUp = 0
    /// Timed fast mode: rounds run by itself after the head rule is satisfied while the random sample found at least
    /// autoRoundHits matches (RESULTS 41's auto round 2), at most this many.
    public var autoExtraRounds = 1
    /// Timed mode: share of alpha spent on round 1's bound (the rest is split evenly over rounds 2...K). Round 1's
    /// bound comes from a small random sample while the rest still holds many matches, which is where a bound at its
    /// nominal level does fail now and then (RESULTS 43); a smaller share makes it looser and rarer to fail.
    public var firstRoundAlphaShare = 0.5
    public init() {}
}

/// The phone's search parameters (Search.swift), here so the replay tests run exactly what ships.
/// Head rule = RESULTS 41 (head 150, +50 while >= 3% of the last 100 head photos are yes, up to 2,000; then 150 random
/// photos of the rest; one more round by itself if those found >= 3). Rounds = RESULTS 43, re-tuned for the measured
/// ~0.67 photos/s (MAC 10-10 13:56): a round ends every doubling of judge calls from 120 (~3 min), each with >= 30
/// random photos of the rest, so the first bound shows at 150 calls (~3.7 min; it used to wait for the whole head:
/// median 17, worst 50 min on the replay) and the matches still stream in head order. Interleaving the random photos
/// with the head was measured and lost (same calls, later matches). Round 1's bound gets 20% of alpha: at 50% it
/// overclaimed in 2 of 10,044 replayed rounds (each a < 1% event, but the replay rule is 0); at 20%: 0 of 10,044.
public func phoneFastParams(accept: Double) -> StreamParams {
    var p = StreamParams()
    p.headSize = 150; p.headChunk = 50; p.headWindow = 100; p.headMax = 2000; p.tailBudget = 150
    p.autoRoundHits = 3; p.autoRounds = 2; p.autoExtraRounds = 1
    p.roundCalls = 120; p.roundTopUp = 30; p.interleave = 0; p.firstRoundAlphaShare = 0.2
    p.accept = accept
    return p
}

/// Fast mode: run another round by itself? Only when the random check of the rest found several matches (many are
/// likely hiding there: the fast ranking put them low, e.g. small cars in the background, RESULTS 41). The bound stays
/// honest at every round: the error budget of all later rounds was split before round 1's tail sample was drawn.
public func fastModeWantsMore(_ r: Round, params p: StreamParams) -> Bool {
    if r.last { return false }
    if p.roundCalls > 0 {      // timed: keep going while the head rule is not satisfied (the round ended on time)
        if !r.headDone { return true }
        return p.autoRoundHits > 0 && r.roundsAfterHeadDone < p.autoExtraRounds && r.certificate.tailHits >= p.autoRoundHits
    }
    return p.autoRoundHits > 0 && r.k < p.autoRounds && r.certificate.tailHits >= p.autoRoundHits
}

/// Most rounds a timed run can have, fixed by n before any label. Every round but the one in which the head rule is
/// satisfied ends with calls >= its budget, so round k >= 2 ends with calls >= m0 * 2^(k-2), m0 = min(roundCalls,
/// headSize); round K is made to judge everything regardless.
public func timedMaxRounds(n: Int, params p: StreamParams) -> Int {
    let m0 = max(1, min(p.roundCalls, p.headSize))
    var k = 2, c = m0
    while c < n { c *= 2; k += 1 }
    return k
}

/// Error budget of round k in timed mode: share * alpha for round 1, (1 - share) * alpha / (K-1) for rounds 2...K, so
/// every bound shown holds simultaneously (union bound) whenever the person stops.
public func timedAlpha(_ k: Int, maxRounds K: Int, alpha: Double, firstShare: Double = 0.5) -> Double {
    k == 1 ? firstShare * alpha : (1 - firstShare) * alpha / Double(max(K - 1, 1))
}

public struct Round: Sendable {
    public let k: Int
    public let found: [Int]            // positions in `order` the judge said yes to (head, tail sample, earlier rounds)
    public let judged: Int, nHead: Int
    public let certificate: Certificate
    public let last: Bool
    /// Timed mode: fast mode's head rule has been satisfied (in this round or before); rounds since it was.
    public var headDone = false, roundsAfterHeadDone = 0
}

/// Deterministic generator (SplitMix64) so a search is reproducible from its seed.
public struct SeededRNG: RandomNumberGenerator {
    var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

/// `n`: number of in-scope photos, already ranked best-first (positions 0..<n). `judge(positions)` returns P(yes)
/// for each. `onRound` is called after every round; return false from it to stop early.
public func streamRounds(n: Int, params p: StreamParams = StreamParams(), seed: UInt64 = 0,
                         judge: ([Int]) async throws -> [Double], onRound: (Round) async -> Bool) async rethrows {
    if p.roundCalls > 0 && p.stream {
        var rng = SeededRNG(seed: seed)
        let perm = Array(0..<n).shuffled(using: &rng)
        try await streamTimed(n: n, perm: perm, params: p, judge: judge, onRound: onRound); return
    }
    var seen = [Int: Double]()
    func pj(_ pos: [Int]) async throws -> [Double] {
        let new = pos.filter { seen[$0] == nil }
        if !new.isEmpty { for (q, v) in zip(new, try await judge(new)) { seen[q] = v } }
        return pos.map { seen[$0]! }
    }
    var rng = SeededRNG(seed: seed)
    var nHead = min(p.headSize, n)
    _ = try await pj(Array(0..<nHead))
    while nHead < min(p.headMax, n) {
        let last = try await pj(Array(max(0, nHead - (p.headWindow ?? p.headChunk))..<nHead))
        if Double(last.filter { $0 >= p.accept }.count) / Double(max(last.count, 1)) < p.headStopRate { break }
        nHead = min(nHead + p.headChunk, n); _ = try await pj(Array(0..<nHead))
    }
    var h = nHead, nLater = 0
    while true { h = min(n, 2 * h); if h < n { nLater += 1 } else { break } }
    let perm = Array(0..<n).shuffled(using: &rng)          // fixed in advance: later rounds take its first unjudged items
    var k = 0
    var firstSample: [Int] = []
    while true {
        k += 1
        let tail = n - nHead
        let nT = min(tail, p.tailBudget * (1 << (k - 1)))
        var ts: [Int]
        if k == 1 {
            ts = Array(Array(nHead..<n).shuffled(using: &rng).prefix(nT)); firstSample = ts
        } else {
            ts = Array(perm.filter { $0 >= nHead }.prefix(nT))
        }
        let ph = try await pj(Array(0..<nHead)), pt = try await pj(ts)
        let yh = ph.filter { $0 >= p.accept }.count, yt = pt.map { $0 >= p.accept }
        let inRound = Set(Array(0..<nHead) + ts)
        let prev = seen.filter { $0.value >= p.accept && !inRound.contains($0.key) }.map { $0.key }
        let aK = !p.stream ? p.alpha : (k == 1 ? p.alpha / 2 : p.alpha / 2 / Double(max(nLater, 1)))
        let cert = certify(found: yh + yt.filter { $0 }.count + prev.count, nTail: tail, tailLabels: yt, alpha: aK)
        let found = (0..<nHead).filter { ph[$0] >= p.accept } + zip(ts, pt).filter { $0.1 >= p.accept }.map { $0.0 } + prev
        let last = !p.stream || nHead >= n
        let go = await onRound(Round(k: k, found: found.sorted(), judged: seen.count, nHead: nHead, certificate: cert, last: last))
        if last || !go { break }
        nHead = min(n, 2 * nHead)
        _ = firstSample
    }
}

/// Timed rounds (StreamParams.roundCalls > 0). `perm`: a uniform permutation of 0..<n fixed before any label.
/// Why the bound stays honest: the random sample of round k is every perm item drawn so far that lies in the tail
/// [nHead, n). Items are only ever skipped when they were inside the head when drawn, so that set is the first m items
/// of perm restricted to the tail; and the head boundary, the draw times and the time budgets depend on the head's
/// labels and on perm positions, never on a tail label. Any reordering of the tail's positions leaves the run unchanged,
/// so given m the sample is a uniform m-subset of the tail. Mirrored exactly by eval/tune_stop_time.py sim_timed.
func streamTimed(n: Int, perm: [Int], params p: StreamParams,
                 judge: ([Int]) async throws -> [Double], onRound: (Round) async -> Bool) async rethrows {
    var seen = [Int: Double]()
    let K = timedMaxRounds(n: n, params: p)
    var b = 0, pp = 0, since = 0, satRound = 0
    func judgeOne(_ q: Int) async throws {
        if seen[q] == nil { seen[q] = try await judge([q])[0] }
    }
    func draw() async throws -> Bool {
        while pp < n && (perm[pp] < b || seen[perm[pp]] != nil) { pp += 1 }
        if pp >= n { return false }
        try await judgeOne(perm[pp]); pp += 1
        return true
    }
    func sample() -> [Int] { perm[0..<pp].filter { $0 >= b } }
    func headRuleSatisfied() -> Bool {
        if b >= min(p.headMax, n) { return true }
        if b < p.headSize || (b - p.headSize) % max(p.headChunk, 1) != 0 { return false }
        let w = p.headWindow ?? p.headChunk
        let last = (max(0, b - w)..<b).map { seen[$0]! }
        return Double(last.filter { $0 >= p.accept }.count) / Double(max(last.count, 1)) < p.headStopRate
    }
    var k = 0
    while true {
        k += 1
        let budget = k >= K ? n : min(n, satRound == 0 ? p.roundCalls << min(k - 1, 40) : 2 * seen.count)
        while seen.count < budget && b < n {
            if p.interleave > 0 && since >= p.interleave {
                since = 0
                if try await draw() { continue }
            }
            try await judgeOne(b); b += 1; since += 1
            if satRound == 0 && headRuleSatisfied() { satRound = k; break }
        }
        if seen.count >= n { b = n }
        let want = satRound > 0 ? p.tailBudget : p.roundTopUp
        while b < n, sample().count < want, try await draw() {}
        let ts = sample()
        let found = seen.filter { $0.value >= p.accept }.map { $0.key }.sorted()
        let cert = certify(found: found.count, nTail: n - b, tailLabels: ts.map { seen[$0]! >= p.accept },
                           alpha: timedAlpha(k, maxRounds: K, alpha: p.alpha, firstShare: p.firstRoundAlphaShare))
        let last = b >= n
        var r = Round(k: k, found: found, judged: seen.count, nHead: b, certificate: cert, last: last)
        r.headDone = satRound > 0; r.roundsAfterHeadDone = satRound > 0 ? k - satRound : 0
        let go = await onRound(r)
        if last || !go { break }
    }
}
