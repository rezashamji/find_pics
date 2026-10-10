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
    public init() {}
}

/// Fast mode: run another round by itself? Only when the random check of the rest found several matches (many are
/// likely hiding there: the fast ranking put them low, e.g. small cars in the background, RESULTS 41). The bound stays
/// honest at every round: the error budget of all later rounds was split before round 1's tail sample was drawn.
public func fastModeWantsMore(_ r: Round, params p: StreamParams) -> Bool {
    !r.last && p.autoRoundHits > 0 && r.k < p.autoRounds && r.certificate.tailHits >= p.autoRoundHits
}

public struct Round: Sendable {
    public let k: Int
    public let found: [Int]            // positions in `order` the judge said yes to (head, tail sample, earlier rounds)
    public let judged: Int, nHead: Int
    public let certificate: Certificate
    public let last: Bool
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
