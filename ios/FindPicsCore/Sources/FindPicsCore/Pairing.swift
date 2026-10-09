// Two opposite looks of one person ("heavier" vs "fit"): port of engine._two_groups / engine._split_pair.
import Foundation

public let pairSure = 0.9   // a photo goes to an album only if the split is >= 90% sure
public let pairMin = 20     // fewer shared photos: too few to see two groups (caller uses the rank margin)

func logit(_ p: Double) -> Double { let q = min(max(p, 1e-6), 1 - 1e-6); return log(q / (1 - q)) }

func percentile(_ xs: [Double], _ q: Double) -> Double {   // numpy default (linear interpolation)
    let s = xs.sorted(); let pos = q / 100 * Double(s.count - 1)
    let lo = Int(pos.rounded(.down)), hi = min(lo + 1, s.count - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
}

public let pairSdFloor = 1e-2   // a mixture component's spread never goes below this (logit units)
public let pairOneEvent = 0.75  // a group with >= this share of its weight in ONE event is that event, not a lasting look

/// EM for a 1-D two-Gaussian mixture from starting means mu0 (engine._em2). Returns (mu, sd, w).
func em2(_ x: [Double], _ mu0: [Double]) -> (mu: [Double], sd: [Double], w: [Double]) {
    let n = Double(x.count)
    let mean = x.reduce(0, +) / n
    let std = sqrt(x.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / n)
    var mu = mu0, sd = [max(std, 1e-3), max(std, 1e-3)], w = [0.5, 0.5]
    for _ in 0..<200 {
        var r = [[Double]](repeating: [0, 0], count: x.count)
        for (i, v) in x.enumerated() {
            let l = (0..<2).map { -0.5 * pow((v - mu[$0]) / sd[$0], 2) - log(sd[$0]) + log(w[$0]) }
            let m = max(l[0], l[1]); let e = [exp(l[0] - m), exp(l[1] - m)]; let s = e[0] + e[1]
            r[i] = [e[0] / s, e[1] / s]
        }
        var nk = [1e-9, 1e-9], newMu = [0.0, 0.0]
        for (i, v) in x.enumerated() { for k in 0..<2 { nk[k] += r[i][k]; newMu[k] += r[i][k] * v } }
        newMu = [newMu[0] / nk[0], newMu[1] / nk[1]]
        var vs = [0.0, 0.0]
        for (i, v) in x.enumerated() { for k in 0..<2 { vs[k] += r[i][k] * pow(v - newMu[k], 2) } }
        sd = [max(sqrt(vs[0] / nk[0]), pairSdFloor), max(sqrt(vs[1] / nk[1]), pairSdFloor)]; w = [nk[0] / n, nk[1] / n]
        let done = abs(newMu[0] - mu[0]) <= 1e-6 + 1e-5 * abs(mu[0]) && abs(newMu[1] - mu[1]) <= 1e-6 + 1e-5 * abs(mu[1])
        mu = newMu
        if done { break }
    }
    return (mu, sd, w)
}

/// Exact best 1-D split into two groups (largest between-group variance; first best cut wins): the two group means
/// (low, high), both equal when all values are equal (engine._two_means). Deterministic and global.
func twoMeans(_ x: [Double]) -> [Double] {
    let s = x.sorted(), n = s.count
    var c = [Double](repeating: 0, count: n), acc = 0.0
    for (i, v) in s.enumerated() { acc += v; c[i] = acc }
    let tot = c[n - 1]
    var best = -1.0, out = [s[0], s[0]]
    for k in 1..<max(n, 1) where s[k] != s[k - 1] {
        let m0 = c[k - 1] / Double(k), m1 = (tot - c[k - 1]) / Double(n - k)
        let b = Double(k) * Double(n - k) * (m0 - m1) * (m0 - m1)
        if b > best { best = b; out = [m0, m1] }
    }
    return out
}

/// 1-D two-component Gaussian mixture by EM (engine._two_groups). Returns (P(upper group) per value, two groups clearly
/// better than one by BIC (> 10), midpoint between the group means). `groups`: event label per value, optional.
/// Two starts (quartiles; exact two-means split). A fit is degenerate when a component shrank to the spread floor or
/// (with groups) holds >= pairOneEvent of its weight in ONE event: a long event's photos share one median, a point mass
/// whose likelihood is unbounded. Non-degenerate fits win, then the higher likelihood, then the quartile start; if
/// every start is degenerate, the best of them (RESULTS 39).
public func twoGroups(_ x: [Double], groups: [String]? = nil) -> (post: [Double], clear: Bool, mid: Double) {
    let n = Double(x.count)
    let mean = x.reduce(0, +) / n
    let std = sqrt(x.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / n)
    func logLiks(_ mu: [Double], _ sd: [Double], _ w: [Double]) -> (ll: [[Double]], total: Double) {
        var ll = [[Double]](), total = 0.0
        for v in x {
            let l = (0..<2).map { -0.5 * pow((v - mu[$0]) / sd[$0], 2) - log(sd[$0]) + log(w[$0]) - 0.5 * log(2 * .pi) }
            let m = max(l[0], l[1]); total += m + log(exp(l[0] - m) + exp(l[1] - m)); ll.append(l)
        }
        return (ll, total)
    }
    func oneEventShare(_ ll: [[Double]]) -> Double {
        guard let g = groups else { return 0 }
        var best = 0.0
        for k in 0..<2 {
            var per = [String: Double](), tot = 0.0
            for (i, l) in ll.enumerated() {
                let m = max(l[0], l[1]); let e = [exp(l[0] - m), exp(l[1] - m)]; let r = e[k] / (e[0] + e[1])
                per[g[i], default: 0] += r; tot += r
            }
            best = max(best, (per.values.max() ?? 0) / max(tot, 1e-12))
        }
        return best
    }
    typealias Fit = (good: Bool, L: Double, start: Int, mu: [Double], sd: [Double], w: [Double], ll: [[Double]])
    var best: Fit? = nil
    for (k, start) in [[percentile(x, 25), percentile(x, 75)], twoMeans(x)].enumerated() {
        let (mu, sd, w) = em2(x, start)
        let (ll, L) = logLiks(mu, sd, w)
        let bad = sd.contains { $0 <= pairSdFloor * (1 + 1e-9) } || oneEventShare(ll) >= pairOneEvent
        let f: Fit = (!bad, L, k, mu, sd, w, ll)
        if let b = best {   // non-degenerate first, then likelihood; ties keep the earlier start
            if (f.good && !b.good) || (f.good == b.good && f.L > b.L) { best = f }
        } else { best = f }
    }
    let f = best!
    let up = f.mu[1] > f.mu[0] ? 1 : 0
    let post = f.ll.map { l -> Double in let m = max(l[0], l[1]); let e = [exp(l[0] - m), exp(l[1] - m)]; return e[up] / (e[0] + e[1]) }
    let s1 = max(std, 1e-2)
    let l1 = x.map { -0.5 * pow(($0 - mean) / s1, 2) - log(s1) - 0.5 * log(2 * .pi) }.reduce(0, +)
    let bic1 = 2 * log(n) - 2 * l1, bic2 = 5 * log(n) - 2 * f.L
    return (post, bic2 < bic1 - 10, (f.mu[0] + f.mu[1]) / 2)
}

/// Album index (0 = first album, 1 = second) per shared photo, nil = in neither; returns nil when there are not two
/// clear groups (caller falls back to the rank margin). `eventOf`: photos <= 3 h apart share one median score.
public func splitPair(pA: [String: Double], pB: [String: Double], eventOf: [String: String]? = nil) -> [String: Int?]? {
    let ids = pA.keys.filter { pB[$0] != nil }.sorted()
    if ids.count < pairMin { return nil }
    var x = ids.map { logit(pA[$0]!) - logit(pB[$0]!) }
    if let ev = eventOf {
        var groups = [String: [Int]]()
        for (i, id) in ids.enumerated() { groups[ev[id] ?? "_\(id)", default: []].append(i) }
        var med = x
        for (_, ix) in groups {
            let s = ix.map { x[$0] }.sorted(); let c = s.count
            let m = c % 2 == 1 ? s[c / 2] : (s[c / 2 - 1] + s[c / 2]) / 2
            for i in ix { med[i] = m }
        }
        x = med
    }
    let (post, clear, mid) = twoGroups(x, groups: eventOf.map { ev in ids.map { ev[$0] ?? "_\($0)" } })
    if !clear { return nil }
    var out = [String: Int?]()
    for (i, id) in ids.enumerated() {
        let album: Int? = (post[i] >= pairSure && x[i] > mid) ? 0 : ((post[i] <= 1 - pairSure && x[i] < mid) ? 1 : nil)
        out.updateValue(album, forKey: id)   // `out[id] = nil` would DELETE the key; in-between photos must stay listed
    }
    return out
}

/// Fraction of this person's own photos scoring at or below each photo (engine: rel = searchsorted(sorted, p, right)/n).
public func withinPersonRank(_ p: [String: Double]) -> [String: Double] {
    let srt = p.values.sorted(), n = Double(max(srt.count, 1))
    func upper(_ v: Double) -> Int { var lo = 0, hi = srt.count; while lo < hi { let m = (lo + hi) / 2; if srt[m] <= v { lo = m + 1 } else { hi = m } }; return lo }
    return p.mapValues { Double(upper($0)) / n }
}

/// Fallback when splitPair sees no clear two groups (engine.make_exclusive rank margin): a photo goes to the album it
/// ranks >= `margin` higher in (within the person's own photos); a photo judged for only one album follows that album's
/// usual rule (rank above `ownCut`). nil = not clearly either.
public func rankMarginPair(pA: [String: Double], pB: [String: Double], margin: Double = 0.3, ownCut: Double = 0.5) -> [String: Int?] {
    let ra = withinPersonRank(pA), rb = withinPersonRank(pB)
    var out = [String: Int?]()
    for id in Set(pA.keys).union(pB.keys) {
        let v: Int?
        switch (ra[id], rb[id]) {
        case let (a?, b?): v = a - b >= margin ? 0 : (b - a >= margin ? 1 : nil)
        case let (a?, nil): v = a > ownCut ? 0 : nil
        case let (nil, b?): v = b > ownCut ? 1 : nil
        default: continue
        }
        out.updateValue(v, forKey: id)     // `out[id] = nil` would DELETE the key ("not clearly either" must stay)
    }
    return out
}
