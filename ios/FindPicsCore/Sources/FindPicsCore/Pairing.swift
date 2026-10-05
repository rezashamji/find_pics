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

/// 1-D two-component Gaussian mixture by EM. Returns (P(upper group) per value, two groups clearly better than one by
/// BIC (> 10), midpoint between the group means).
public func twoGroups(_ x: [Double]) -> (post: [Double], clear: Bool, mid: Double) {
    let n = Double(x.count)
    let mean = x.reduce(0, +) / n
    let std = sqrt(x.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / n)
    var mu = [percentile(x, 25), percentile(x, 75)], sd = [max(std, 1e-3), max(std, 1e-3)], w = [0.5, 0.5]
    func logLik(_ v: Double, _ k: Int, constant: Bool) -> Double {
        -0.5 * pow((v - mu[k]) / sd[k], 2) - log(sd[k]) + log(w[k]) - (constant ? 0.5 * log(2 * .pi) : 0)
    }
    for _ in 0..<200 {
        var r = [[Double]](repeating: [0, 0], count: x.count)
        for (i, v) in x.enumerated() {
            let l = [logLik(v, 0, constant: false), logLik(v, 1, constant: false)]
            let m = max(l[0], l[1]); let e = [exp(l[0] - m), exp(l[1] - m)]; let s = e[0] + e[1]
            r[i] = [e[0] / s, e[1] / s]
        }
        var nk = [1e-9, 1e-9], newMu = [0.0, 0.0]
        for (i, v) in x.enumerated() { for k in 0..<2 { nk[k] += r[i][k]; newMu[k] += r[i][k] * v } }
        newMu = [newMu[0] / nk[0], newMu[1] / nk[1]]
        var vs = [0.0, 0.0]
        for (i, v) in x.enumerated() { for k in 0..<2 { vs[k] += r[i][k] * pow(v - newMu[k], 2) } }
        sd = [max(sqrt(vs[0] / nk[0]), 1e-2), max(sqrt(vs[1] / nk[1]), 1e-2)]; w = [nk[0] / n, nk[1] / n]
        let done = abs(newMu[0] - mu[0]) <= 1e-6 + 1e-5 * abs(mu[0]) && abs(newMu[1] - mu[1]) <= 1e-6 + 1e-5 * abs(mu[1])
        mu = newMu
        if done { break }
    }
    var l2 = 0.0, post = [Double]()
    let up = mu[1] > mu[0] ? 1 : 0
    for v in x {
        let l = [logLik(v, 0, constant: true), logLik(v, 1, constant: true)]
        let m = max(l[0], l[1]); let e = [exp(l[0] - m), exp(l[1] - m)]
        l2 += m + log(e[0] + e[1]); post.append(e[up] / (e[0] + e[1]))
    }
    let s1 = max(std, 1e-2)
    let l1 = x.map { -0.5 * pow(($0 - mean) / s1, 2) - log(s1) - 0.5 * log(2 * .pi) }.reduce(0, +)
    let bic1 = 2 * log(n) - 2 * l1, bic2 = 5 * log(n) - 2 * l2
    return (post, bic2 < bic1 - 10, (mu[0] + mu[1]) / 2)
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
    let (post, clear, mid) = twoGroups(x)
    if !clear { return nil }
    var out = [String: Int?]()
    for (i, id) in ids.enumerated() {
        let album: Int? = (post[i] >= pairSure && x[i] > mid) ? 0 : ((post[i] <= 1 - pairSure && x[i] < mid) ? 1 : nil)
        out.updateValue(album, forKey: id)   // `out[id] = nil` would DELETE the key; in-between photos must stay listed
    }
    return out
}
