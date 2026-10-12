// "How complete is this answer": the judge checks the top-ranked photos (the head) and a random sample of the rest (the
// tail); an exact one-sided Clopper-Pearson upper bound on the tail's hit rate bounds the matches still hiding there.
// Port of audit.py (elusion test). The bound is relative to the AI judge's yes/no answers, not to a person's.
import Foundation

/// Regularized incomplete beta I_x(a, b) (continued fraction, modified Lentz; Numerical Recipes betacf).
func incBeta(_ x: Double, _ a: Double, _ b: Double) -> Double {
    if x <= 0 { return 0 }
    if x >= 1 { return 1 }
    let lbeta = lgamma(a + b) - lgamma(a) - lgamma(b)
    let front = exp(lbeta + a * log(x) + b * log(1 - x))
    if x > (a + 1) / (a + b + 2) { return 1 - incBeta(1 - x, b, a) }
    let tiny = 1e-300
    var c = 1.0, d = 1 - (a + b) * x / (a + 1)
    if abs(d) < tiny { d = tiny }
    d = 1 / d; var h = d
    for m in 1...500 {
        let m2 = Double(2 * m), md = Double(m)
        var aa = md * (b - md) * x / ((a + m2 - 1) * (a + m2))
        d = 1 + aa * d; if abs(d) < tiny { d = tiny }; c = 1 + aa / c; if abs(c) < tiny { c = tiny }
        d = 1 / d; h *= d * c
        aa = -(a + md) * (a + b + md) * x / ((a + m2) * (a + m2 + 1))
        d = 1 + aa * d; if abs(d) < tiny { d = tiny }; c = 1 + aa / c; if abs(c) < tiny { c = tiny }
        d = 1 / d; let del = d * c; h *= del
        if abs(del - 1) < 1e-15 { break }
    }
    return front * h / a
}

/// Inverse of I_x(a, b) in x (bisection to 1e-13): the beta distribution's quantile function.
func betaPPF(_ q: Double, _ a: Double, _ b: Double) -> Double {
    var lo = 0.0, hi = 1.0
    for _ in 0..<200 { let mid = (lo + hi) / 2; if incBeta(mid, a, b) < q { lo = mid } else { hi = mid }; if hi - lo < 1e-13 { break } }
    return (lo + hi) / 2
}

/// One-sided Clopper-Pearson upper bound on a binomial proportion.
public func cpUpper(hits: Int, n: Int, alpha: Double) -> Double {
    if n == 0 || hits >= n { return 1 }
    return betaPPF(1 - alpha, Double(hits + 1), Double(n - hits))
}

/// Judge calls needed so that zero hits in the sample certifies <= maxMissed matches hiding in the tail.
public func tailSampleSize(nTail: Int, maxMissed: Double, alpha: Double = 0.05) -> Int {
    if nTail <= 0 { return 0 }
    return min(nTail, Int(ceil(Double(nTail) * log(1 / alpha) / max(maxMissed, 1e-9))))
}

public struct Certificate: Equatable, Sendable {
    public let found: Int, nTail: Int, tailSampled: Int, tailHits: Int
    public let missedPoint: Double, missedUpper: Double, recallPoint: Double, recallLower: Double, alpha: Double
}

public func certify(found: Int, nTail: Int, tailLabels: [Bool], alpha: Double = 0.05) -> Certificate {
    let n = tailLabels.count, hits = tailLabels.filter { $0 }.count
    if nTail == 0 {
        return Certificate(found: found, nTail: 0, tailSampled: 0, tailHits: 0, missedPoint: 0, missedUpper: 0,
                           recallPoint: found > 0 ? 1 : .nan, recallLower: found > 0 ? 1 : .nan, alpha: alpha)
    }
    let up = Double(nTail) * cpUpper(hits: hits, n: n, alpha: alpha)
    let pt = n > 0 ? Double(nTail) * Double(hits) / Double(n) : .nan
    return Certificate(found: found, nTail: nTail, tailSampled: n, tailHits: hits, missedPoint: pt, missedUpper: up,
                       recallPoint: Double(found) + pt > 0 ? Double(found) / (Double(found) + pt) : .nan,
                       recallLower: Double(found) + up > 0 ? Double(found) / (Double(found) + up) : 0, alpha: alpha)
}

/// The album note in words a person can use (Reza 10-07: "the most useful number is: we are X% sure we found
/// everything" - which cannot be computed honestly; what can: what was checked, how many could hide in the rest, and
/// how well the judge does on what it checks). `testedFind`: the judge's measured share of real matches it keeps
/// (RESULTS 34: 0.93 pooled on real libraries, old sim; ~0.89 scaled to the real phone weights, RESULTS 38).
public func completenessNote(_ c: Certificate, inScope: Int, testedFind: String = "about 9 in 10") -> String {
    let judge = " In tests, the AI judge kept \(testedFind) of the real matches it looked at."
    if c.nTail == 0 { return "Checked all \(inScope.formatted()) photos that could match." + judge }
    let checked = max(0, inScope - c.nTail)
    let rest = c.nTail.formatted()
    let hide = c.missedUpper < 1 ? "Probably none are among the other \(rest) (95% sure)."
        : "Up to about \(Int(c.missedUpper.rounded(.up)).formatted()) more could be among the other \(rest) (95% sure); "
          + "\"Look at everything\" checks them."
    return "Checked the \(checked.formatted()) most likely of \(inScope.formatted()) photos. " + hide + judge
}

// MARK: - precision from a hand-checked sample
//
// THE PROBLEM THIS SOLVES (Reza, 10-11, after "photos of me" returned 41,541 photos): "yes look like me but
// obviously i cant look at all". A result set of tens of thousands cannot be audited by eye, and until it is
// audited nobody - him or me - knows whether the number means anything. The honest instrument is a random
// sample: check n of the N results by eye, and report the share that were right WITH its uncertainty, rather
// than a bare percentage. This is also the "confidence level" he asked the product to show.
//
// Random, not the top of the ranking: the best-scoring results are the easiest, so auditing those overstates
// precision. The sample must be drawn uniformly from everything the search returned.

/// One-sided Clopper-Pearson LOWER bound on a binomial proportion (the mirror of `cpUpper`).
public func cpLower(hits: Int, n: Int, alpha: Double) -> Double {
    if n == 0 || hits <= 0 { return 0 }
    return betaPPF(alpha, Double(hits), Double(n - hits + 1))
}

public struct PrecisionEstimate: Equatable, Sendable {
    /// How many of the sample were right, out of how many were checked, out of the whole result set.
    public let right: Int, checked: Int, found: Int
    /// right / checked, and the two-sided Clopper-Pearson interval at `alpha` (0.05 -> 95%).
    public let point: Double, lower: Double, upper: Double
    public let alpha: Double
    /// The point estimate applied to the whole set, rounded: "about 37,000 of the 41,541 are right".
    public var rightInSet: Int { Int((point * Double(found)).rounded()) }
}

/// `labels`: one true/false per sampled result, in any order. Exact (Clopper-Pearson) interval, so it stays
/// honest at 20 checks and at 0 or 100% right - where a normal approximation gives a zero-width interval.
public func precisionFromSample(labels: [Bool], found: Int, alpha: Double = 0.05) -> PrecisionEstimate {
    let n = labels.count, hits = labels.filter { $0 }.count
    guard n > 0 else {
        return PrecisionEstimate(right: 0, checked: 0, found: found, point: .nan, lower: 0, upper: 1, alpha: alpha)
    }
    return PrecisionEstimate(right: hits, checked: n, found: found, point: Double(hits) / Double(n),
                             lower: cpLower(hits: hits, n: n, alpha: alpha / 2),
                             upper: cpUpper(hits: hits, n: n, alpha: alpha / 2), alpha: alpha)
}

/// The same estimate in words, for the album screen.
public func precisionNote(_ p: PrecisionEstimate) -> String {
    guard p.checked > 0 else { return "Nothing checked by hand yet." }
    let pc = { (x: Double) in "\(Int((x * 100).rounded()))%" }
    return "You checked \(p.checked) of these at random and \(p.right) were right (\(pc(p.point))). "
        + "So between \(pc(p.lower)) and \(pc(p.upper)) of all \(p.found.formatted()) are likely right "
        + "- around \(p.rightInSet.formatted()) photos."
}

/// How many random results to check by hand for an interval no wider than `width` (at the worst case p = 0.5).
/// 20 checks -> about +-20 points; 50 -> about +-14; 100 -> about +-10. Deliberately small: the point is a
/// number Reza can actually produce in two minutes, not a labelling project.
public func precisionSampleSize(width: Double = 0.20, found: Int) -> Int {
    guard found > 0 else { return 0 }
    let n = Int(ceil(0.9604 / (width * width)))   // 1.96^2 * 0.25 / (width/2)^2
    return max(1, min(found, n))
}
