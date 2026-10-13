import Foundation

/// "Is this a selfie?" as ONE DOT PRODUCT on the image vector the library already stores.
///
/// WHY THIS EXISTS (M46 -> M50, 10-11/12). A bare "selfies" used to go to the photo judge and never finished:
/// 18 minutes in it had not reached its first 150 photos. The front/back camera tag answers it for free - but the
/// tag is EXIF, EXIF needs the whole original, and ~170k of Reza's 187k originals live in iCloud, so only 579
/// photos carry it. My idea was to fall back to face geometry (a big face near the middle). The cluster measured
/// both on 17,594 Wikimedia Commons iPhone photos labelled front/back by their own EXIF, split by uploader:
///     logistic regression on the PE-Core-B-16 vector   held-out AUC 0.994
///     face geometry (area fraction, centring, counts)  held-out AUC 0.932, and 0.50 with no face in the photo
///     vector + geometry                                +0.000
/// So the vector alone wins and geometry adds nothing - and the centring half of my idea was already dead on
/// Reza's own photos (front 0.424 vs back 0.415 off-centre: people do not centre their own faces).
/// A dot product over 187k stored vectors is milliseconds, needs no image read, no download and no judge call.
public struct SelfieDetector: Sendable {
    public let w: [Float]
    public let b: Float

    public init(w: [Float], b: Float) { self.w = w; self.b = b }

    /// The log-odds that this photo is a selfie. `v`: the stored L2-normalised PE-Core-B-16 image vector.
    /// Positive means selfie-like; the threshold is picked per library (see `pickThreshold`).
    public func logit(_ v: UnsafeBufferPointer<Float>) -> Float {
        guard v.count == w.count else { return -.infinity }
        var s: Float = 0
        for i in 0..<w.count { s += w[i] * v[i] }
        return s + b
    }

    public func logit(_ v: [Float]) -> Float { v.withUnsafeBufferPointer { logit($0) } }
}

/// How well the detector does on photos whose camera tag we DO know, i.e. on this phone's own library.
public struct SelfieCheck: Equatable, Sendable {
    public let positives: Int, negatives: Int
    /// Area under the ROC curve: the chance a random front-camera photo scores above a random back-camera one.
    public let auc: Double
    public let recallAtHalf: Double, falsePositiveRateAtHalf: Double
    /// The lowest threshold whose selfies are >= 95% really front-camera, on THIS library, and what it recalls.
    public let threshold95: Float?, recallAt95: Double
}

/// `scores` paired with "was this photo front-camera", for the tagged photos only. The cluster's thresholds come
/// from Commons at Commons' prevalence and from 1,280 px thumbnails; the phone stores 448 px renditions, so the
/// threshold is re-picked here against the phone's own labels (M50 step 2).
public func checkSelfieDetector(_ labelled: [(score: Float, isFront: Bool)]) -> SelfieCheck {
    let pos = labelled.filter { $0.isFront }.map { $0.score }
    let neg = labelled.filter { !$0.isFront }.map { $0.score }
    guard !pos.isEmpty, !neg.isEmpty else {
        return SelfieCheck(positives: pos.count, negatives: neg.count, auc: .nan, recallAtHalf: .nan,
                           falsePositiveRateAtHalf: .nan, threshold95: nil, recallAt95: .nan)
    }
    // AUC by rank (ties count a half), which is exact and needs no bucketing.
    var wins = 0.0
    for p in pos { for n in neg { wins += p > n ? 1 : (p == n ? 0.5 : 0) } }
    let auc = wins / (Double(pos.count) * Double(neg.count))
    let recHalf = Double(pos.filter { $0 >= 0.5 }.count) / Double(pos.count)
    let fprHalf = Double(neg.filter { $0 >= 0.5 }.count) / Double(neg.count)
    // The lowest cut reaching 95% precision: walk every score as a candidate, highest first.
    var best: Float? = nil, bestRecall = Double.nan
    for t in Set(labelled.map { $0.score }).sorted(by: >) {
        let tp = pos.filter { $0 >= t }.count, fp = neg.filter { $0 >= t }.count
        if tp + fp > 0, Double(tp) / Double(tp + fp) >= 0.95 { best = t; bestRecall = Double(tp) / Double(pos.count) }
    }
    return SelfieCheck(positives: pos.count, negatives: neg.count, auc: auc, recallAtHalf: recHalf,
                       falsePositiveRateAtHalf: fprHalf, threshold95: best, recallAt95: bestRecall)
}

/// The cut that reaches `precision` on the tagged photos, and what it recalls there (M56 step 1).
/// nil when no cut reaches that precision at all.
public func selfieCut(_ labelled: [(score: Float, isFront: Bool)], precision: Double) -> (cut: Float, recall: Double)? {
    let pos = labelled.filter { $0.isFront }.map { $0.score }
    let neg = labelled.filter { !$0.isFront }.map { $0.score }
    guard !pos.isEmpty, !neg.isEmpty else { return nil }
    var best: (Float, Double)? = nil
    for t in Set(labelled.map { $0.score }).sorted(by: >) {
        let tp = pos.filter { $0 >= t }.count, fp = neg.filter { $0 >= t }.count
        if tp + fp > 0, Double(tp) / Double(tp + fp) >= precision {
            best = (t, Double(tp) / Double(pos.count))
        }
    }
    return best
}
