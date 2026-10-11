// "Are the vectors enough for the first screen?" (RESULTS 45, MAC_INBOX M40; 10-11).
//
// Every photo already has a PE-Core-B-16 image vector, so ranking a search by it takes milliseconds. The photo judge
// then costs ~2-4 s per photo on the phone (0.3 photos/s once hot). RESULTS 45 measured, on 216 searches over 20 real
// Flickr libraries, when the vector ranking's top 20 is already what the judge would have put there, and found one
// signal the phone can read before any judge call: the 20th-best fast score itself. If even the 20th photo is very
// similar to the look text, the library holds many clear matches and the top of the ranking is right; if not (few real
// matches, or a look the vectors express badly), the first screen needs the judge.
//
// Only searches of the kind measured may skip the judge: a plain look with at most the object / subject question.
// Person, selfie, moment, exclude / filter and look-less searches always go to the judge (vectors cannot express them,
// and none were measured).
import Foundation

public enum FirstScreenSource: String, Sendable, Equatable {
    /// Show the vector ranking's top photos at once as the first screen ("likely matches"); the judge checks them next.
    case vectors
    /// The first screen should wait for judge-checked photos (vector results may be shown, marked as unchecked).
    case judge
}

public struct FirstScreenDecision: Equatable, Sendable {
    public let source: FirstScreenSource
    /// Why, in a few words, for the device log ("kth 0.2214 >= 0.2100", "person search", ...).
    public let reason: String
    /// The `screen`-th best fast score of the in-scope photos (nil when the search kind already decided).
    public let kthScore: Float?
}

/// Photos on the first screen the rule speaks for.
public let vectorFirstScreen = 20
/// PE-Core-B-16 look score (mean look similarity - mean avoid similarity, `lookScores`) that the 20th-best in-scope photo
/// must reach for the vector top 20 to stand on its own. RESULTS 45: fitted on all 20 libraries with the "safe" rule
/// (the lowest threshold whose training searches above it have >= 90% of their top 20 judge-yes). Leave-one-library-out
/// (thresholds 0.2067-0.2083): 32 of 216 searches go to vectors, 28 of them truly enough, judge-relative precision of
/// their top 20 0.958; eye on the audited libraries 78/80, the same as the judge-verified screens (78/80).
public let vectorEnoughScore: Float = 0.2074
/// Per photo: a vector top-20 photo whose own score reaches this is shown at once as "likely" (RESULTS 45: lowest
/// threshold with >= 90% judge-yes above it, all 20 libraries; leave-one-library-out 0.2000-0.2054 -> judge-relative
/// precision 1,409/1,575 = 0.89, eye on the audited libraries 154/172 = 0.90).
public let likelyAtOnceScore: Float = 0.2036

/// nil when the album is of a kind the rule was measured on; else the reason it must go to the judge.
public func vectorFirstIneligible(_ a: Album) -> String? {
    if a.looks.isEmpty { return "no look to rank by" }
    if a.person != nil || !a.withPeople.isEmpty { return "person search" }
    if a.camera == "front" { return "selfie search" }
    if a.anchor != nil || a.until != nil || a.window != nil { return "moment search" }
    if a.excludeQuestion != nil || a.filterQuestion != nil { return "extra question" }
    if let q = a.judgeQuestion, !isSubjectPhotoQuestion(q), !isAnywhereObjectQuestion(q) { return "other question form" }
    return nil
}

/// The planner's object form "Is there (a real) X anywhere in this photo ...?" (RESULTS 26 fix 1).
public func isAnywhereObjectQuestion(_ q: String) -> Bool {
    q.range(of: #"^\s*is\s+there\s+.+\banywhere\b"#, options: [.regularExpression, .caseInsensitive]) != nil
}

/// The k-th largest value (1-based) of `xs`, or nil when there are fewer than k. O(n * k) worst case, O(n) typical.
public func kthLargest(_ xs: [Float], _ k: Int) -> Float? {
    guard k > 0, xs.count >= k else { return nil }
    var top = [Float](); top.reserveCapacity(k)            // ascending; top[0] = the current k-th largest
    for x in xs where !x.isNaN {
        if top.count < k {
            top.insert(x, at: top.firstIndex(where: { $0 > x }) ?? top.count)
        } else if x > top[0] {
            top.removeFirst()
            top.insert(x, at: top.firstIndex(where: { $0 > x }) ?? top.count)
        }
    }
    return top.count == k ? top[0] : nil
}

/// The decision, from the plan and the in-scope fast scores only (no judge call; milliseconds for 187k photos).
/// `scores`: lookScores of the in-scope photos (the same numbers the fast stage ranks by).
public func firstScreenDecision(album: Album, scores: [Float], screen: Int = vectorFirstScreen,
                                threshold: Float = vectorEnoughScore) -> FirstScreenDecision {
    if let why = vectorFirstIneligible(album) { return FirstScreenDecision(source: .judge, reason: why, kthScore: nil) }
    guard let kth = kthLargest(scores, screen) else {
        return FirstScreenDecision(source: .judge, reason: "fewer than \(screen) photos in scope", kthScore: nil)
    }
    let ok = kth >= threshold
    return FirstScreenDecision(source: ok ? .vectors : .judge,
                               reason: "score #\(screen) " + String(format: "%.4f", Double(kth)) + (ok ? " >= " : " < ")
                                   + String(format: "%.4f", Double(threshold)),
                               kthScore: kth)
}

/// The same threshold per photo: of the vector top `screen`, the photos whose own fast score reaches `threshold`, best
/// first (indices into `scores`). These can be shown at once as "likely" (RESULTS 45 per-photo rows); the judge checks
/// them and the rest of the head next. Empty for a search kind the rule was not measured on.
public func likelyAtOnce(album: Album, scores: [Float], screen: Int = vectorFirstScreen,
                         threshold: Float = likelyAtOnceScore) -> [Int] {
    if vectorFirstIneligible(album) != nil { return [] }
    let idx = scores.indices.filter { !scores[$0].isNaN && scores[$0] >= threshold }
    return Array(idx.sorted { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }.prefix(screen))
}
