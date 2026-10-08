// Slow passes must look different from a stuck app (Reza 10-08: "i dont see anything being run on find pics" while the
// iCloud pass was moving 3-4 photos a minute against 39,005). The banner shows a bar, the measured rate and a rough
// time left, from what this pass has done since it started (no guess before there is data).
import Foundation

/// "about 3 h left (~12 per minute)" once >= 5 items and >= 60 s have been observed in this pass; nil before that.
public func progressEstimate(done: Int, total: Int, startDone: Int, elapsed: TimeInterval) -> String? {
    let n = done - startDone
    guard n >= 5, elapsed >= 60, total > done else { return nil }
    let perSec = Double(n) / elapsed
    let left = Double(total - done) / perSec
    let rate = perSec * 60 >= 1 ? "~\(Int((perSec * 60).rounded())) per minute" : "~\(Int((perSec * 3600).rounded())) per hour"
    let when: String
    switch left {
    case ..<90: when = "about a minute left"
    case ..<3600: when = "about \(Int((left / 60).rounded())) min left"
    case ..<(48 * 3600): when = "about \(Int((left / 3600).rounded())) h left"
    default: when = "about \(Int((left / 86400).rounded())) days left"
    }
    return "\(when) (\(rate))"
}

/// Photos and videos searchable so far, for a banner that only ever goes up (Reza 10-08 08:47: the per-launch count
/// "52" after "3,446" read as lost work; the pass's denominator is what actually shrinks). A pass with `passTotal`
/// items left to read at its start, `passDone` of them read: everything else in the library is already searchable.
public func searchableSoFar(libraryCount: Int, passTotal: Int, passDone: Int) -> Int {
    max(0, min(libraryCount, libraryCount - max(0, passTotal - passDone)))
}
