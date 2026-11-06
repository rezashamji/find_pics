// RESULTS 71 (cluster M102), turned into the two things the screen has to decide: WHAT THE QUICK ANSWER'S NOTE SAYS
// about how much it finds, and WHICH second button to offer.
//
// Why this exists at all. The judge-free answer is now the default (M85: a judge call is 2.4-3.2 s measured on Reza's
// phone, so ~65 of them is minutes), and RESULTS 71 measured what that costs in answers rather than in seconds:
// ~.06 F1 below judging everything on dev. But the ONE number hides the thing the person needs - it is close for
// objects, dates and places, and much worse for colours, person looks, multi-hop and vague requests. A single
// sentence ("the AI did not open any photo") is honest about the mechanism and silent about the size, so the note
// states the measured recall FOR THE KIND OF REQUEST THIS IS.
//
// Everything below is a measured table, not a model. No interpolation, no formula: the cluster reports recall at
// five coverage levels and eight request categories, and this file reads that table off. When the phone's coverage
// sits between two levels we quote the LOWER level (M102(2): "the nearest coverage level <= the phone's"), because
// recall at the phone's real coverage is at least that and the note must not overstate.
import Foundation

/// The eight request categories RESULTS 71 measures (dev table "NJ (t=.55) vs AJ's answer"), plus `all` for a
/// request whose shape says nothing - M102(2): "if unsure, use ALL".
public enum RequestKind: String, Sendable, CaseIterable {
    case objects, attributes, personLook, timePlace, multiHop, vagueOnePhoto, twoAlbums, followUps, all
}

/// Coverage = the share of the photos in scope that have a written facts line. Below `coverageFloor` the note says
/// counts only (M102(2) / D4): at Reza's 1.3% today the measured recall rows are all read off a library where
/// nearly nothing was described, and quoting them as "what this phone does" would be quoting the wrong experiment.
public let coverageFloor = 0.10

/// The coverage levels the cluster measured, ascending. A phone's coverage is quoted at the greatest level <= it.
public let measuredCoverages: [Double] = [0.10, 0.25, 0.50, 1.00]

/// dev, "COVERAGE: NJ recall vs AJ per category", MEAN PER REQUEST (the parenthesised number), rows 10/25/50/100%.
/// Mean per request, not pooled: one request = one answer screen, and pooling lets a single 4,000-photo request
/// speak for thirty small ones. The 100% row is D1's headline (objects .80, attributes .70, person+look .67,
/// time/place .83, multi-hop .54, vague .61, two albums .78, follow-ups .67).
/// TWO ROWS ARE NOT MONOTONIC in coverage - multi-hop falls .69 -> .54 and vague rises .44 -> .61 at 100% - and
/// that is the measurement, on 13 and 27 requests respectively. It is copied as measured rather than smoothed.
/// ALL is mean per request too, from the row the cluster added in M105(a) after MAC-M104 asked for it. I had used
/// their POOLED row as a stand-in and argued it could only understate; they checked and that was not quite true -
/// at 25% coverage pooled .672 is ABOVE mean .670, so the stand-in could have overstated by a hair. Measured row,
/// not a stand-in, is the only version of this that is safe to leave in.
let njRecallTable: [RequestKind: [Double]] = [
    .objects:       [0.72, 0.73, 0.76, 0.80],
    .attributes:    [0.62, 0.63, 0.65, 0.70],
    .personLook:    [0.54, 0.56, 0.61, 0.67],
    .timePlace:     [0.83, 0.83, 0.83, 0.83],
    .multiHop:      [0.69, 0.69, 0.69, 0.54],
    .vagueOnePhoto: [0.44, 0.44, 0.44, 0.61],
    .twoAlbums:     [0.73, 0.74, 0.78, 0.78],
    .followUps:     [0.59, 0.60, 0.62, 0.66],
    .all:           [0.662, 0.670, 0.692, 0.731],
]

/// The measured recall of the judge-free answer against judging everything, for this kind of request at this
/// coverage - or nil below `coverageFloor`, where the note stays counts-only.
public func njRecall(_ kind: RequestKind, coverage: Double) -> Double? {
    guard coverage >= coverageFloor, let row = njRecallTable[kind] else { return nil }
    var i = 0
    for (k, level) in measuredCoverages.enumerated() where coverage >= level { i = k }
    return row[i]
}

/// D5 (M102(3)): "verify the shown answer" is offered only where the cluster measured judge-free PRECISION below
/// .90, i.e. where the quick answer puts photos in that the judge would throw out. dev precision mean per request:
/// objects .75, attributes .67, person+look .58, multi-hop .54, vague .81, follow-ups below .90 -> offered;
/// time/place .91 and two albums .90 -> not offered, because there the wrong photos are rare and the button would
/// spend minutes confirming. ALL is offered (dev precision mean .748).
/// This corrects M97's premise, which was mine: I had told the screen the quick answer's weakness was only the
/// photos it MISSED. RESULTS 71 measures it missing ~27% AND including ~25% the judge rejects.
public func verifyShownOffered(_ kind: RequestKind) -> Bool {
    switch kind {
    case .timePlace, .twoAlbums: return false
    case .objects, .attributes, .personLook, .multiHop, .vagueOnePhoto, .followUps, .all: return true
    }
}

/// The note's sentence, in Reza's plain register (M102(2) wording), or nil below `coverageFloor`.
public func njRecallSentence(_ kind: RequestKind, coverage: Double) -> String? {
    guard let r = njRecall(kind, coverage: coverage) else { return nil }
    let tenths = Int((r * 10).rounded())
    if tenths >= 10 { return "Usually finds nearly all of what a full check finds for this kind of question." }
    return "Usually finds about \(tenths) in 10 of what a full check finds for this kind of question."
}

// MARK: - which kind a plan is

/// The judge-kind leaves (CascadeRoute.cascadeRouteKind) that make a request a PERSON+LOOK request. M102(2) puts
/// "photo look" here with the two person rules: RESULTS 71's category c is person-AND-look, so a sepia question and
/// a "is he wearing a hat" question are the same category to it.
let personLookLeafKinds: Set<String> = ["red-box person", "person attribute", "photo look"]

/// The judge-kind leaves that make it an ATTRIBUTES request. M102(2) names "colour / attribute leaves"; the other
/// four judge rules (count, text / sign, relation, small / background detail) all ask about a property of the scene
/// rather than naming a thing in it, so they are read as attribute leaves too. That reading is the CONSERVATIVE
/// one on purpose: attributes is measured lower than objects at every coverage level (.62 vs .72 at 10%), so a
/// leaf landing here understates the answer rather than flattering it.
let attributeLeafKinds: Set<String> = ["colour", "count", "text / sign", "relation", "small / background detail"]

/// Every category this request's shape could be. More than one can fit - a follow-up that asks for two albums -
/// and `requestKind` then takes the one with the LOWEST measured recall, because the note's job is not to flatter
/// the answer. "else objects" (M102(2)) is the floor: a question with no judge-kind leaf is an object/scene
/// question, which is what the millisecond scorer is measured best at.
public func requestKindCandidates(_ plan: Plan, isFollowUp: Bool) -> Set<RequestKind> {
    guard !plan.albums.isEmpty else { return [.all] }
    var out = Set<RequestKind>()
    if isFollowUp { out.insert(.followUps) }
    if plan.albums.count >= 2 { out.insert(.twoAlbums) }
    for a in plan.albums {
        // multi-hop = the plan chains steps (an anchor photo, an "until" step, a window around a match) rather
        // than filtering one set of photos.
        if a.anchor != nil || a.until != nil || a.window != nil { out.insert(.multiHop) }
        // vague, one photo = "find me the best picture of ...": one photo, chosen, with nothing said about what
        // makes it right. Measured worst of all eight at low coverage (.44).
        if a.want == "best", a.maxItems == 1 { out.insert(.vagueOnePhoto) }
        if a.dateFrom != nil || a.dateTo != nil || a.timePhrase != nil || a.timeOfDay != nil
            || !(a.place ?? "").isEmpty { out.insert(.timePlace) }
        for q in [a.judgeQuestion, a.filterQuestion, a.excludeQuestion, a.anchor?.judgeQuestion,
                  a.until?.judgeQuestion].compactMap({ $0 }) where !q.isEmpty {
            let kind = cascadeRouteKind(q, crop: (a.person ?? "").isEmpty ? "False" : "box")
            if personLookLeafKinds.contains(kind) { out.insert(.personLook) }
            if attributeLeafKinds.contains(kind) { out.insert(.attributes) }
            if kind == "trip-no-place" || cascadeLeafKind(q, crop: "False", exclude: false) == "trip-no-place" {
                out.insert(.timePlace)
            }
        }
        if !(a.person ?? "").isEmpty || !a.withPeople.isEmpty { out.insert(.personLook) }
    }
    if out.isEmpty { out.insert(.objects) }
    return out
}

/// The one category this request is quoted as, at this coverage: the candidate with the lowest measured recall.
/// Coverage is a parameter because the ordering is not fixed - multi-hop is above attributes at 10% coverage
/// (.69 vs .62) and below it at 100% (.54 vs .70) - so "the most cautious of these" depends on where the phone is.
public func requestKind(_ plan: Plan, isFollowUp: Bool, coverage: Double) -> RequestKind {
    requestKind(candidates: requestKindCandidates(plan, isFollowUp: isFollowUp), coverage: coverage)
}

/// The same choice from an already-computed candidate set. The search engine takes this route: the plan's shape is
/// known before the search starts, but COVERAGE is not - it is the share of the in-scope photos that turned out to
/// have a written line, which only the millisecond pass knows.
public func requestKind(candidates: Set<RequestKind>, coverage: Double) -> RequestKind {
    let at = max(coverage, measuredCoverages[0])            // order the candidates even below the note's floor
    return candidates.min {
        let a = njRecall($0, coverage: at) ?? 1, b = njRecall($1, coverage: at) ?? 1
        return a == b ? $0.rawValue < $1.rawValue : a < b
    } ?? .all
}
