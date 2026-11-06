// THE MILLISECOND SCORER BEHIND THE FAST ANSWER: a P for EVERY photo in scope, with no judge call.
//
// Policy V (RESULTS 65, M94(2)): a photo the night pass has described is scored by the towervec head on its written
// line; a photo with no line is scored by the vecB head on its image vector alone. The second half is the whole
// point. About 1% of Reza's library has a line today, so a fast answer that could only score described photos would
// be a fast answer about 1% of his photos - and the night pass will never be "finished" anyway, because he keeps
// taking photos. Policy V is what lets a search work while the writing is still going.
//
// CascadeSession pulls these scores one photo at a time (ms(photo, scorerQuestion)), and folding a head costs a
// matrix build, so they are computed up front instead: one folded head per LEAF question of the plan (a question
// like "food or drink from 2019 in Rome" has several), then one batched pass over the stored vectors.
// The pass itself is FindPicsCore.policyVScores, and it is there rather than here because it is TESTED there: it
// reads the vectors in 1,024-row blocks (on the phone they are a memory-mapped file, and all 148k rows at once is
// ~600 MB of an app that has about 1.1 GB), and a mistake in which block row belongs to which photo would give every
// photo a perfectly plausible score belonging to a different photo. Nothing on screen would look wrong.
import Foundation
@preconcurrency import FindPicsCore

struct MilliScorer: Sendable {
    /// scorer question -> P per photo, in the search's `scoped` order
    private let byQuestion: [String: [Float]]
    private let described: Set<Int>
    /// photos with no stored image vector at all: nothing read them, so nothing can score them (see `scorable`)
    private let unscorable: Set<Int>
    let scorable: Int

    var describedCount: Int { described.count }
    func has(_ photo: Int) -> Bool { described.contains(photo) }

    /// A leaf question we did not precompute would otherwise answer every photo from one lookup miss, so it is
    /// logged and answered 0.5 - no information - rather than 0, which reads as a confident no.
    func score(_ photo: Int, _ sq: String) -> Float {
        guard let row = byQuestion[sq] else { Judge.logMem("MILLI MISS \(sq)"); return 0.5 }
        return unscorable.contains(photo) ? 0 : row[photo]
    }

    /// The leaf scorer questions of a plan, in the order CascadeSession will ask them.
    static func leafQuestions(_ t: QuestionTree) -> [String] {
        switch t {
        case .leaf(let q): return [scorerQuestion(q)]
        case .meta(_, let rest): return rest.map(leafQuestions) ?? []
        case .parts(_, let ch): return ch.flatMap(leafQuestions)
        }
    }

    /// nil when the vecB head or the vectors are missing, i.e. when there is no millisecond answer to give.
    static func make(plan: QuestionTree, scoped: [String], index: PhotoIndex, embedder: Embedder) async -> MilliScorer? {
        let t0 = Date()
        guard let vecURL = Bundle.module.url(forResource: "vecB", withExtension: "fphead", subdirectory: "Models/facts"),
              let vecHead = try? ScorerHead(contentsOf: vecURL) else {
            Judge.logMem("MILLI no vecB head"); return nil
        }
        let twHead = Bundle.module.url(forResource: "towervec_head", withExtension: "fphead", subdirectory: "Models/facts")
            .flatMap { try? ScorerHead(contentsOf: $0) }
        let questions = Array(Set(leafQuestions(plan)))
        guard !questions.isEmpty else { return nil }

        let (units, unitItem, _) = await index.units(ids: scoped)
        guard units.count > 0, units.dim == vecHead.dImg else {
            Judge.logMem("MILLI vectors dim \(units.dim) vs head \(vecHead.dImg), \(units.count) units"); return nil
        }
        // one row per PHOTO: its first unit, which is what the food AUC .995 was measured on (a video's later frames
        // are not mixed in here - the facts line is written from one frame, so the two sides would not match)
        var firstUnit = [Int?](repeating: nil, count: scoped.count)
        for (u, item) in unitItem.enumerated() where firstUnit[item] == nil { firstUnit[item] = u }
        var itemRow = [Int](), unitRow = [Int](), unscorable = Set<Int>()
        for (item, u) in firstUnit.enumerated() {
            if let u { itemRow.append(item); unitRow.append(u) } else { unscorable.insert(item) }
        }

        let store = FactsStore()
        _ = await store.load()
        let factRows = await store.rows
        var described = Set<Int>()
        for (item, id) in scoped.enumerated() where factRows[id] != nil { described.insert(item) }
        // the facts question vector needs the writer's own text encoder (33 MB, NOT the 1.7 GB writer: loading that
        // here put the app at 4.60 GB resident and 1.07 GB free)
        var writer: FactsWriter? = nil
        if !described.isEmpty, twHead != nil {
            let w = FactsWriter()
            if (try? await w.loadEncoderOnly()) != nil { writer = w } else { Judge.logMem("MILLI no facts encoder") }
        }

        var byQuestion = [String: [Float]]()
        let sub = units.subset(unitRow)
        for sq in questions {
            guard let tv = try? embedder.vector(of: sq) else { continue }
            let qv = writer == nil ? nil : try? await writer!.questionVector(sq)
            let foldedT = (qv == nil || twHead == nil) ? nil : twHead!.fold(q: qv, t: tv)
            byQuestion[sq] = policyVScores(rows: sub, photoRow: itemRow, photos: scoped.count,
                                           towervec: foldedT, factsVector: { factRows[scoped[$0]]?.vector },
                                           vec: vecHead.fold(q: nil, t: tv))
        }
        await writer?.unload()
        // described is only what the TOWERVEC side actually used: with no towervec head or no encoder every photo is
        // scored by vecB, and saying "described" then would overstate what the answer rests on
        let usedFacts = (twHead != nil && writer != nil) ? described : []
        let s = MilliScorer(byQuestion: byQuestion, described: usedFacts, unscorable: unscorable,
                            scorable: itemRow.count)
        Judge.logMem("MILLI \(questions.count) leaf question(s), \(itemRow.count) of \(scoped.count) scorable, "
                     + "\(usedFacts.count) from written lines, \(Int(Date().timeIntervalSince(t0) * 1000)) ms")
        return s
    }
}
