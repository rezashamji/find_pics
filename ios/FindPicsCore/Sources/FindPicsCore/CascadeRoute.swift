// RESULTS 68: the R59d CASCADE (RESULTS 59d, adopted shape; 61c/61d writer; policy V of RESULTS 65) as pure logic.
// Port of the measured harness eval/partial65/cascade65.py (variant r59d, K = 0) and eval/facts_scorer/route.py; checked
// call by call against recordings of that harness (Tests/.../Fixtures/cascade.json from eval/cascade_port/fixture.py;
// the full dev + TRAIN recordings with FP_CASCADE_FULL=<dir>).
// One engine judge call (question, crop flag, photos) is answered by its QuestionTree (trip metadata, OR / AND parts):
//   leaf of a SCORER kind -> the millisecond P (towervec on the facts line; vecB when the photo has no line = policy V)
//   leaf of a JUDGE kind (route.py: red-box person, count, text, photo look, colour, person word, relation, background
//   detail, "or"; a trip question whose place matches nothing; a plan's exclude question) -> the judge on the top
//   M = max(50, 5% of the leaf's scope) photos by millisecond P, one more block of M while >= 50% of the last block were
//   judge matches (P >= .7); every photo of the call at or above the last judged photo's millisecond P gets the judge's
//   P, the rest keep the millisecond P.
// The harness learns each leaf's SCOPE (every photo the engine asked that leaf about) from a first pass A; recordScope
// = that bookkeeping, answer = pass B (the measured cascade).
import Foundation

// MARK: - route kinds (= route.py RULES)

let cascadeColours = "red|orange|yellow|green|blue|purple|pink|magenta|teal|turquoise|brown|beige|grey|gray|black|white|golden|gold|silver|navy|violet|maroon|cyan|colou?r|colou?rful|neon|pastel"
let cascadeRules: [(String, String)] = [
    ("count", #"(?i)\b(two|three|four|five|six|seven|eight|nine|ten|several|group of|crowd|many|multiple|\d+|pair of|alone|only one|single|couple)\b"#),
    ("text / sign", #"(?i)\b(text|sign|signs|signage|written|writing|reads?|says|words?|letters|logo|menu|banner|label|poster|number plate)\b"#),
    ("photo look", #"(?i)\b(blurr?y|blurred|out of focus|motion blur|black[- ]and[- ]white|monochrome|sepia|overexposed|underexposed|washed out|close-?up|long exposure|silhouette|reflections?|aerial|panorama|bokeh|macro|wide|tilted|grainy|dark|bright|hazy|smog|foggy|fog|lit|lighting|light|lights|sunlight|sunset|night)\b"#),
    ("colour", #"(?i)\b("# + cascadeColours + #")\b"#),
    ("person attribute", #"(?i)\b(person|man|woman|men|women|boy|girl|child|children|kid|kids|people|someone|bride|groom|baby|he|she|guy|lady|player|players)\b"#),
    ("relation", #"(?i)\b(next to|beside|behind|in front of|on top of|underneath|under|above|below|between|near|nearby|against|leaning|surrounded by|attached to|hanging|on (his|her|their|its))\b"#),
    ("small / background detail", #"(?i)\b(background|in the distance|distant|far away|tiny|small|little|corner|edge|faint|partially|barely)\b"#),
    ("or-composition", #"(?i)\bor\b"#),
]

/// route.route: "" = scorer; else the name of the rule that sends the question to the judge. `crop` is the engine's crop
/// flag as Python prints it ("box" = red-box person crop; "False" / "True" otherwise).
public func cascadeRouteKind(_ question: String, crop: String) -> String {
    if crop == "box" || has(#"(?i)\bred box\b"#, question) { return "red-box person" }
    let q = subLit(tailPat, question, "")
    for (name, p) in cascadeRules where has(p, q) { return name }
    return ""
}

/// The kind of one LEAF of the cascade's tree (cascade.py decompose): route kind, else "trip-no-place" for a trip question
/// (whose place matched nothing, or it would not be a leaf), else "exclude" for a plan's exclude question, else "" (scorer).
public func cascadeLeafKind(_ leaf: String, crop: String, exclude: Bool) -> String {
    let k = cascadeRouteKind(leaf, crop: crop)
    if !k.isEmpty { return k }
    if has(tripPat, subLit(tailPat, leaf, "")) { return "trip-no-place" }
    return exclude ? "exclude" : ""
}

/// data.scorer_question (QNORM off, as in every shipped run): the question the millisecond scorers read.
public func scorerQuestion(_ q: String) -> String {
    var s = q
    if s.hasPrefix("PERSON["), let r = s.range(of: "::") { s = String(s[r.upperBound...]) }
    s = subLit(#"\b[Tt]he person in the red box\b"#, s, "a person")
    s = subLit(#"\s*\bin the red box\b"#, s, "")
    return strip(s)
}

/// route.budget: the judge budget for a leaf with n in-scope photos, min(n, max(50, ceil(5% of n))) (Python arithmetic).
public func judgeBudget(_ n: Int, frac: Double = 0.05, floor: Int = 50) -> Int {
    let k = Int(frac * Double(n) * 1000)          // int(frac * n * 1000) truncates toward zero
    let c = (k + 999) / 1000                       // -(-k // 1000) for k >= 0
    return min(n, max(floor, c))
}

/// The text towervec reads = this prefix + the writer line (eval/facts_scorer/data.py --meta=1): address parts without
/// digits, de-duplicated in order, and the capture month in UTC ("taken: November 2013. ").
public func factsPrefix(place: String?, taken: Date?) -> String {
    var parts = [String]()
    for p in (place ?? "").components(separatedBy: ",") {
        let t = strip(p)
        if t.isEmpty || t.unicodeScalars.contains(where: { CharacterSet.decimalDigits.contains($0) }) || parts.contains(t) { continue }
        parts.append(t)
    }
    var s = parts.isEmpty ? "" : "place: " + parts.joined(separator: ", ") + ". "
    if let d = taken {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let c = cal.dateComponents([.year, .month], from: d)
        let months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                      "November", "December"]
        s += "taken: \(months[c.month! - 1]) \(c.year!). "
    }
    return s
}

// MARK: - the session (one search)

/// One engine judge call. `photos` = global photo ids (the app's row in its library); `local[i]` = the row of photos[i] in
/// `places`, the place index of the photo set the engine is working on (the library or a subset of it: a place match
/// depends on every place text of that set); `faces` = the engine's face row per photo (-1 = none).
public struct CascadeCall: Sendable {
    public var question: String, crop: String, reference: Bool, exclude: Bool
    public var photos: [Int], local: [Int], faces: [Int]
    public var places: PlaceIndex?
    public init(question: String, crop: String, reference: Bool = false, exclude: Bool = false, photos: [Int], local: [Int],
                faces: [Int], places: PlaceIndex?) {
        self.question = question; self.crop = crop; self.reference = reference; self.exclude = exclude
        self.photos = photos; self.local = local; self.faces = faces; self.places = places
    }
}

public struct JudgeItem: Hashable, Sendable { public let photo: Int, face: Int, reference: Bool }
public struct JudgedKey: Hashable, Sendable { public let photo: Int; public let question: String; public let crop: String }

public final class CascadeSession: @unchecked Sendable {
    /// millisecond scores, policy V: hasFacts(photo) ? towervec(photo, scorer question) : vecB(photo, scorer question)
    let hasFacts: (Int) -> Bool
    let towervec: (Int, String) -> Float
    let vecB: (Int, String) -> Float
    /// the judge (cached by the caller or not): items, leaf question, crop -> P per item
    let judge: ([JudgeItem], String, String) async throws -> [Float]

    struct ScopeEntry { var ms: Float; var face: Int }
    struct LeafKey: Hashable { let q: String; let crop: String }
    /// pass A: each judge leaf's in-scope photos in first-seen order (a Python dict: an update keeps the position)
    var scopeOrder = [LeafKey: [Int]]()
    var scope = [LeafKey: [Int: ScopeEntry]]()
    /// pass B
    var thresholds = [LeafKey: Double?]()
    public private(set) var judged = Set<JudgedKey>()
    public private(set) var extensionJudged = [String: Int]()      // "q|crop" -> photos judged by budget + extension
    var answers = [String: Float]()                                 // judge answers this session holds
    /// judge-leaf evaluations in pass B, and how many had no pass-A scope (budget over that call's photos only)
    public private(set) var judgeLeafCalls = 0, fallbackLeafCalls = 0
    /// photos pass B asked each judge leaf about (diagnostic: how pass A's scope relates to what pass B asks)
    var askedInB = [LeafKey: Set<Int>]()

    public init(hasFacts: @escaping (Int) -> Bool, towervec: @escaping (Int, String) -> Float,
                vecB: @escaping (Int, String) -> Float,
                judge: @escaping ([JudgeItem], String, String) async throws -> [Float]) {
        self.hasFacts = hasFacts; self.towervec = towervec; self.vecB = vecB; self.judge = judge
    }

    public func ms(_ photo: Int, _ scorerQ: String) -> Float {
        hasFacts(photo) ? towervec(photo, scorerQ) : vecB(photo, scorerQ)
    }

    /// Start pass B (the harness resets the judged set and thresholds between passes; scopes stay).
    public func beginAnswers() {
        thresholds = [:]; judged = []; extensionJudged = [:]; askedInB = [:]
    }

    // (leaf question, kind, positions into the call) for every leaf of a tree (cascade.py leaves)
    private func leaves(_ t: QuestionTree, _ pos: [Int], _ call: CascadeCall) -> [(String, String, [Int])] {
        switch t {
        case .leaf(let q):
            return [(q, cascadeLeafKind(q, crop: call.crop, exclude: call.exclude), pos)]
        case .meta(let mk, let rest):
            guard let rest = rest else { return [] }
            return leaves(rest, pos.filter { mk[call.local[$0]] }, call)
        case .parts(_, let ch):
            return ch.flatMap { leaves($0, pos, call) }
        }
    }

    /// pass A bookkeeping (cascade.py record_d): every judge leaf's in-scope photos with their millisecond P and face row.
    public func recordScope(_ call: CascadeCall) {
        if call.reference || call.photos.isEmpty { return }
        let tree = QuestionTree.decompose(call.question, places: call.places)
        var faceOf = [Int: Int]()                     // dict(zip(rows, fr)): the last face row given for a photo wins
        for (p, f) in zip(call.photos, call.faces) { faceOf[p] = f }
        for (q, kind, pos) in leaves(tree, Array(call.photos.indices), call) where !kind.isEmpty && !pos.isEmpty {
            let key = LeafKey(q: q, crop: call.crop), sq = scorerQuestion(q)
            var d = scope[key] ?? [:], order = scopeOrder[key] ?? []
            for i in pos {
                let p = call.photos[i]
                if d[p] == nil { order.append(p) }
                d[p] = ScopeEntry(ms: ms(p, sq), face: faceOf[p]!)
            }
            scope[key] = d; scopeOrder[key] = order
        }
    }

    private func akey(_ p: Int, _ q: String, _ crop: String, _ face: Int, _ ref: Bool) -> String {
        "\(p)\u{1}\(q)\u{1}\(crop)\u{1}\(face)\u{1}\(ref)"
    }

    /// judge (once per photo, question, crop, face) and count every given photo as judged (cascade.py judge_ids)
    private func judgeItems(_ items: [JudgeItem], _ q: String, _ crop: String) async throws {
        var need = [JudgeItem](), seen = Set<String>()
        for it in items {
            let k = akey(it.photo, q, crop, it.face, it.reference)
            if answers[k] == nil && !seen.contains(k) { need.append(it); seen.insert(k) }
        }
        if !need.isEmpty {
            let p = try await judge(need, q, crop)
            for (it, x) in zip(need, p) { answers[akey(it.photo, q, crop, it.face, it.reference)] = x }
        }
        for it in items { judged.insert(JudgedKey(photo: it.photo, question: q, crop: crop)) }
    }

    private func answer(_ p: Int, _ q: String, _ crop: String, _ face: Int, _ ref: Bool = false) -> Float {
        answers[akey(p, q, crop, face, ref)]!
    }

    /// cascade.py leaf_threshold: the millisecond P above which this leaf's photos are judged (nil = no pass-A scope)
    private func threshold(_ q: String, _ crop: String) async throws -> Double? {
        let key = LeafKey(q: q, crop: crop)
        if let t = thresholds[key] { return t }
        guard let d = scope[key], !d.isEmpty, let ids = scopeOrder[key] else { thresholds[key] = .some(nil); return nil }
        // sorted(..., key=-ms): stable, ties keep first-seen order
        let order = ids.enumerated().sorted { a, b in
            let x = d[a.element]!.ms, y = d[b.element]!.ms
            return x != y ? x > y : a.offset < b.offset
        }.map { $0.element }
        let n = order.count, M = judgeBudget(n)
        var F = min(n, M)
        let item = { (p: Int) in JudgeItem(photo: p, face: d[p]!.face, reference: false) }
        try await judgeItems(order[0..<F].map(item), q, crop)
        while F < n {
            let blk = F >= M ? order[(F - M)..<F] : order[0..<F]
            let yes = blk.filter { Double(answer($0, q, crop, d[$0]!.face)) >= 0.7 }.count
            if Double(yes) / Double(blk.count) < 0.5 { break }
            let F2 = min(n, F + M)
            try await judgeItems(order[F..<F2].map(item), q, crop)
            F = F2
        }
        extensionJudged["\(q)|\(crop)"] = F
        let t: Double = F >= n ? -.infinity : Double(d[order[F - 1]]!.ms)
        thresholds[key] = .some(t)
        return t
    }

    /// kth largest (cascade.py kth over one call's millisecond P): the fallback threshold when a leaf has no pass-A scope
    private func kth(_ v: [Float], _ k: Int) -> Double {
        let s = v.sorted(by: >)
        return k >= s.count ? -.infinity : k > 0 ? Double(s[k - 1]) : .infinity
    }

    private func evaluate(_ t: QuestionTree, _ pos: [Int], _ call: CascadeCall) async throws -> [Float] {
        if pos.isEmpty { return [] }
        switch t {
        case .meta(let mk, let rest):
            guard let rest = rest else { return pos.map { mk[call.local[$0]] ? 0.99 : 0.01 } }
            var out = [Float](repeating: 0.01, count: pos.count)
            let at = pos.indices.filter { mk[call.local[pos[$0]]] }
            if !at.isEmpty {
                let p = try await evaluate(rest, at.map { pos[$0] }, call)
                for (k, i) in at.enumerated() { out[i] = p[k] }
            }
            return out
        case .parts(let op, let children):
            var acc: [Float]? = nil
            for ch in children {
                let p = try await evaluate(ch, pos, call)
                acc = acc.map { a in zip(a, p).map { op == .or ? max($0, $1) : min($0, $1) } } ?? p
            }
            return acc ?? []
        case .leaf(let q):
            let kind = cascadeLeafKind(q, crop: call.crop, exclude: call.exclude), sq = scorerQuestion(q)
            let pt = pos.map { ms(call.photos[$0], sq) }
            if kind.isEmpty { return pt }
            var tj = try await threshold(q, call.crop)
            judgeLeafCalls += 1
            askedInB[LeafKey(q: q, crop: call.crop), default: []].formUnion(pos.map { call.photos[$0] })
            if tj == nil { tj = kth(pt, judgeBudget(pt.count)); fallbackLeafCalls += 1 }
            let keep = pt.indices.filter { Double(pt[$0]) >= tj! }
            var out = pt
            if !keep.isEmpty {
                try await judgeItems(keep.map { JudgeItem(photo: call.photos[pos[$0]], face: call.faces[pos[$0]], reference: false) }, q, call.crop)
                for j in keep { out[j] = answer(call.photos[pos[j]], q, call.crop, call.faces[pos[j]]) }
            }
            return out
        }
    }

    /// pass B: the cascade's P for every photo of one engine judge call (cascade.py judge_rows, mode B, variant r59d).
    public func answer(_ call: CascadeCall) async throws -> [Float] {
        if call.photos.isEmpty { return [] }
        if call.reference {     // reference-photo comparisons: the whole question, judge kinds judged on every photo
            let kind = cascadeRouteKind(call.question, crop: call.crop)
            if kind.isEmpty { let sq = scorerQuestion(call.question); return call.photos.map { ms($0, sq) } }
            let items = zip(call.photos, call.faces).map { JudgeItem(photo: $0, face: $1, reference: true) }
            try await judgeItems(items, call.question, call.crop)
            return items.map { answer($0.photo, call.question, call.crop, $0.face, true) }
        }
        let tree = QuestionTree.decompose(call.question, places: call.places)
        return try await evaluate(tree, Array(call.photos.indices), call)
    }
}
