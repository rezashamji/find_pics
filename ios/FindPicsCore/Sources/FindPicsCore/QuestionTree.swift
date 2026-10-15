// Judge questions that are COMPOSITIONS, answered by parts (RESULTS 59d, adopted 10-15). Port of findpics.compose (the
// pre-registered R59d rules of eval/facts_scorer/route.py); checked against the Python on 660 benchmark questions with
// their libraries' place texts (Tests/.../Fixtures/compose.json).
//   (a) a trip / place question ("Is this a photo taken during a trip to Puerto Rico?", "... with a sepia tone taken in
//       Spain?"): the place comes from the photo's own place text (accent-, case- and spelling-tolerant match against the
//       address tokens), never from the pixels; what is left of the question after the place phrase (if anything but
//       frame / trip-kind / date words) is judged on the photos at that place; elsewhere P = .01.
//   (b) "A or B" is judged as A and as B (P = max); "A and also B" / two questions at once as both (P = min).
// Complex bench dev, all-judge stack (cached 9B): ALL +.011 [+.001, +.022] (a +.011, d +.033, g +.024, h +.030).
// The app's judge call site wraps the plain judge with `QuestionTree.judge` (MAC_INBOX M74).
import Foundation

// MARK: - regex helpers with positions (UTF-16 offsets, all slicing through NSString)

private struct RM { let range: NSRange; let groups: [NSRange] }

private func rx(_ pattern: String, _ caseless: Bool) -> String { caseless ? "(?i)" + pattern : pattern }

private func finds(_ pattern: String, _ s: String, caseless: Bool = false) -> [RM] {
    let ns = s as NSString
    return regex(rx(pattern, caseless)).matches(in: s, range: NSRange(location: 0, length: ns.length)).map { m in
        RM(range: m.range, groups: (0..<m.numberOfRanges).map { m.range(at: $0) })
    }
}

private func slice(_ s: String, _ a: Int, _ b: Int) -> String {
    let ns = s as NSString
    let lo = max(0, min(a, ns.length)), hi = max(lo, min(b, ns.length))
    return ns.substring(with: NSRange(location: lo, length: hi - lo))
}

private func len16(_ s: String) -> Int { (s as NSString).length }

private func pyWords(_ s: String) -> [String] { s.split(whereSeparator: { $0.isWhitespace }).map(String.init) }

private func rstripChars(_ s: String, _ chars: String) -> String {
    var t = s
    while let c = t.last, chars.contains(c) { t.removeLast() }
    return t
}

private func cap(_ s: String) -> String { s.isEmpty ? s : s.prefix(1).uppercased() + s.dropFirst() }

// MARK: - constants (same as compose.py)

let tailPat = #"(?i)\s*\(not a drawing, painting, statue, toy, model or picture of one\)"#
let tailText = " (not a drawing, painting, statue, toy, model or picture of one)"
let tripPat = #"(?i)\b(trips?|vacation|holiday|journey|tour|visit|visiting|travel\w*)\b|\btaken (in|at|during|on)\b|\b(photo|picture|shot) from\b|\bduring (a|an|the|our|my)\b"#
let candPat = #"(?<![\w])([A-Z][\w'’.\-]*(?:\s+(?:(?:of|and|or|de|del|la|le|el|du|&)\s+)?[A-Z][\w'’.\-]*)*)"#
let candSkip: Set<String> = ["Is", "Was", "Are", "Were", "Does", "Do", "Did", "Has", "Have", "I", "A", "An", "The", "This", "That"]
let featureWords: Set<String> = ["airport", "bay", "beach", "beaches", "bridge", "canyon", "castle", "cathedral", "cave", "caves",
    "church", "coast", "desert", "fall", "falls", "forest", "fort", "garden", "gardens", "glacier", "harbor", "harbour", "hill",
    "hills", "island", "islands", "isle", "lake", "lakes", "market", "mount", "mountain", "mountains", "mt", "museum", "national",
    "old", "palace", "park", "peak", "port", "reef", "river", "ruins", "springs", "square", "station", "street", "temple", "tower",
    "trail", "valley", "volcano", "waterfall", "waterfalls", "zoo"]
let frameWords: Set<String> = ["a", "an", "and", "any", "anywhere", "are", "area", "around", "at", "did", "do", "does", "during",
    "excursion", "excursions", "featuring", "from", "getaway", "hike", "hikes", "hiking", "holiday", "holidays", "honeymoon", "i",
    "image", "images", "in", "is", "it", "journey", "my", "near", "of", "on", "our", "outing", "photo", "photos", "picture",
    "pictures", "place", "real", "region", "road", "s", "shot", "shots", "show", "showing", "shows", "sightseeing", "somewhere",
    "take", "taken", "that", "the", "there", "these", "this", "those", "to", "tour", "travel", "traveling", "travelling", "travels",
    "trip", "trips", "us", "vacation", "vacations", "video", "visible", "visit", "visited", "visiting", "was", "we", "were"]
let timeWordPat = #"(?i)^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december|spring|summer|fall|autumn|winter|early|late|mid|(?:19|20)\d\d|\d{1,2}(?:st|nd|rd|th)?|\d{4}s|weekend|week|month|year)$"#
let detWords: Set<String> = ["a", "an", "the", "any", "some", "its", "his", "her", "their", "my", "our"]
let funcWords: Set<String> = ["a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into", "is", "of", "on", "onto", "or",
    "taken", "than", "the", "to", "with"]
let adjWords: Set<String> = ["beige", "big", "black", "blue", "bright", "brown", "cyan", "dark", "giant", "gold", "golden", "gray",
    "green", "grey", "huge", "large", "light", "little", "long", "magenta", "maroon", "metal", "mini", "miniature", "navy", "neon",
    "new", "old", "orange", "pale", "pastel", "pink", "plain", "plastic", "purple", "red", "short", "silver", "small", "spotted",
    "striped", "tall", "teal", "tiny", "turquoise", "violet", "white", "wooden", "yellow", "young"]
let auxAlt = "(?:is|are|was|were|does|do|did|has|have|can|could)"
let negWords: Set<String> = ["not", "no", "none", "non", "neither", "nor", "else", "otherwise", "something", "anything"]
let openerPats = [
    #"^(?:is|are|was|were) there (?:(?:any|some) )?"#,
    #"^(?:is|was) this (?:an? )?(?:close-up |close up )?(?:photo|picture|image|shot|video|selfie|snapshot)(?: taken)? (?:of|showing|with|featuring) "#,
    #"^(?:is|was|are|were) (?:the|this|these) (?:person|people|photo|image|picture)(?: in (?:the|this) (?:photo|picture|image|red box))? "#,
    #"^(?:is|are|was|were|does|do|did|has|have|can) (?:this|these|it|the|there) "#,
    #"^\w+ "#]
let termPat = #",|\s*§|\s+(?:anywhere|visible|clearly|taken)\b|\s+in (?:this|the) (?:photo|image|picture|video)\b|\s+(?:in|at|from|near) (?:the )?(?=[A-Z])"#

// MARK: - place matching

/// accents and case folded, everything but a-z / 0-9 -> one space ("León" -> "leon", not "le n"). = compose.fold
public func foldText(_ s: String) -> String {
    let d = s.decomposedStringWithCompatibilityMapping
    var kept = String.UnicodeScalarView()
    for u in d.unicodeScalars where u.properties.canonicalCombiningClass == .notReordered { kept.append(u) }
    return sub("[^a-z0-9]+", " ", String(kept).lowercased()).trimmingCharacters(in: .whitespaces)
}

/// Levenshtein distance with early exit (> cap -> cap + 1). = compose.lev
public func editDistance(_ a: String, _ b: String, cap: Int = 2) -> Int {
    let A = Array(a.unicodeScalars), B = Array(b.unicodeScalars)
    if abs(A.count - B.count) > cap { return cap + 1 }
    var prev = Array(0...B.count)
    for i in 1...max(1, A.count) where !A.isEmpty {
        var cur = [i] + [Int](repeating: 0, count: B.count)
        for j in stride(from: 1, through: B.count, by: 1) {
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (A[i - 1] != B[j - 1] ? 1 : 0))
        }
        if cur.min()! > cap { return cap + 1 }
        prev = cur
    }
    return prev[B.count]
}

/// One library's place texts as token sets. = compose.PlaceIndex
public final class PlaceIndex: @unchecked Sendable {
    let toks: [Set<String>]
    let vocab: [String]
    private var near = [String: Set<String>]()
    private var memo = [String: [Bool]]()
    private let lock = NSLock()

    public init(_ placeTexts: [String?]) {
        toks = placeTexts.map { Set(pyWords(foldText($0 ?? ""))) }
        vocab = Array(toks.reduce(into: Set<String>()) { $0.formUnion($1) }).sorted()
    }

    func maskWords(_ words: [String], _ fuzzy: Bool) -> [Bool] {
        var m = [Bool](repeating: true, count: toks.count)
        for w in words {
            var alts: Set<String> = [w]
            if fuzzy && w.count >= 6 {
                if near[w] == nil { near[w] = Set(vocab.filter { $0.count >= 6 && editDistance(w, $0) <= 2 }) }
                alts.formUnion(near[w]!)
            }
            for k in 0..<toks.count where m[k] { m[k] = !alts.isDisjoint(with: toks[k]) }
        }
        return m
    }

    public func mask(_ name: String) -> [Bool] {
        lock.lock(); defer { lock.unlock() }
        return maskLocked(name)
    }

    private func maskLocked(_ name: String) -> [Bool] {
        if let m = memo[name] { return m }
        let m = compute(name)
        memo[name] = m
        return m
    }

    private func compute(_ name: String) -> [Bool] {
        let zeros = [Bool](repeating: false, count: toks.count)
        let parts = subLit(#"(?i)\s+(?:and|or)\s+|\s*&\s*"#, name, "\u{1}").split(separator: "\u{1}", omittingEmptySubsequences: false)
            .map(String.init).filter { !strip($0).isEmpty }
        if parts.count > 1 {
            var m = zeros
            for p in parts { let pm = maskLocked(p); for k in 0..<m.count where pm[k] { m[k] = true } }
            return m
        }
        let words = pyWords(foldText(name)).filter { $0.count > 1 }
        if words.isEmpty { return zeros }
        let core = words.filter { !placeKind.contains($0) }
        let core2 = core.filter { !featureWords.contains($0) }
        // fuzzy only for spelling variants: not when a plain word of a multiword name matches exactly on its own
        let solo = words.count > 1 && core2.contains { maskWords([$0], false).contains(true) }
        let fuzzies: [Bool] = solo ? [false] : [false, true]
        for fuzzy in fuzzies {
            for ws in [words] + (!core.isEmpty && core.count < words.count ? [core] : []) {
                var m = maskWords(ws, fuzzy)
                if !fuzzy && ws.count > 1 {
                    let joined = ws.joined()
                    for k in 0..<m.count where toks[k].contains(joined) { m[k] = true }
                }
                if m.contains(true) { return m }
            }
        }
        if !core2.isEmpty && core2.count < core.count {
            for fuzzy in fuzzies {
                let m = maskWords(core2, fuzzy)
                if m.contains(true) { return m }
            }
        }
        return zeros
    }
}

// MARK: - question text rules

public func placeCandidates(_ question: String) -> [String] {
    let q = subLit(tailPat, question, "")
    var out = [String]()
    for m in finds(candPat, q) {
        var words = pyWords((q as NSString).substring(with: m.groups[1]))
        while let f = words.first, candSkip.contains(f) { words.removeFirst() }
        if !words.isEmpty { out.append(rstripChars(words.joined(separator: " "), ".?!,")) }
    }
    return out.filter { !$0.isEmpty }
}

/// The question with the place phrase removed ("" = nothing left beyond frame / trip-kind / date words). = compose.remainder
public func questionRemainder(_ question: String, _ name: String) -> String {
    let tail = has(tailPat, question)
    var q = strip(subLit(tailPat, question, ""))
    let n = esc(name)
    q = subLit(#"(?i),?\s*\band\s+(?:was|is|were)\s+(?:this|the)\s+(?:photo|picture|image|video)\s+taken\s+(?:at|in|on|near)\s+(?:(?:a|an|the)\s+)?"# + n + #"(?![\w])"#, q, "")
    q = subLit(#"(?i)\s*\b(?:(?:in|at|to|from|on|near|of|around|across|through|visiting)\s+)?(?:(?:a|an|the)\s+)?"# + n + #"(?![\w])"#, q, "")
    q = subLit(#"(?i)\s*\b(?:(?:taken|shot)\s+)?(?:during|on|from|in)\s+(?:a|an|the|our|my)\s+(?:trips?|vacation|holiday|journey|tour|visit|travels?)\b"#, q, "")
    q = subLit(#"(?i)\s+(?:taken|shot)\s*(?=[?,]|$)"#, q, "")
    q = strip(subFn(#"\s+([?,.])"#, subLit(#"\s+"#, q, " ")) { $0.group(1)! })
    q = subLit(#",\s*\?$"#, q, "?")
    let words = pyWords(foldText(q)).filter { !frameWords.contains($0) && !has(timeWordPat, $0) }
    if words.isEmpty { return "" }
    q = strip(rstripChars(q, "?"))
    return q + (tail ? tailText : "") + "?"
}

/// start/end (UTF-16) of each match of pat outside parentheses (case-insensitive, as compose._top_level)
private func topLevel(_ body: String, _ pat: String) -> [(Int, Int)] {
    var marks = [Int: (Int, Int)]()
    for m in finds(pat, body, caseless: true) { marks[m.range.location] = (m.range.location, m.range.location + m.range.length) }
    var out = [(Int, Int)](), depth = 0
    let ns = body as NSString
    for i in 0..<ns.length {
        let ch = ns.character(at: i)
        if ch == 40 { depth += 1 } else if ch == 41 { depth = max(0, depth - 1) } else if depth == 0, let mk = marks[i] { out.append(mk) }
    }
    return out
}

private func fixArticle(_ pre: String, _ alt: String) -> String {
    guard !alt.isEmpty, let m = finds(#"(?i)\b(an?) $"#, pre).first else { return pre }
    let g = (pre as NSString).substring(with: m.groups[1])
    let art = "aeiou".contains(alt.prefix(1).lowercased()) ? "an" : "a"
    return slice(pre, 0, m.groups[1].location) + (g == g.lowercased() ? art : cap(art)) + " "
}

/// "PRE X <conj> Y [<conj> Z] POST" -> [PRE X POST, PRE Y POST, ...]; nil = no safe split. = compose._word_split
private func wordSplit(_ body: String, _ conj: String) -> [String]? {
    let ors = topLevel(body, #"\s+"# + conj + #"\s+"#)
    guard !ors.isEmpty else { return nil }
    var oEnd = 0
    for p in openerPats { if let m = finds(p, body, caseless: true).first { oEnd = m.range.location + m.range.length; break } }
    if ors[0].0 < oEnd { return nil }
    let last = slice(body, ors[ors.count - 1].1, len16(body))
    let cut = finds(termPat, last).first?.range.location ?? len16(last)
    let post = slice(last, cut, len16(last))
    var ys = [String]()
    for k in 0..<(ors.count - 1) { ys.append(slice(body, ors[k].1, ors[k + 1].0)) }
    ys.append(slice(last, 0, cut))
    let region = slice(body, oEnd, ors[0].0)
    let dets = finds(#"\b(\w+)\s+"#, region).filter { detWords.contains(((region as NSString).substring(with: $0.groups[1])).lowercased()) }
    let xs: Int
    if let y0 = pyWords(ys[0]).first, detWords.contains(y0.lowercased()) {
        xs = oEnd + (dets.first?.range.location ?? 0)
    } else {
        xs = oEnd + (dets.last.map { $0.range.location + $0.range.length } ?? 0)
    }
    let pre = slice(body, 0, xs), x = slice(body, xs, ors[0].0)
    var alts = [x] + ys
    if alts.contains(where: { strip($0).isEmpty || funcWords.contains(pyWords($0).last!.lowercased()) }) || pyWords(x).count > 8 { return nil }
    // "... in real life or not?" is ONE yes/no question (product guard, 10-15; = compose.NEG)
    if ys.contains(where: { negWords.contains(pyWords($0)[0].lowercased()) }) { return nil }
    let y0 = pyWords(ys[0])[0].lowercased(), x0 = pyWords(x)[0].lowercased()
    if y0.hasSuffix("ing") && !x0.hasSuffix("ing") && !detWords.contains(x0) { return nil }
    let head = pyWords(ys[ys.count - 1])
    if head.count >= 2 && alts.dropLast().allSatisfy({ pyWords($0).count == 1 && adjWords.contains($0.lowercased()) }) {
        alts = alts.dropLast().map { $0 + " " + head.dropFirst().joined(separator: " ") } + [ys[ys.count - 1]]
    }
    return alts.map { strip(fixArticle(pre, $0) + $0 + post) }
}

public enum Compose: String, Sendable { case or, and }

/// (op, parts) with op .or (P = max) or .and (P = min); nil = not a composition we split. = compose.split_parts
public func splitParts(_ question: String) -> (Compose, [String])? {
    let raw = strip(question)
    var sents = [String](), s0 = 0
    for m in finds(#"\?\s+(?=[A-Z])"#, raw) { sents.append(slice(raw, s0, m.range.location)); s0 = m.range.location + m.range.length }
    sents.append(slice(raw, s0, len16(raw)))
    sents = sents.map(strip).filter { !$0.isEmpty }
    if sents.count > 1 { return (.and, sents.map { rstripChars($0, " ?") + "?" }) }
    let r0 = rstripChars(raw, " ?")
    let cl = topLevel(r0, #",\s+or\s+(?="# + auxAlt + #"\b)"#)
    if !cl.isEmpty {
        var pieces = [String](), a0 = 0
        for (a, b) in cl { pieces.append(slice(r0, a0, a)); a0 = b }
        pieces.append(slice(r0, a0, len16(r0)))
        return (.or, pieces.map { cap(strip($0)) + "?" })
    }
    var body = raw
    if let m = finds(tailPat, raw).first {      // the tail stays where it was, as one placeholder (count=1)
        body = slice(raw, 0, m.range.location) + " §" + slice(raw, m.range.location + m.range.length, len16(raw))
    }
    body = rstripChars(strip(body), " ?")
    let end: (String) -> String = { s in subLit(#"\s*§"#, rstripChars(s, " ?."), tailText) + "?" }
    if has(#"(?i)\band also\b"#, body) {
        guard let p = wordSplit(body, "and also") else { return nil }
        return (.and, p.map(end))
    }
    if !topLevel(body, #"\s+or\s+"#).isEmpty {
        guard let p = wordSplit(subLit(#",\s+or\s+"#, body, " or "), "or") else { return nil }
        return (.or, p.map(end))
    }
    return nil
}

// MARK: - the tree

public indirect enum QuestionTree: Equatable, Sendable {
    case leaf(String)
    case meta([Bool], QuestionTree?)
    case parts(Compose, [QuestionTree])

    /// = compose.decompose
    public static func decompose(_ question: String, places: PlaceIndex?, depth: Int = 0, metaOK: Bool = true) -> QuestionTree {
        if metaOK, let PI = places, has(tripPat, subLit(tailPat, question, "")) {
            for c in placeCandidates(question) {
                let mk = PI.mask(c)
                if mk.contains(true) {
                    let rem = questionRemainder(question, c)
                    return .meta(mk, rem.isEmpty ? nil : decompose(rem, places: places, depth: depth + 1, metaOK: false))
                }
            }
        }
        if depth < 4, let sp = splitParts(question) {
            return .parts(sp.0, sp.1.map { decompose($0, places: places, depth: depth + 1, metaOK: metaOK) })
        }
        return .leaf(question)
    }

    /// P per item: metadata .99 / .01, a remainder judged only at the place (.01 elsewhere), OR = max, AND = min.
    /// `items` index the library (and the masks); judgeLeaf(items, question) answers a plain question.
    public func evaluate(_ items: [Int], _ judgeLeaf: ([Int], String) async throws -> [Float]) async rethrows -> [Float] {
        if items.isEmpty { return [] }
        switch self {
        case .leaf(let q):
            return try await judgeLeaf(items, q)
        case .meta(let mk, let rest):
            guard let rest = rest else { return items.map { mk[$0] ? 0.99 : 0.01 } }
            var out = [Float](repeating: 0.01, count: items.count)
            let at = items.indices.filter { mk[items[$0]] }
            if !at.isEmpty {
                let p = try await rest.evaluate(at.map { items[$0] }, judgeLeaf)
                for (k, i) in at.enumerated() { out[i] = p[k] }
            }
            return out
        case .parts(let op, let children):
            var acc: [Float]? = nil
            for ch in children {
                let p = try await ch.evaluate(items, judgeLeaf)
                acc = acc.map { a in zip(a, p).map { op == .or ? max($0, $1) : min($0, $1) } } ?? p
            }
            return acc ?? []
        }
    }

    /// One photo (the app judges photo by photo): `item` indexes the library / the masks; judgeLeaf(question) = the
    /// judge's P for this photo. Same values as evaluate([item]).
    public func evaluateOne(_ item: Int, _ judgeLeaf: (String) async throws -> Float) async rethrows -> Float {
        switch self {
        case .leaf(let q):
            return try await judgeLeaf(q)
        case .meta(let mk, let rest):
            guard mk[item] else { return 0.01 }
            guard let rest = rest else { return 0.99 }
            return try await rest.evaluateOne(item, judgeLeaf)
        case .parts(let op, let children):
            var acc: Float? = nil
            for ch in children {
                let p = try await ch.evaluateOne(item, judgeLeaf)
                acc = acc.map { op == .or ? max($0, p) : min($0, p) } ?? p
            }
            return acc ?? 0
        }
    }

    /// The app's judge wrapper: a composed question by its parts, a plain one as is.
    public static func judge(_ question: String, items: [Int], places: PlaceIndex?,
                             _ judgeLeaf: ([Int], String) async throws -> [Float]) async rethrows -> [Float] {
        let t = decompose(question, places: places)
        if case .leaf(let q) = t, q == question { return try await judgeLeaf(items, question) }
        return try await t.evaluate(items, judgeLeaf)
    }
}
