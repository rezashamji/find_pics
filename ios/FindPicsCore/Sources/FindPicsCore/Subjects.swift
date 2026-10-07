// A request that names ONE specific pet or thing ("my dog Max at the beach", "my blue car", "Luna the cat"): the plan
// alone searches the category ("a dog"), which answers a different question. These rules decide when to ask for 1-3
// example photos (the way an unknown person gets the face picker) and rewrite the album so the subject search
// (SubjectSearch.swift in the app) runs with the rest of the plan as its scope.
// Port of src/findpics/subjects.py, checked against it on 2,319 cases (Tests/.../Fixtures/subjects.json).
import Foundation

let subjectPets = ["dog", "puppy", "pup", "cat", "kitten", "kitty", "horse", "pony", "bird", "parrot", "budgie", "rabbit", "bunny",
    "hamster", "guinea pig", "ferret", "goldfish", "turtle", "tortoise", "lizard", "gecko", "snake", "hen", "goat"]
let subjectThings = ["car", "truck", "van", "jeep", "bike", "bicycle", "motorcycle", "motorbike", "scooter", "boat", "kayak", "canoe",
    "guitar", "piano", "violin", "ukulele", "backpack", "bag", "purse", "handbag", "suitcase", "watch", "ring",
    "necklace", "bracelet", "shoes", "sneakers", "boots", "jacket", "coat", "hat", "cap", "dress", "sweater", "scarf",
    "glasses", "sunglasses", "teddy bear", "teddy", "stuffed animal", "plush", "toy", "doll", "mug", "couch", "sofa",
    "chair", "desk", "tent", "skateboard", "surfboard", "snowboard", "stroller", "plant", "camper"]
let subjectKinds = Set(subjectPets + subjectThings)

func wordSet(_ s: String) -> Set<String> { Set(s.split(whereSeparator: { $0 == " " || $0 == "\n" }).map(String.init)) }

/// words that end the "my <modifiers> <kind>" phrase, and that are never a name
let subjectStop = wordSet("""
photo photos picture pictures pic pics image images video videos clip clips shot shots selfie selfies of with
and or the a an at in on from to for by near under over is are was were be been this that these those all every some
any me my our your his her their its i we you he she they it them him not no without but only just also when where
while who which what how why as if than then so very s one ones
""")
let subjectNeg: Set<String> = ["not", "without", "except", "excluding", "exclude", "drop", "remove", "minus", "nor"]
/// "selfies in my car", "the dog on my couch": a place the photo was taken, not the subject
let subjectWhere: Set<String> = ["in", "inside", "into", "on", "onto", "from", "under", "off", "out"]
/// "my car keys": the kind is only a modifier of another noun
let subjectCompoundAfter = wordSet("""
keys key seat seats ride rides trip trips wash show shows accident food bowl toys bed park parking
leash collar house door window case strap band charger cover lesson lessons class classes walker sitter groomer vet race
racing dealer repair insurance lot garage rack tire tires wheel engine hair fur treats treat cage tank stand string strings
shop store box
""")
let subjectNotNames = wordSet("""
january february march april may june july august september october november december monday tuesday
wednesday thursday friday saturday sunday christmas xmas halloween easter thanksgiving
""")
/// a condition question that only restates the subject ("Is there a real dog anywhere in this photo (not a drawing...)?")
let subjectTrivial = wordSet("""
is are there this that a an the any some real photo photos image images picture pictures video videos clip
frame of in visible anywhere not drawing painting statue toy model or one does do show shown showing it its specific named
called with contain contains containing s my your our person
""")
let subjectTokenPat = #"[A-Za-z0-9]+(?:'[A-Za-z]+)?"#

public struct SubjectAsk: Equatable, Sendable {
    public var albums: [Int]
    public var name: String?        // "Max" when the request names it
    public var kind: String         // "dog", "car"
    public var words: String        // modifiers + kind as said, lowercased: "blue car", "sister's dog", "dog"
    public var fromSaved = false    // recognised from a name whose photos were picked earlier
    public var key: String { name.map { "name:" + normText($0) } ?? "my:" + normText(words) }
    /// "Max" / "your blue car"
    public var display: String { name ?? "your \(words)" }
    /// "Show me Max: pick 1-3 clear photos of Max."
    public var prompt: String {
        name.map { "Show me \($0): pick 1-3 clear photos of \($0)." } ?? "Show me \(display): pick 1-3 clear photos of it."
    }
    /// for the side-by-side question: "Max" / "this car"
    public var judgeName: String { name ?? "this \(kind)" }
    public init(albums: [Int], name: String?, kind: String, words: String, fromSaved: Bool = false) {
        self.albums = albums; self.name = name; self.kind = kind; self.words = words; self.fromSaved = fromSaved
    }
}

/// A subject whose example photos were picked before (stored by the app with the photo ids).
public struct SavedSubject: Codable, Equatable, Sendable {
    public var name: String?
    public var kind: String
    public var words: String
    public var ids: [String]
    public init(name: String?, kind: String, words: String, ids: [String] = []) {
        self.name = name; self.kind = kind; self.words = words; self.ids = ids
    }
    enum CodingKeys: String, CodingKey { case name, kind, words, ids }
    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        name = try c.decodeIfPresent(String.self, forKey: .name)
        kind = try c.decode(String.self, forKey: .kind)
        words = try c.decodeIfPresent(String.self, forKey: .words) ?? kind
        ids = try c.decodeIfPresent([String].self, forKey: .ids) ?? []
    }
    public var key: String { name.map { "name:" + normText($0) } ?? "my:" + normText(words.isEmpty ? kind : words) }
}

func isAlphaToken(_ t: String) -> Bool { !t.isEmpty && t.unicodeScalars.allSatisfy { ($0.value >= 65 && $0.value <= 90) || ($0.value >= 97 && $0.value <= 122) } }
func capitalizedToken(_ t: String) -> Bool { guard let c = t.unicodeScalars.first else { return false }; return c.value >= 65 && c.value <= 90 }
func subjectNameOK(_ t: String) -> Bool {
    let lo = t.lowercased()
    return isAlphaToken(t) && t.count >= 2 && !subjectStop.contains(lo) && !subjectKinds.contains(lo) && !subjectNotNames.contains(lo)
}
func upperFirst(_ t: String) -> String { t.prefix(1).uppercased() + t.dropFirst() }

func subjectAlbums(_ p: Plan, kind: String, name: String?) -> [Int] {
    if p.albums.count == 1 { return [0] }
    var out = [Int]()
    for (k, a) in p.albums.enumerated() {
        if let n = name, normText(a.person ?? "") == normText(n) { out.append(k); continue }
        let words = wordSet(normText(([a.name] + a.looks + [a.judgeQuestion ?? "", a.filterQuestion ?? ""]).joined(separator: " ")))
        let kt = normText(kind).split(separator: " ").map(String.init)
        if kt.allSatisfy({ words.contains($0) || words.contains($0 + "s") }) || (name.map { words.contains(normText($0)) } ?? false) {
            out.append(k)
        }
    }
    return out
}

/// (name, kind, words) for every "my/our <0-2 modifiers> <kind> [Name | named X]", "the <kind> named X", "X the <kind>".
func scanSubjects(_ text0: String, planNames: Set<String>) -> [(String?, String, String)] {
    let text = text0.replacingOccurrences(of: "\u{2019}", with: "'")
    let ns = text as NSString
    let ms = regex(subjectTokenPat).matches(in: text, range: NSRange(location: 0, length: ns.length))
    let tok = ms.map { ns.substring(with: $0.range) }
    let lo = tok.map { $0.lowercased() }
    let n = tok.count
    func kindAt(_ j: Int) -> (String?, Int) {
        if j + 1 < n, subjectKinds.contains(lo[j] + " " + lo[j + 1]) { return (lo[j] + " " + lo[j + 1], j + 2) }
        if subjectKinds.contains(lo[j]) { return (lo[j], j + 1) }
        return (nil, j)
    }
    func nameAfter(_ end: Int) -> String? {
        guard end < n else { return nil }
        if ["named", "called"].contains(lo[end]), end + 1 < n, subjectNameOK(tok[end + 1]) { return upperFirst(tok[end + 1]) }
        if subjectNameOK(tok[end]), capitalizedToken(tok[end]) || planNames.contains(lo[end]) { return upperFirst(tok[end]) }
        return nil
    }
    func sentenceStart(_ k: Int) -> Bool {
        let before = ns.substring(to: ms[k].range.location).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let c = before.last else { return true }
        return ".!?\n".contains(c)
    }
    func negBefore(_ from: Int, _ to: Int) -> Bool { (max(0, from)..<max(max(0, from), to)).contains { subjectNeg.contains(lo[$0]) } }
    var out = [(String?, String, String)]()
    for i in 0..<n {
        if lo[i] == "my" || lo[i] == "our" {
            if negBefore(i - 5, i) || (i >= 1 && subjectWhere.contains(lo[i - 1])) { continue }
            var j = i + 1, mods = [String](), kind: String? = nil, end = i + 1
            while j < n {
                (kind, end) = kindAt(j)
                if kind != nil || subjectStop.contains(lo[j]) || mods.count == 2 { break }
                mods.append(lo[j]); j += 1
            }
            guard let k = kind, !(end < n && subjectCompoundAfter.contains(lo[end])) else { continue }
            out.append((nameAfter(end), k, (mods + [k]).joined(separator: " ")))
            continue
        }
        let (kind, end) = kindAt(i)
        guard let k = kind else { continue }
        if i >= 1, ["the", "a", "an"].contains(lo[i - 1]), end + 1 < n, ["named", "called"].contains(lo[end]),
           subjectNameOK(tok[end + 1]), !negBefore(i - 4, i), !(i >= 2 && (lo[i - 2] == "my" || lo[i - 2] == "our")) {
            out.append((upperFirst(tok[end + 1]), k, k))
        }
        if i >= 2, lo[i - 1] == "the", capitalizedToken(tok[i - 2]), subjectNameOK(tok[i - 2]), !sentenceStart(i - 2),
           !(end < n && subjectCompoundAfter.contains(lo[end])), !negBefore(i - 5, i - 2) {
            out.append((tok[i - 2], k, k))
        }
    }
    return out
}

/// Which albums are about ONE specific pet / thing, and what to ask for. `saved`: subjects whose photos were picked
/// before (a saved name is recognised without "my dog").
public func namedSubjects(_ p: Plan, message: String, history: [String] = [], saved: [SavedSubject] = []) -> [SubjectAsk] {
    let msgs = history + [message]
    var planNames = Set<String>()
    for a in p.albums { for x in [a.person ?? ""] + a.withPeople where !normText(x).isEmpty { planNames.insert(normText(x)) } }
    for x in p.unknownPeople where !normText(x).isEmpty { planNames.insert(normText(x)) }
    var found = [SubjectAsk](), keys = Set<String>()
    for m in msgs {
        for (name, kind, words) in scanSubjects(m, planNames: planNames) {
            var ask = SubjectAsk(albums: [], name: name, kind: kind, words: words)
            ask.albums = subjectAlbums(p, kind: kind, name: name)
            if !ask.albums.isEmpty && !keys.contains(ask.key) { keys.insert(ask.key); found.append(ask) }
        }
    }
    for s in saved {
        guard let nm = s.name else { continue }
        let saidIt = msgs.contains { m in
            let t = m.replacingOccurrences(of: "\u{2019}", with: "'")
            return searchAll(subjectTokenPat, t).contains { $0.group(0) == nm }
        }
        var albums = p.albums.indices.filter { normText(p.albums[$0].person ?? "") == normText(nm) }
        if albums.isEmpty && saidIt { albums = subjectAlbums(p, kind: s.kind, name: nm) }
        let ask = SubjectAsk(albums: albums, name: nm, kind: s.kind, words: s.words.isEmpty ? s.kind : s.words, fromSaved: true)
        if !albums.isEmpty && !keys.contains(ask.key) { keys.insert(ask.key); found.append(ask) }
    }
    return found
}

/// Index of the saved subject this ask means, or nil (then ask for photos). Exact key first; "my dog" with no
/// modifiers also means the one saved dog when there is exactly one.
public func resolveSubject(_ ask: SubjectAsk, saved: [SavedSubject]) -> Int? {
    if let i = saved.firstIndex(where: { $0.key == ask.key }) { return i }
    if ask.name == nil && ask.words == ask.kind {
        let same = saved.indices.filter { saved[$0].kind == ask.kind }
        if same.count == 1 { return same[0] }
    }
    return nil
}

func subjectRename(_ q: String?, _ ask: SubjectAsk) -> String? {
    guard var q = q, !q.isEmpty else { return q }
    let the = "the \(ask.kind)"
    q = subLit(#"(?i)\bthe person in the red box\b"#, q, the)
    if let name = ask.name {
        let nm = esc(name)
        q = subLit(#"(?i)\b(?:an?|the|my|our) (?:\w+ )?"# + esc(ask.kind) + #" (?:named|called) "# + nm + #"\b"#, q, the)
        q = subLit(#"(?i)\b"# + nm + #"\b"#, q, the)
        q = subLit(#"(?i)\bthe the\b"#, q, "the")
    }
    return q
}

/// The album's own condition to ask about the subject's photos ("... at the beach"), or nil when the question only
/// restates the subject ("Is there a real dog anywhere in this photo?").
public func subjectCondition(_ q0: String?, _ ask: SubjectAsk) -> String? {
    guard let q = subjectRename(q0, ask), !q.isEmpty else { return nil }
    var skip = subjectTrivial.union(normText(ask.words).split(separator: " ").map(String.init))
    skip.formUnion(normText(ask.kind).split(separator: " ").map { $0 + "s" })
    if let n = ask.name { skip.formUnion(normText(n).split(separator: " ").map(String.init)) }
    return normText(q).split(separator: " ").contains { !skip.contains(String($0)) } ? q : nil
}

/// The plan with each subject album rewritten for the subject search: the subject is not a person (any person on the
/// album must ALSO be in the photo: withPeople), and questions say "the dog", never its name (the judge cannot know it).
public func subjectPlan(_ p0: Plan, asks: [SubjectAsk]) -> Plan {
    var p = p0
    for ask in asks {
        let nn = normText(ask.name ?? "")
        for k in ask.albums {
            var a = p.albums[k]
            if let person = a.person, !person.isEmpty, normText(person) != nn,
               ![normText(ask.words), normText("my " + ask.words)].contains(normText(person)) {
                a.withPeople = [person] + a.withPeople.filter { normText($0) != normText(person) }
            }
            a.person = nil
            a.withPeople = a.withPeople.filter { normText($0) != nn || nn.isEmpty }
            a.judgeQuestion = subjectCondition(a.judgeQuestion, ask)
            a.filterQuestion = subjectCondition(a.filterQuestion, ask)
            a.excludeQuestion = subjectRename(a.excludeQuestion, ask)
            p.albums[k] = a
        }
        let drop = Set([nn, normText(ask.words), normText("my " + ask.words), normText("our " + ask.words)]).subtracting([""])
        p.unknownPeople = p.unknownPeople.filter { !drop.contains(normText($0)) }
    }
    return p
}
