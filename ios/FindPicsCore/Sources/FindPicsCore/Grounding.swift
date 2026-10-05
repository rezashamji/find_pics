// Code-enforced checks on the planner's JSON: port of converse.ground (+ planner.ground_dates / ground_place /
// strip_identity_conditions / fix_red_box) and converse._unanswerable / _drop_unanswerable. The planner model can be
// wrong; these rules cannot (dates from the words, people must be named, no questions one photo cannot answer...).
// Checked against the Python on 2,300 planner outputs (Tests/.../Fixtures/ground.json).
import Foundation

// MARK: - regex helpers with Python semantics

func esc(_ s: String) -> String { NSRegularExpression.escapedPattern(for: s) }

func searchAll(_ pattern: String, _ s: String) -> [Match] {
    let ns = s as NSString
    return regex(pattern).matches(in: s, range: NSRange(location: 0, length: ns.length)).map { m in
        Match(groups: (0..<m.numberOfRanges).map { m.range(at: $0).location == NSNotFound ? nil : ns.substring(with: m.range(at: $0)) })
    }
}

/// re.sub with a function replacement
func subFn(_ pattern: String, _ s: String, _ f: (Match) -> String) -> String {
    let ns = s as NSString
    var out = "", last = 0
    for m in regex(pattern).matches(in: s, range: NSRange(location: 0, length: ns.length)) {
        out += ns.substring(with: NSRange(location: last, length: m.range.location - last))
        let g = Match(groups: (0..<m.numberOfRanges).map { m.range(at: $0).location == NSNotFound ? nil : ns.substring(with: m.range(at: $0)) })
        out += f(g); last = m.range.location + m.range.length
    }
    return out + ns.substring(from: last)
}

/// re.sub with a literal replacement (no template escapes)
func subLit(_ pattern: String, _ s: String, _ rep: String) -> String { subFn(pattern, s) { _ in rep } }

func has(_ pattern: String, _ s: String) -> Bool { search(pattern, s) != nil }

func strip(_ s: String) -> String { s.trimmingCharacters(in: .whitespacesAndNewlines) }

func appendNote(_ p: inout Plan, _ n: String) { p.notes = strip(p.notes + " " + n) }

// MARK: - constants (same patterns as converse.py)

let calPat = #"(?i)\d|\b(today|tonight|yesterday|ago|last|past|this|next|recent|recently|decade|century|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december|spring|summer|fall|autumn|winter|christmas|thanksgiving|halloween|easter|new year|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"#
let relativePat = #"(?i)\b(before|after|since|until|prior to)\b(?!.*\b(19|20)\d\d\b)(?!.*\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)"#
let relationalPat = #"(?i)\b(same|identical)\b[^?]*\bas (the|in|that|a)\b|\breference (photo|image|picture)\b|\b(previous|earlier|other|another|first|anchor|later) (photo|image|picture|event|day|trip|time)\b|\bsame (year|day|week|month|trip) as\b|\b(also|again|later)\b[^?]*\b(appear|appears|seen|shown|photographed|docking|docked)\b|\bsame (one|top|shirt|jacket|scarf|hat|person|dog|car|outfit|clothes)\b[^?]*\b(as|worn|in \d{4})\b|\b(twice|three times|consecutive)\b"#
let windowWords: [(String, String)] = [("same_day", #"(?i)\b(the|that) day\b"#), ("same_week", #"(?i)\b(the|that) week\b"#),
    ("same_month", #"(?i)\b(the|that) month\b"#), ("same_year", #"(?i)\b(the|that) year\b"#),
    ("same_event", #"(?i)\b(trip|vacation|holiday|party|wedding|concert|game|event)\b"#)]
let monthsAlt = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
let monthRangePat = #"(?i)\b(?:between )?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?: \d{4})? (?:to|till|til|until|through|thru|and|-) (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?: \d{4})?(?: (?:of )?(?:last|this) year)?\b"#
let relPhrasePat = #"(?i)\b(?:(?:last|this|past)\s+(?:summer|winter|spring|fall|autumn|week|weekend|month|year|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|yesterday|today|tonight|this morning|(?:the\s+)?(?:week|weekend) before last)\b"#
let occasionPat = #"\b(?:on |at |for |from )?(?:(?:my|our|his|her|their|your|the|\w+ s) )?(?:\d+(?:st|nd|rd|th)? |first |last )?(?:birthday|bday|anniversary|wedding day|wedding|graduation)(?: party| celebration| dinner)?\b"#
let namedPat = #"(?i)\b(person|man|woman|boy|girl|child|kid|baby|guy|lady|someone|friend)\s+(named|called)\b"#
let metadataPat = #"(?i)\b(raw (file|version|format)|\d+ ?fps|frames per second|\d+ ?mm\b|telephoto|wide[- ]angle lens|lens\b(?! flare)|audio|sound track|music track|location tag|geotag|tagged|rated|rating|favou?rited|filter applied|backup|file\b|\d ?k\b|1080p|720p|high[- ]res|resolution|timestamp|seconds long|minutes long|duration|trimmed|time (is )?between|time of day|between \d{1,2} ?(am|pm)|shot on (a|my) phone|edited|unedited|exif)\b"#
let invisibleEventPat = #"(?i)\b(pass(?:ed|es|ing)? away|died|dies|dying|death|(?:got|get|gets|getting|became|becoming) (?:sick|ill)|fell ill|falling ill|(?:was|got|being|getting) sold|broke up|breaking up|got divorced|getting divorced|moved out|moving out|retired|retiring)\b"#
let relWords = "brother|sister|mom|mother|dad|father|grandma|grandmother|grandpa|grandfather|uncle|aunt|cousin|son|daughter|wife|husband|niece|nephew|grandson|granddaughter|friend|boyfriend|girlfriend|partner"
let isNamePat = #"\b(?:[Ii]s|[Aa]re) (?:this|that|the) (?:person|man|woman|guy|girl|boy|kid|child)\s+[A-Z][a-z]+\b|\b(?:[Ii]s|[Aa]re) (?:this|that|it) (?:(?:my|our|the|your) )?(?:(?:"# + relWords + #")(?: [A-Z][a-z]+)?|the person in the (?:photo|image|picture|video))\??$|(?i:\b(?:uncle|aunt|grandma|grandpa|cousin)\s+)[A-Z][a-z]+"#
let inventedLookPat = #"(?i)\b(long|short|gray|grey|white|blonde|blond|brown|black|dark|red|curly|straight) hair\b"#
let exampleLeaks: [(String, String)] = [("heavy build", "(?i)heav|weight|fat|big|overweight|chubby|build"),
    ("round face", "(?i)round|face|heav"), ("slice of bread", "(?i)bread|toast|sandwich|loaf"),
    ("grand canyon", "(?i)grand canyon"), ("burger", "(?i)burger"), ("sandwich", "(?i)sandwich")]
let everythingPat = #"(?i)\b(photos|pics|pictures|videos|clips|images|everything|anything|all|memories|footage|album|shots)\b[^.?!]{0,40}?\b(from|at|during|on)\b|\b(trip|vacation|holiday|weekend|day|night)\b"#
let identityOnly = #"(is|are) (this|that|it|the person|the person in the red box|this person) NAME|(does|do|is) (the person|the person in the red box|this person|he|she|they) (look|looks|resemble|resembles) (like )?NAME|(does|do) (this|the) (photo|image|picture|video) (show|contain|include|feature|have) NAME|(is|are) NAME (in|present in|visible in|shown in|pictured in) (this|the) (photo|image|picture|video)|(is|are) NAME (in|present|visible|there|pictured)( here)?|(does|do) NAME appear( in (this|the) (photo|image|picture|video))?"#

func personal(_ q: String) -> Bool {
    has(#"\bI\b"#, q) || has(#"(?i)\b(me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours)\b|\bthe (user|owner)\b"#, q)
}

func badQ(_ q: String?) -> Bool {
    guard let q = q, !q.isEmpty else { return false }
    return has(relationalPat, q) || personal(q) || has(namedPat, q) || has(metadataPat, q) || has(invisibleEventPat, q) || has(isNamePat, q)
}

func namesIn(_ q: String, _ names: [String], person: String? = nil) -> [String] {
    if !q.isEmpty && has(#"(?i)\b(screenshot|chat|message|text|texts|written|says|reads|label|sign|caption|contact|conversation|email|name)\b"#, q) { return [] }
    let own = Set(normText(person ?? "").split(separator: " ").map(String.init))
    var toks = Set<String>()
    for n in names { for t in normText(n).split(separator: " ").map(String.init) where t.count > 2 && !own.contains(t) && t != "me" { toks.insert(t) } }
    return toks.filter { !q.isEmpty && has(#"\b"# + esc($0) + #"\b"#, normText(q)) }.sorted()
}

// MARK: - _unanswerable / _drop_unanswerable

func leak(_ p: Plan, _ said: String) -> String? {
    for a in p.albums {
        var texts = a.looks + [a.judgeQuestion ?? "", a.filterQuestion ?? "", a.excludeQuestion ?? ""]
        if let an = a.anchor { texts += an.looks + [an.judgeQuestion ?? ""] }
        for t in texts { for (ph, ok) in exampleLeaks where t.lowercased().contains(ph) && !has(ok, said) { return ph } }
    }
    return nil
}

/// A reason the plan needs another try (nil = fine). Same order of checks as the Python.
public func unanswerable(_ p: Plan, names: [String], said: String) -> String? {
    if !said.isEmpty && !has("(?i)hair", said) {
        for a in p.albums { for q in [a.judgeQuestion ?? "", a.filterQuestion ?? ""] + a.looks where has(inventedLookPat, q) { return "invented look: \(q)" } }
    }
    if !said.isEmpty, let l = leak(p, said) { return "example leak: \(l)" }
    for a in p.albums {
        let own = Set(normText(a.person ?? "").split(separator: " ").map(String.init))
        var toks = Set<String>()
        for n in names { for t in normText(n).split(separator: " ").map(String.init) where t.count > 2 && !own.contains(t) && t != "me" { toks.insert(t) } }
        for q in [a.judgeQuestion, a.filterQuestion, a.excludeQuestion] {
            if let q = q, toks.contains(where: { has(#"\b"# + esc($0) + #"\b"#, normText(q)) }) { return "names a person: \(q)" }
        }
        let qs: [String?] = [a.judgeQuestion, a.anchor?.judgeQuestion, a.until?.judgeQuestion, a.excludeQuestion, a.filterQuestion]
        for q in qs.compactMap({ $0 }) where !q.isEmpty {
            if has(invisibleEventPat, q) || has(relationalPat, q) || has(metadataPat, q) || has(namedPat, q) || has(isNamePat, q) || personal(q) {
                return "not answerable from one photo: \(q)"
            }
        }
        if let an = a.anchor, let jq = a.judgeQuestion, normText(jq) == normText(an.judgeQuestion ?? "") { return "repeats the anchor" }
    }
    return nil
}

public func dropUnanswerable(_ p0: Plan, said: String, names: [String]) -> Plan {
    var p = p0
    for i in p.albums.indices {
        if !has("(?i)hair", said) {
            p.albums[i].looks = p.albums[i].looks.filter { !has(inventedLookPat, $0) }
            if let q = p.albums[i].judgeQuestion, has(inventedLookPat, q) {
                appendNote(&p, "[dropped '\(q)': it guessed what the person looks like]"); p.albums[i].judgeQuestion = nil
            }
        }
        let bad = exampleLeaks.filter { !has($0.1, said) }.map { $0.0 }
        p.albums[i].looks = p.albums[i].looks.filter { x in !bad.contains { x.lowercased().contains($0) } }
        if p.albums[i].anchor != nil { p.albums[i].anchor!.looks = p.albums[i].anchor!.looks.filter { x in !bad.contains { x.lowercased().contains($0) } } }
        if bad.contains(where: { (p.albums[i].judgeQuestion ?? "").lowercased().contains($0) }) { p.albums[i].judgeQuestion = nil }
        if bad.contains(where: { (p.albums[i].filterQuestion ?? "").lowercased().contains($0) }) { p.albums[i].filterQuestion = nil }
        if bad.contains(where: { (p.albums[i].excludeQuestion ?? "").lowercased().contains($0) }) { p.albums[i].excludeQuestion = nil }
    }
    for i in p.albums.indices {
        let person = p.albums[i].person
        func plain(_ looks: [String]) -> String? {
            let lk = looks.map { subLit(#"(?i)\s+(named|called)\s+\w+"#, $0, "") }
                .filter { !badQ("Does this photo show \($0)?") && namesIn($0, names, person: person).isEmpty }
            return lk.first.map { "Does this photo show \($0)?" }
        }
        if let q = p.albums[i].judgeQuestion, !namesIn(q, names, person: person).isEmpty {
            appendNote(&p, "[dropped '\(q)': the judge cannot recognize people by name]"); p.albums[i].judgeQuestion = plain(p.albums[i].looks)
        }
        if let q = p.albums[i].filterQuestion, !namesIn(q, names, person: person).isEmpty {
            appendNote(&p, "[dropped '\(q)': the judge cannot recognize people by name]"); p.albums[i].filterQuestion = nil
        }
        if let q = p.albums[i].excludeQuestion, !namesIn(q, names, person: person).isEmpty {
            appendNote(&p, "[dropped '\(q)': the judge cannot recognize people by name]"); p.albums[i].excludeQuestion = nil
        }
        if let q = p.albums[i].judgeQuestion, let an = p.albums[i].anchor, normText(q) == normText(an.judgeQuestion ?? ""),
           !(!said.isEmpty && has(everythingPat, said)) {
            appendNote(&p, "[searching for the thing itself, not everything around it]"); p.albums[i].anchor = nil; p.albums[i].window = nil
        }
        if let q = p.albums[i].judgeQuestion, let an = p.albums[i].anchor, normText(q) == normText(an.judgeQuestion ?? "") {
            appendNote(&p, "[showing everything from that moment]"); p.albums[i].judgeQuestion = nil; p.albums[i].looks = []
        }
        if let q = p.albums[i].judgeQuestion, badQ(q) {
            appendNote(&p, "[could not express '\(q)' as a question about one photo; searching for what it looks like instead]")
            p.albums[i].judgeQuestion = plain(p.albums[i].looks)
        }
        if let u = p.albums[i].until, badQ(u.judgeQuestion) || !namesIn(u.judgeQuestion ?? "", names).isEmpty {
            appendNote(&p, "[dropped the end moment '\(u.judgeQuestion ?? "")': no photo can show it]"); p.albums[i].until = nil
        }
        if let an = p.albums[i].anchor, has(invisibleEventPat, an.judgeQuestion ?? "") {
            appendNote(&p, "[dropped the moment '\(an.judgeQuestion ?? "")': no photo can show it]"); p.albums[i].anchor = nil; p.albums[i].window = nil
        }
        if let an = p.albums[i].anchor, badQ(an.judgeQuestion) {
            p.albums[i].anchor!.judgeQuestion = plain(an.looks) ?? an.judgeQuestion
        }
        if badQ(p.albums[i].excludeQuestion) {
            appendNote(&p, "[dropped exclusion '\(p.albums[i].excludeQuestion!)': not answerable from one photo]"); p.albums[i].excludeQuestion = nil
        }
        if badQ(p.albums[i].filterQuestion) {
            appendNote(&p, "[dropped condition '\(p.albums[i].filterQuestion!)': not answerable from one photo]"); p.albums[i].filterQuestion = nil
        }
    }
    return p
}

// MARK: - planner.py helpers

func groundDates(_ p: inout Plan, _ request: String) {
    let req = normText(request)
    for i in p.albums.indices {
        let tp = normText(p.albums[i].timePhrase ?? "")
        if tp.isEmpty || !req.contains(tp) {
            if p.albums[i].dateFrom != nil || p.albums[i].dateTo != nil {
                appendNote(&p, "[dates removed from '\(p.albums[i].name)': no time phrase in the request supports them]")
            }
            p.albums[i].dateFrom = nil; p.albums[i].dateTo = nil; p.albums[i].timePhrase = nil
        }
    }
}

func groundPlace(_ p: inout Plan, _ request: String) {
    let req = normText(request)
    for i in p.albums.indices {
        if let pl = p.albums[i].place, !pl.isEmpty, !req.contains(normText(pl)) {
            appendNote(&p, "[place '\(pl)' removed from '\(p.albums[i].name)': not in the request]"); p.albums[i].place = nil
        }
    }
}

func fixRedBox(_ p: inout Plan) {
    for i in p.albums.indices {
        if p.albums[i].person == nil || p.albums[i].person == "", let q0 = p.albums[i].judgeQuestion, q0.lowercased().contains("red box") {
            var q = subLit(#"(?i)\b(the|a) person in (the|a) red box\b"#, q0, "someone")
            q = subLit(#"(?i)\s*\bin (the|a) red box\b"#, q, "")
            p.albums[i].judgeQuestion = strip(subLit(#"\s{2,}"#, q, " "))
        }
    }
}

func boxName(_ q: String, _ toks: [String]) -> String {
    let alt = toks.map(esc).joined(separator: "|")
    return subFn(#"(?i)\b(?:"# + alt + #")(?:\s+(?:"# + alt + #"))*('s)?(?!\w)"#, q) { "the person in the red box" + ($0.group(1) ?? "") }
}

func stripIdentityConditions(_ p: inout Plan) {
    for i in p.albums.indices {
        guard let person = p.albums[i].person, !person.isEmpty else { continue }
        let toks = normText(person).split(separator: " ").map(String.init).filter { $0.count > 2 }
        if !toks.isEmpty {
            if let q = p.albums[i].filterQuestion, !q.isEmpty { p.albums[i].filterQuestion = boxName(q, toks) }
            if let q = p.albums[i].excludeQuestion, !q.isEmpty { p.albums[i].excludeQuestion = boxName(q, toks) }
        }
        if let jq = p.albums[i].judgeQuestion, !jq.isEmpty,
           fullmatch(#"(is|are) there (a |any )?(person|people|someone|a face|faces)( visible)?( in (the|this) (photo|image|picture|video))?"#, normText(jq)) != nil {
            p.albums[i].judgeQuestion = nil; continue
        }
        if let jq = p.albums[i].judgeQuestion, !jq.isEmpty {
            let q = normText(jq)
            let qw = Set(q.split(separator: " ").map(String.init))
            if !toks.contains(where: { qw.contains($0) }) { continue }
            let name = "(?:" + toks.map(esc).joined(separator: "|") + ")(?: s)?"
            let qn = subLit(#"\b"# + name + "(?: " + name + #")*\b"#, q, "NAME")
            if fullmatch(identityOnly, qn) != nil {
                appendNote(&p, "[identity-style condition removed from '\(p.albums[i].name)': identity uses face matching]")
                p.albums[i].judgeQuestion = nil; p.albums[i].looks = []; p.albums[i].avoid = []
            } else {
                p.albums[i].judgeQuestion = boxName(jq, toks)
            }
        }
    }
}

func dropNameQuestions(_ a: inout Album, person: String?, owner: String) {
    let isMe = ["me", "i", "myself"].contains(normText(person ?? ""))
    let names = [person ?? ""] + (isMe ? [owner] : [])
    if !namesIn(a.judgeQuestion ?? "", names).isEmpty { a.judgeQuestion = nil }
    if !namesIn(a.filterQuestion ?? "", names).isEmpty { a.filterQuestion = nil }
    if !namesIn(a.excludeQuestion ?? "", names).isEmpty { a.excludeQuestion = nil }
}

// MARK: - ground

let filler: Set<String> = ["all", "every", "everything", "my", "the", "our", "of", "from", "photo", "photos", "picture", "pictures", "pic",
    "pics", "image", "images", "video", "videos", "clip", "clips", "show", "me", "find", "get", "give", "please", "and", "whole",
    "entire", "library", "camera", "roll", "any", "some", "a", "an", "i", "want", "see", "can", "you"]

func nameTheThing(_ p: inout Plan, _ message: String) {
    guard p.albums.count == 1 else { return }
    let a = p.albums[0]
    if a.judgeQuestion != nil || (a.person != nil && a.person != "") || a.anchor != nil || a.until != nil || a.dateFrom != nil ||
        a.dateTo != nil || (a.place != nil && a.place != "") || a.timeOfDay != nil || a.filterQuestion != nil ||
        a.excludeQuestion != nil || !a.withPeople.isEmpty { return }
    let rest = searchAll("[a-z]+", message.lowercased()).map { $0.group(0)! }.filter { !filler.contains($0) }
    guard rest.count == 1 else { return }
    let w = rest[0]
    let sing: String
    if w.hasSuffix("ies") { sing = String(w.dropLast(3)) + "y" }
    else if has("(ss|ch|sh|x)es$", w) { sing = String(w.dropLast(2)) }
    else if w.hasSuffix("s") && !w.hasSuffix("ss") { sing = String(w.dropLast()) }
    else { sing = w }
    p.albums[0].judgeQuestion = "Is this \("aeiou".contains(sing.first!) ? "an" : "a") \(sing)?"
}

func truthy(_ s: String?) -> Bool { s != nil && s != "" }

public func ground(_ p0: Plan, message: String, history: [String], today: Day, owner: String = "me", people: [String] = []) -> Plan {
    var p = p0
    let said = (history + [message]).joined(separator: " \n ")
    for i in p.albums.indices {
        if let tp = p.albums[i].timePhrase, !tp.isEmpty, resolveRelative(tp, today: today) == nil, (!has(calPat, tp) || has(relativePat, tp)) {
            appendNote(&p, "[dates removed from '\(p.albums[i].name)': '\(tp)' names no calendar time]"); p.albums[i].timePhrase = nil
        }
    }
    let saidTok = Set(normText(said).split(separator: " ").map(String.init))
    let spanPat = #"(?i)\b((?:from |between |in )?(?:19|20)\d\d(?:\s*(?:-|to|and|through|until)\s*(?:19|20)\d\d)?)\b"#
    var span = searchAll(spanPat, message).map { $0.group(1)! }
    if span.isEmpty { span = searchAll(spanPat, said).map { $0.group(1)! } }
    var relSet = Set(searchAll(relPhrasePat, message).map { $0.group(0)! })
    relSet.formUnion(searchAll(monthRangePat, message).map { $0.group(0)! })
    var rel = relSet.sorted()
    rel = rel.filter { x in !rel.contains { y in x != y && y.lowercased().contains(x.lowercased()) } }
    let datedCount = p.albums.filter { truthy($0.dateFrom) || truthy($0.dateTo) }.count
    for i in p.albums.indices {
        let hasDates = truthy(p.albums[i].dateFrom) || truthy(p.albums[i].dateTo)
        if !truthy(p.albums[i].timePhrase) && hasDates && span.count == 1 {
            p.albums[i].timePhrase = span[0]
        } else if (!truthy(p.albums[i].timePhrase) || !normText(said).contains(normText(p.albums[i].timePhrase!))) && hasDates &&
                    rel.count == 1 && datedCount == 1 {
            p.albums[i].timePhrase = rel[0]
        }
        guard let tp = p.albums[i].timePhrase, !tp.isEmpty else { continue }
        let years = searchAll(#"\b(19\d\d|20\d\d)\b"#, tp).map { Int($0.group(1)!)! }.sorted()
        if !years.isEmpty && !has(#"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|spring|summer|fall|autumn|winter|christmas|week|month|day|before|after|since|until|prior|s\b)"#, tp)
            && !has(#"\d0s\b"#, tp) {
            p.albums[i].dateFrom = "\(years.first!)-01-01"; p.albums[i].dateTo = "\(years.last! + 1)-01-01"
        }
        if !normText(said).contains(normText(tp)) {
            let toks = normText(tp).split(separator: " ").map(String.init).filter { !["to", "and", "from", "through", "between", "in", "the"].contains($0) }
            if !toks.isEmpty && toks.allSatisfy({ saidTok.contains($0) }) { p.albums[i].timePhrase = said }
        }
    }
    groundDates(&p, said); groundPlace(&p, said)
    for i in p.albums.indices {
        guard let tp = p.albums[i].timePhrase, !tp.isEmpty else { continue }
        let ntp = normText(tp)
        if has(#"(?i)\b"# + esc(ntp) + #"\s+(tree|trees|lights|decorations|ornaments|costumes?|eggs?|cards?|sweaters?|markets?|movies?|songs?|cookies|wreath|stockings?|presents|gifts)\b"#, normText(said)),
           fullmatch(#"(?i)(christmas|xmas|halloween|easter|thanksgiving|valentine s|valentines)"#, ntp) != nil {
            appendNote(&p, "[no dates: '\(tp)' names a thing here, not a time]")
            p.albums[i].dateFrom = nil; p.albums[i].dateTo = nil; p.albums[i].timePhrase = nil; continue
        }
        if has(#"(?i)\bthe last (night|day|morning|evening|afternoon)\b"#, said),
           fullmatch(#"(?:on )?(?:the )?last (night|day|morning|evening|afternoon)"#, ntp) != nil {
            appendNote(&p, "[no dates: 'the last \(ntp.split(separator: " ").last!)' is a point in a trip, not yesterday]")
            p.albums[i].dateFrom = nil; p.albums[i].dateTo = nil; p.albums[i].timePhrase = nil; continue
        }
        if ["last", "latest", "most recent", "recent", "newest", "last one"].contains(ntp) {
            p.albums[i].dateFrom = nil; p.albums[i].dateTo = nil; p.albums[i].timePhrase = nil; continue
        }
        let rest = strip(subLit(#"\s+"#, subLit(occasionPat, ntp, " "), " "))
        if rest != ntp && !(!rest.isEmpty && has(calPat, rest)) {
            appendNote(&p, "[I don't know the date of '\(tp)'; no date limit]")
            p.albums[i].dateFrom = nil; p.albums[i].dateTo = nil; p.albums[i].timePhrase = nil; continue
        }
        let r = rest != ntp ? resolveRelative(rest, today: today) : resolveRelative(tp, today: today)
        if r != nil && rest != ntp { appendNote(&p, "[I don't know the date of the occasion in '\(tp)'; searching \(rest)]") }
        if let r = r {
            p.albums[i].dateFrom = r.0; p.albums[i].dateTo = r.1
        } else if let dt = p.albums[i].dateTo,
                  has(#"(?i)\b(and|to|through|thru|until|till|-)\s*("# + monthsAlt + #")[a-z]*\.?\s+(\d{1,2})(st|nd|rd|th)?\b"#, tp) {
            let ms = searchAll(#"(?i)\b("# + monthsAlt + #")[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b"#, tp)
            if let m = ms.last, let y = Int(dt.prefix(4)) {
                let mi = monthsAlt.split(separator: "|").map(String.init).firstIndex(of: String(m.group(1)!.lowercased().prefix(3)))! + 1
                let d = Int(m.group(2)!)!
                if d >= 1 && d <= Day.daysIn(y, mi), Day(y, mi, d).description == dt { p.albums[i].dateTo = Day(y, mi, d).adding(1).description }
            }
        }
    }
    let saidWords = normText(said).split(separator: " ").map(String.init)
    let firstPerson = has(#"(?i)\b(i|me|my|mine|myself|i'm|im|i've|ive|i'd|we|us|our|selfie|selfies)\b"#, said)
    for i in p.albums.indices {
        guard let person = p.albums[i].person, !person.isEmpty else { continue }
        let isOwner = ["me", "i", "myself"].contains(normText(person)) ||
            Set(normText(person).split(separator: " ").map(String.init)).isSubset(of: Set(normText(owner).split(separator: " ").map(String.init)))
        if isOwner && !firstPerson && !normText(owner).split(separator: " ").map(String.init).contains(where: { $0.count > 2 && saidWords.contains($0) }) {
            appendNote(&p, "[person removed from '\(p.albums[i].name)': the request does not mention you]")
            dropNameQuestions(&p.albums[i], person: person, owner: owner); p.albums[i].person = nil; continue
        }
        if !isOwner && !normText(person).split(separator: " ").map(String.init).contains(where: { t in
            t.count > 2 && saidWords.contains { w in w.count > 2 && (w.hasPrefix(t) || t.hasPrefix(w)) } }) {
            appendNote(&p, "[person '\(person)' removed from '\(p.albums[i].name)': not named in the request. Who is it? Say their name, or add reference photos]")
            dropNameQuestions(&p.albums[i], person: person, owner: owner); p.albums[i].person = nil
        }
    }
    var known = Set(people.map(normText)); known.formUnion([normText(owner), "me", "i", "myself"])
    let sw = normText(said).split(separator: " ").map(String.init)
    for i in p.albums.indices {
        var keep = [String]()
        for w in p.albums[i].withPeople {
            let nw = normText(w)
            if !nw.split(separator: " ").map(String.init).contains(where: { t in t.count > 2 && sw.contains { x in x.count > 2 && (x.hasPrefix(t) || t.hasPrefix(x)) } }) {
                appendNote(&p, "['\(w)' dropped from '\(p.albums[i].name)': not named in the request]"); continue
            }
            if known.contains(nw) || known.contains(where: { $0.count > 2 && (nw.contains($0) || $0.contains(nw)) }) {
                keep.append(w)
            } else if !nw.isEmpty {
                p.unknownPeople.append(w)
                appendNote(&p, "['\(w)' is not known yet, so this album does not require them]")
            }
        }
        p.albums[i].withPeople = keep.filter { normText($0) != normText(p.albums[i].person ?? "") }
        if !truthy(p.albums[i].person) && !p.albums[i].withPeople.isEmpty { p.albums[i].person = p.albums[i].withPeople.removeFirst() }
    }
    var seen = Set<String>(), uniq = [String]()
    for u in p.unknownPeople where !seen.contains(u) { seen.insert(u); uniq.append(u) }
    p.unknownPeople = uniq.filter { u in
        let base = normText(subLit(#"(?i)^my\s+"#, u, ""))
        return !base.isEmpty && !known.contains(normText(u)) && !known.contains(base) &&
            base.split(separator: " ").allSatisfy { sw.contains(String($0)) }
    }
    let tod = resolveTimeOfDay(message) ?? resolveTimeOfDay(said)
    for i in p.albums.indices {
        if let tod = tod, (truthy(p.albums[i].timeOfDay) || p.albums.count == 1) {
            if p.albums[i].timeOfDay != tod { appendNote(&p, "[time of day \(tod), local time where each photo was taken]") }
            p.albums[i].timeOfDay = tod
            if tod.prefix(5) > tod.suffix(5), let dt = p.albums[i].dateTo, let d = Day(iso: dt) { p.albums[i].dateTo = d.adding(1).description }
        } else if tod == nil {
            p.albums[i].timeOfDay = nil
        }
    }
    let subjReq = search(#"(?i)\b(\w+) (?:photos|pictures|pics|shots|images)\b|\b(?:photos|pictures|pics|shots|images) of (?:my |the |a |an |some )?(\w+)"#, message)
    let artWords = #"(?i)\b(drawings?|drawn|paintings?|painted|statues?|toys?|cartoons?|art|artwork|murals?|sculptures?|figurines?|models?|stuffed|plush|prints?|posters?|sketch\w*|illustrat\w*|logos?|signs?)\b"#
    for i in p.albums.indices {
        if let sr = subjReq, let m = fullmatch(#"(?i)is there (?:an? |any |some )?(.+?) (?:in|visible in) (?:this|the) (?:photo|image|picture)\??"#, strip(p.albums[i].judgeQuestion ?? "")),
           !truthy(p.albums[i].person), !has(#"(?i)\bwith\b|\bwhere\b|\banywhere\b"#, message) {
            let word = (sr.group(1) ?? sr.group(2) ?? "").lowercased()
            if !word.isEmpty && !["my", "the", "all", "old", "new", "best", "these", "those"].contains(word) && m.group(1)!.lowercased().contains(word) {
                p.albums[i].judgeQuestion = "Is this a photo of \(m.group(1)!)?"
            }
        }
        if let m = fullmatch(#"(?i)is there (an? |any |some )?(.+?) (?:anywhere )?(?:in|visible in) (?:this|the) (?:photo|image|picture|video|clip|frame)\??"#, strip(p.albums[i].judgeQuestion ?? "")),
           !truthy(p.albums[i].person), !has(artWords, said) {
            let art = ["a", "an"].contains(strip(m.group(1) ?? "").lowercased()) ? "a real " : "real "
            p.albums[i].judgeQuestion = "Is there \(art)\(m.group(2)!) anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
        }
        if has(#"(?i)\bselfies?\b"#, message) && !(p.albums[i].judgeQuestion ?? "").lowercased().contains("selfie") &&
            (p.albums.count == 1 || p.albums[i].name.lowercased().contains("selfie")) {
            p.albums[i].judgeQuestion = "Is this a selfie?"
        }
        if let q = p.albums[i].judgeQuestion, has(#"(?i)\bscreen ?shots?\b"#, q), p.albums[i].media == "any" { p.albums[i].media = "photo" }
    }
    stripIdentityConditions(&p); fixRedBox(&p)
    let placeQ = #"(?:Is|Was) (?:this|the) (?:photo|image|picture|video|clip|item)(?: taken| shot| from)? (?:in|at|from) ([A-Z][\w'-]*(?: [A-Z][\w'-]*)*)\??"#
    for i in p.albums.indices {
        for f in 0..<2 {
            let q = f == 0 ? p.albums[i].judgeQuestion : p.albums[i].filterQuestion
            if let m = fullmatch(placeQ, strip(q ?? "")), normText(said).contains(normText(m.group(1)!)),
               (!truthy(p.albums[i].place) || normText(p.albums[i].place!) == normText(m.group(1)!)) {
                p.albums[i].place = m.group(1)!
                if f == 0 { p.albums[i].judgeQuestion = nil } else { p.albums[i].filterQuestion = nil }
            }
        }
    }
    let mediaQ = #"(?i)(?:is|was) (?:this|it) (?:a |an )?(?:video|photo|picture|image|clip|video clip|video file|photo file|file|recording|home video|movie)\??"#
    for i in p.albums.indices {
        if let an = p.albums[i].anchor, let aq = an.judgeQuestion, [nil, "same_place", "same_event"].contains(p.albums[i].window),
           let m = fullmatch(#"(?:Is|Was) (?:this|the) (?:(?:photo|image|picture|video)(?: taken| shot)? (?:in|at|from) )?(?:the )?([A-Z][\w'-]*(?: [A-Z][\w'-]*)*)\??"#, strip(aq)),
           normText(said).contains(normText(m.group(1)!)), !["A", "An", "The"].contains(String(m.group(1)!.split(separator: " ")[0])),
           (!truthy(p.albums[i].place) || normText(p.albums[i].place!) == normText(m.group(1)!)) {
            p.albums[i].place = m.group(1)!; p.albums[i].anchor = nil; p.albums[i].window = nil
        }
        if fullmatch(mediaQ, strip(p.albums[i].judgeQuestion ?? "")) != nil { p.albums[i].judgeQuestion = nil }
        if fullmatch(mediaQ, strip(p.albums[i].filterQuestion ?? "")) != nil { p.albums[i].filterQuestion = nil }
        if fullmatch(mediaQ, strip(p.albums[i].excludeQuestion ?? "")) != nil { p.albums[i].excludeQuestion = nil }
    }
    let wm = search(#"(?i)\b(?:the|that) (day|week|weekend|month|year) (?:when )?(?:i|we) (?:went|was|were|visited|flew|drove|traveled|travelled|stayed|got) (?:to|at|in) (?:the )?([\w' -]+?)(?=[,.!?]| and | no | without | but |$)"#, said)
    for i in p.albums.indices {
        if let wm = wm, let pl = p.albums[i].place, !pl.isEmpty, p.albums[i].anchor == nil, normText(pl) == normText(wm.group(2)!) {
            let w = ["day": "same_day", "week": "same_week", "weekend": "same_week", "month": "same_month", "year": "same_year"][wm.group(1)!.lowercased()]!
            p.albums[i].anchor = Step(looks: [pl], judgeQuestion: "Does this photo show \(pl)?"); p.albums[i].window = w; p.albums[i].place = nil
        }
    }
    if p.albums.count > 1 {
        let keep = p.albums.filter { a in truthy(a.person) || truthy(a.judgeQuestion) || !a.looks.isEmpty || truthy(a.dateFrom) || truthy(a.dateTo) ||
            truthy(a.place) || a.anchor != nil || a.media != "any" || truthy(a.filterQuestion) }
        if !keep.isEmpty && keep.count < p.albums.count {
            appendNote(&p, "[\(p.albums.count - keep.count) album(s) with no condition left out]"); p.albums = keep
        }
    }
    func sameExceptQ(_ a: Album, _ b: Album) -> Bool {
        var x = a, y = b
        for z in [0, 1] { _ = z }
        x.name = ""; y.name = ""; x.media = ""; y.media = ""; x.judgeQuestion = nil; y.judgeQuestion = nil
        x.filterQuestion = nil; y.filterQuestion = nil; x.excludeQuestion = nil; y.excludeQuestion = nil
        return x == y
    }
    func qkey(_ q: String?) -> String { normText(subLit(#"(?i)\b(photo|video|image|picture|clip)\b"#, q ?? "", "x")) }
    var merged = [Album]()
    for a in p.albums {
        if let j = merged.firstIndex(where: { b in Set([a.media, b.media]) == Set(["photo", "video"]) && sameExceptQ(a, b) &&
            qkey(a.judgeQuestion) == qkey(b.judgeQuestion) && qkey(a.filterQuestion) == qkey(b.filterQuestion) &&
            qkey(a.excludeQuestion) == qkey(b.excludeQuestion) }) {
            merged[j].media = "any"
            appendNote(&p, "['\(a.name)' merged into '\(merged[j].name)': same search, photos and videos]")
        } else { merged.append(a) }
    }
    p.albums = merged
    for i in p.albums.indices {
        if let jq = p.albums[i].judgeQuestion, let eq = p.albums[i].excludeQuestion, !jq.isEmpty, !eq.isEmpty, normText(jq) == normText(eq) {
            p.albums[i].judgeQuestion = nil; p.albums[i].looks = []
        }
        let w = p.albums[i].window ?? ""
        let explicit = fullmatch(#"(days|minutes)_(before|after):\d+"#, w) != nil
        if p.albums[i].anchor != nil && !["same_day", "same_week", "same_month", "same_year", "same_event", "same_place", "before", "after", "since", "until"].contains(w) && !explicit {
            p.albums[i].window = "same_event"
        }
        if p.albums[i].anchor != nil {
            if fullmatch(#"(days|minutes)_(before|after):\d+|before|after|since|until"#, p.albums[i].window ?? "") == nil {
                for (wn, pat) in windowWords where has(pat, said) { p.albums[i].window = wn; break }
            }
            if let pl = p.albums[i].place, !pl.isEmpty, let an = p.albums[i].anchor,
               normText((an.looks + [an.judgeQuestion ?? ""]).joined(separator: " ")).contains(normText(pl)) {
                p.albums[i].place = nil
            }
        }
        if truthy(p.albums[i].window) && p.albums[i].anchor == nil { p.albums[i].window = nil }
        if p.albums[i].until != nil && p.albums[i].anchor == nil {
            p.albums[i].anchor = p.albums[i].until; p.albums[i].window = "before"; p.albums[i].until = nil
        }
        if p.albums[i].until != nil && !["after", "since"].contains(p.albums[i].window ?? "") {
            p.albums[i].window = [nil, "same_event", "same_day"].contains(p.albums[i].window) ? "after" : "since"
        }
        if p.albums[i].anchor != nil, fullmatch(#"(days|minutes)_(before|after):\d+|before|after|since|until"#, p.albums[i].window ?? "") != nil,
           !has(#"(?i)\b(before|after|prior|earlier|later|following|leading up|since|until)\b"#, said) {
            p.albums[i].window = "same_event"
        }
        if let aq = p.albums[i].anchor?.judgeQuestion, !aq.isEmpty {
            let q = subLit(#"(?i)\b(the|a) person in the red box\b"#, aq, "someone")
            p.albums[i].anchor!.judgeQuestion = strip(subLit(#"(?i)\s*\bin the red box\b"#, q, ""))
        }
        for f in 0..<2 {
            guard let q = f == 0 ? p.albums[i].excludeQuestion : p.albums[i].filterQuestion, !q.isEmpty else { continue }
            let v = strip(subLit(#"\s{2,}"#, subLit(#"(?i)\s*\bin the red box\b"#, q.replacingOccurrences(of: "the person in the red box", with: "the person"), ""), " "))
            if f == 0 { p.albums[i].excludeQuestion = v } else { p.albums[i].filterQuestion = v }
        }
    }
    nameTheThing(&p, message)
    return p
}
