import Foundation
import XCTest
@testable import FindPicsCore

/// RESULTS 68 (2): CascadeSession = the measured Python cascade (eval/partial65/cascade65.py, r59d, policy V), call by call,
/// on recordings of that harness (eval/cascade_port/record65.py -> fixture.py). Bar (pre-registered): 0 mismatches -
/// every pass-B output bit-equal, the judged set and the extension blocks equal.
/// Bundle fixture: Fixtures/cascade.json (stratified subset). Full dev + TRAIN: FP_CASCADE_FULL=<data/public/cascade_port/full>.
final class CascadeRouteTests: XCTestCase {
    struct Call: Decodable { let q: String; let crop: String; let ref: Bool; let excl: Bool; let sub: Int; let rows: [Int]
        let fr: [Int]; let out: [Double]? }
    struct JP: Decodable { let r: Int; let q: String; let c: String; let f: Int; let ref: Bool; let p: Double }
    struct JK: Decodable { let r: Int; let q: String; let c: String }
    struct SE: Decodable { let r: Int; let ms: Double; let f: Int }
    struct Req: Decodable {
        let qid: String; let user: String; let cat: String; let cfg: String; let error: String?; let subs: [[Int]]
        let A: [Call]; let B: [Call]; let tv: [String: [String: Double]]; let vec: [String: [String: Double]]
        let hasf: [Int]; let jp: [JP]; let judged: [JK]; let ext: [String: Int]; let scope: [String: [SE]]
    }
    struct Prefix: Decodable { let place: String?; let taken: Double?; let prefix: String }
    struct Fixture: Decodable { let libs: [String: [String]]; let requests: [Req]; let prefix: [Prefix] }
    struct LibFile: Decodable { let user: String; let place: [String]; let requests: [Req] }

    struct Tally { var requests = 0, calls = 0, callsA = 0, photos = 0, outMismatch = 0, callMismatch = 0, judgedMismatch = 0,
                   extMismatch = 0, scopeMismatch = 0, missing = 0, judgedPairs = 0,
                   refCalls = 0, subsetCalls = 0, exclCalls = 0, errored = 0, extended = 0, leafCalls = 0, fallbacks = 0, leaves = 0, bEqualsA = 0, bInsideA = 0, bOnly = 0
        var failures = [String]() }

    struct RunResult { var outs = [[Float]](); var outsA = [[Float]](); var judged = Set<JudgedKey>(); var ext = [String: Int]()
        var missingJudge = 0, missingScore = 0, scopeBad = [String](), leafCalls = 0, fallbacks = 0,
        leaves = 0, bEqualsA = 0, bInsideA = 0, bOnly = 0 }

    /// replay one recorded (request, config) with the given millisecond scores (default: the recorded ones)
    static func run(_ r: Req, places libPlaces: [String], tv: [String: [String: Double]], vec: [String: [String: Double]])
        async throws -> RunResult {
        var res = RunResult()
        let hasf = Set(r.hasf)
        var missJ = 0, missS = 0
        var jp = [String: Float]()
        for x in r.jp { jp["\(x.r)\u{1}\(x.q)\u{1}\(x.c)\u{1}\(x.f)\u{1}\(x.ref)"] = Float(x.p) }
        let s = CascadeSession(
            hasFacts: { hasf.contains($0) },
            towervec: { p, q in if let v = tv[q]?[String(p)] { return Float(v) }; missS += 1; return .nan },
            vecB: { p, q in if let v = vec[q]?[String(p)] { return Float(v) }; missS += 1; return .nan },
            judge: { items, q, c in
                items.map { it in
                    if let v = jp["\(it.photo)\u{1}\(q)\u{1}\(c)\u{1}\(it.face)\u{1}\(it.reference)"] { return v }
                    missJ += 1; return .nan
                }
            })
        // place index per photo set: the library (-1) or a recorded subset (its library rows in order)
        let libPI = PlaceIndex(libPlaces)
        let subPI = r.subs.map { rows in PlaceIndex(rows.map { libPlaces[$0] }) }
        let subPos = r.subs.map { rows in Dictionary(rows.enumerated().map { ($1, $0) }, uniquingKeysWith: { a, _ in a }) }
        func call(_ c: Call) -> CascadeCall {
            CascadeCall(question: c.q, crop: c.crop, reference: c.ref, exclude: c.excl, photos: c.rows,
                        local: c.sub < 0 ? c.rows : c.rows.map { subPos[c.sub][$0]! }, faces: c.fr,
                        places: c.sub < 0 ? libPI : subPI[c.sub])
        }
        // RESULTS 69 configs "<AJ|DRY|ON>_<precision>": DRY answers pass A from the millisecond scorer (dryAnswer, outputs
        // compared too), ON has no pass A (online scope, compared after pass B); every other config = the R68 recordScope
        let mode = r.cfg.components(separatedBy: "_").first ?? ""
        func checkScope() {     // scopes vs the harness's (the bookkeeping; outputs are the bar)
            for (k, v) in r.scope {
                let parts = k.components(separatedBy: "|")
                let key = CascadeSession.LeafKey(q: parts.dropLast().joined(separator: "|"), crop: parts.last!)
                let order = s.scopeOrder[key] ?? [], d = s.scope[key] ?? [:]
                if order != v.map({ $0.r }) || zip(v, order).contains(where: { Float($0.ms) != d[$1]!.ms || $0.f != d[$1]!.face }) {
                    res.scopeBad.append(k)
                }
            }
            if s.scope.count != r.scope.count { res.scopeBad.append("count") }
        }
        if mode == "ON" { s.online = true }
        else if mode == "DRY" { for c in r.A { res.outsA.append(s.dryAnswer(call(c))) } }
        else { for c in r.A { s.recordScope(call(c)) } }
        if mode != "ON" { checkScope() }
        s.beginAnswers()
        for c in r.B { res.outs.append(try await s.answer(call(c))) }
        if mode == "ON" { checkScope() }
        res.judged = s.judged; res.ext = s.extensionJudged; res.leafCalls = s.judgeLeafCalls; res.fallbacks = s.fallbackLeafCalls;
        for (k, b) in s.askedInB { res.leaves += 1
            if let a = s.scope[k].map({ Set($0.keys) }) { if a == b { res.bEqualsA += 1 } else if b.isSubset(of: a) { res.bInsideA += 1 } } else { res.bOnly += 1 } }
        res.missingJudge = missJ; res.missingScore = missS
        return res
    }

    /// replay one recorded (request, config) with its recorded scores and compare with the harness
    static func replay(_ r: Req, places: [String], into t: inout Tally) async throws {
        let res = try await run(r, places: places, tv: r.tv, vec: r.vec)
        if !res.scopeBad.isEmpty {
            t.scopeMismatch += res.scopeBad.count
            if t.failures.count < 20 { t.failures.append("\(r.qid) \(r.cfg) scope \(res.scopeBad.prefix(2))") }
        }
        for (i, c) in r.A.enumerated() where i < res.outsA.count {     // DRY pass-A outputs (R69)
            let out = res.outsA[i], want = (c.out ?? []).map { Float($0) }
            t.callsA += 1
            let bad = out.count != want.count ? max(out.count, want.count)
                : zip(out, want).filter { $0.0.bitPattern != $0.1.bitPattern }.count
            if bad > 0 {
                t.outMismatch += bad; t.callMismatch += 1
                if t.failures.count < 20 { t.failures.append("\(r.qid) \(r.cfg) A[\(i)] \(c.q.prefix(70)) \(bad)/\(want.count)") }
            }
        }
        for (i, c) in r.B.enumerated() {
            let out = res.outs[i], want = (c.out ?? []).map { Float($0) }
            t.calls += 1; t.photos += want.count
            let bad = out.count != want.count ? max(out.count, want.count)
                : zip(out, want).filter { $0.0.bitPattern != $0.1.bitPattern }.count
            if bad > 0 {
                t.outMismatch += bad; t.callMismatch += 1
                if t.failures.count < 20 { t.failures.append("\(r.qid) \(r.cfg) B[\(i)] \(c.q.prefix(70)) \(bad)/\(want.count)") }
            }
        }
        let want = Set(r.judged.map { JudgedKey(photo: $0.r, question: $0.q, crop: $0.c) })
        t.judgedPairs += want.count
        if res.judged != want {
            t.judgedMismatch += 1
            if t.failures.count < 20 { t.failures.append("\(r.qid) \(r.cfg) judged \(res.judged.count) vs \(want.count)") }
        }
        if res.ext != r.ext {
            t.extMismatch += 1
            if t.failures.count < 20 { t.failures.append("\(r.qid) \(r.cfg) ext \(res.ext) vs \(r.ext)") }
        }
        t.missing += res.missingJudge + res.missingScore
        t.leafCalls += res.leafCalls; t.fallbacks += res.fallbacks
        t.leaves += res.leaves; t.bEqualsA += res.bEqualsA; t.bInsideA += res.bInsideA; t.bOnly += res.bOnly
        t.requests += 1
        t.refCalls += r.B.filter { $0.ref }.count; t.subsetCalls += r.B.filter { $0.sub >= 0 }.count
        t.exclCalls += r.B.filter { $0.excl }.count
        if r.error != nil { t.errored += 1 }
        if r.ext.contains(where: { k, v in v > judgeBudget(r.scope[k]?.count ?? 0) }) { t.extended += 1 }
    }

    static func report(_ name: String, _ t: Tally) -> String {
        "\(name): \(t.requests) (request, config), \(t.calls) pass-B calls, \(t.callsA) DRY pass-A calls, \(t.photos) pass-B photo answers: output mismatches " +
        "\(t.outMismatch) (in \(t.callMismatch) calls), judged-set mismatches \(t.judgedMismatch) (\(t.judgedPairs) judged " +
        "pairs), extension mismatches \(t.extMismatch), scope mismatches \(t.scopeMismatch), missing inputs \(t.missing); " +
        "covers \(t.refCalls) reference-photo calls, \(t.subsetCalls) subset calls, \(t.exclCalls) exclude calls, " +
        "\(t.extended) requests with a budget extension, \(t.errored) harness-errored requests; judge-leaf evaluations " +
        "\(t.leafCalls), of which without a pass-A scope \(t.fallbacks); judge leaves \(t.leaves): photos pass B asked = " +
        "pass-A scope \(t.bEqualsA), a strict subset \(t.bInsideA), leaf not in pass A \(t.bOnly), other \(t.leaves - t.bEqualsA - t.bInsideA - t.bOnly)"
    }

    func testMatchesRecordedHarness() async throws {
        let url = Bundle.module.url(forResource: "cascade", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        var t = Tally()
        for r in f.requests { try await Self.replay(r, places: f.libs[r.user]!, into: &t) }
        print("CASCADE_PARITY " + Self.report("bundle", t))
        XCTAssertGreaterThan(t.requests, 20)
        XCTAssertEqual(t.outMismatch + t.judgedMismatch + t.extMismatch + t.scopeMismatch + t.missing, 0, t.failures.joined(separator: "\n"))
    }

    /// every recorded dev + TRAIN request (data/public/cascade_port/full, not in the bundle); runs only when asked
    func testMatchesRecordedHarnessFull() async throws {
        guard let dir = ProcessInfo.processInfo.environment["FP_CASCADE_FULL"] else { return }
        for split in ["dev", "train"] {
            var t = Tally()
            let d = URL(fileURLWithPath: dir).appendingPathComponent(split)
            let files = try FileManager.default.contentsOfDirectory(atPath: d.path).filter { $0.hasSuffix(".json") }.sorted()
            for fn in files {
                let L = try JSONDecoder().decode(LibFile.self, from: Data(contentsOf: d.appendingPathComponent(fn)))
                for r in L.requests { try await Self.replay(r, places: L.place, into: &t) }
            }
            print("CASCADE_PARITY " + Self.report("full \(split)", t))
            for x in t.failures { print("CASCADE_FAIL \(x)") }
            XCTAssertEqual(t.outMismatch + t.judgedMismatch + t.extMismatch + t.scopeMismatch + t.missing, 0)
        }
    }

    /// RESULTS 69: the DRY / ON scope sources (dryAnswer, online) on recordings of eval/scope69/cascade69.py
    /// (eval/scope69/fixture69.py): bundle Fixtures/cascade69.json always; every dev + TRAIN request with FP_CASCADE69_FULL.
    func testScopeSourcesMatchRecordedHarness() async throws {
        let url = Bundle.module.url(forResource: "cascade69", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        var t = Tally()
        for r in f.requests { try await Self.replay(r, places: f.libs[r.user]!, into: &t) }
        print("CASCADE69_PARITY " + Self.report("bundle", t))
        XCTAssertGreaterThan(t.requests, 10)
        XCTAssertGreaterThan(t.callsA, 0)
        XCTAssertEqual(t.outMismatch + t.judgedMismatch + t.extMismatch + t.scopeMismatch + t.missing, 0, t.failures.joined(separator: "\n"))
    }

    func testScopeSourcesMatchRecordedHarnessFull() async throws {
        guard let dir = ProcessInfo.processInfo.environment["FP_CASCADE69_FULL"] else { return }
        for split in ["dev", "train"] {
            var byMode = [String: Tally]()
            let d = URL(fileURLWithPath: dir).appendingPathComponent(split)
            let files = try FileManager.default.contentsOfDirectory(atPath: d.path).filter { $0.hasSuffix(".json") }.sorted()
            for fn in files {
                let L = try JSONDecoder().decode(LibFile.self, from: Data(contentsOf: d.appendingPathComponent(fn)))
                for r in L.requests { try await Self.replay(r, places: L.place, into: &byMode[r.cfg, default: Tally()]) }
            }
            for (m, t) in byMode.sorted(by: { $0.key < $1.key }) {
                print("CASCADE69_PARITY " + Self.report("full \(split) \(m)", t))
                for x in t.failures { print("CASCADE69_FAIL \(x)") }
                XCTAssertEqual(t.outMismatch + t.judgedMismatch + t.extMismatch + t.scopeMismatch + t.missing, 0)
            }
        }
    }

    func testFactsPrefix() throws {
        let url = Bundle.module.url(forResource: "cascade", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        var bad = 0
        for c in f.prefix {
            let got = factsPrefix(place: c.place, taken: c.taken.map { Date(timeIntervalSince1970: $0) })
            if got != c.prefix { bad += 1; if bad < 5 { print("PREFIX", c.place ?? "nil", got, "|", c.prefix) } }
        }
        XCTAssertEqual(bad, 0, "\(bad)/\(f.prefix.count)")
        XCTAssertGreaterThan(f.prefix.count, 1000)
    }

    func testRulesByHand() {
        XCTAssertEqual(judgeBudget(10), 10)
        XCTAssertEqual(judgeBudget(1000), 50)
        XCTAssertEqual(judgeBudget(1001), 51)          // ceil(50.05)
        XCTAssertEqual(judgeBudget(1990), 100)
        XCTAssertEqual(judgeBudget(187_084), 9355)
        XCTAssertEqual(cascadeRouteKind("Is there a real dog anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?", crop: "False"), "")
        XCTAssertEqual(cascadeRouteKind("Is the person in the red box smiling?", crop: "False"), "red-box person")
        XCTAssertEqual(cascadeRouteKind("Is there a beach?", crop: "box"), "red-box person")
        XCTAssertEqual(cascadeRouteKind("Are there two dogs?", crop: "False"), "count")
        XCTAssertEqual(cascadeRouteKind("Is this a sunset photo?", crop: "False"), "photo look")
        XCTAssertEqual(cascadeRouteKind("Is there a red car?", crop: "False"), "colour")
        XCTAssertEqual(cascadeLeafKind("Is this a photo from a safari trip?", crop: "False", exclude: false), "trip-no-place")
        XCTAssertEqual(cascadeLeafKind("Is the bottle on a kitchen counter?", crop: "False", exclude: false), "")
        XCTAssertEqual(cascadeLeafKind("Is there a bottle?", crop: "False", exclude: true), "exclude")
        XCTAssertEqual(scorerQuestion("PERSON[Sam]::Is the person in the red box wearing a hat?"), "Is a person wearing a hat?")
        XCTAssertEqual(scorerQuestion("Is the dog in the red box sleeping? "), "Is the dog sleeping?")
    }
}
