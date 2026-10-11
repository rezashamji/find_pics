import Foundation
import XCTest
@testable import FindPicsCore

struct StreamCase: Decodable { let name: String; let p: [Double] }

final class StreamingTests: XCTestCase {
    /// Replay of recorded judge answers (public library, 19,218 photos, 6 concepts, 5 seeds): the stated completeness
    /// lower bound must never exceed the TRUE completeness (found / all judge-yes) at any round; the last round finds all.
    func testBoundNeverOverclaims() async throws {
        let url = Bundle.module.url(forResource: "stream", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([StreamCase].self, from: Data(contentsOf: url))
        var rounds = 0, over = 0
        for c in cases {
            let truth = c.p.filter { $0 >= 0.7 }.count
            for seed in 0..<5 {
                var lastFound = 0
                try await streamRounds(n: c.p.count, seed: UInt64(seed), judge: { pos in pos.map { c.p[$0] } }) { r in
                    rounds += 1
                    let trueRecall = Double(r.found.count) / Double(truth)
                    if r.certificate.recallLower > trueRecall + 1e-9 { over += 1 }
                    lastFound = r.found.count
                    return true
                }
                XCTAssertEqual(lastFound, truth, "\(c.name) seed \(seed): the last round must have judged everything")
            }
        }
        print("STREAM rounds \(rounds), overclaims \(over)")
        XCTAssertEqual(over, 0)
    }

    /// Golden case shared with tests/test_engine.py::test_head_stop_window (RESULTS 41): 30 yes in 0-29, 3 in 50-99,
    /// none in 100-149, 2 in 150-199. Window 50 (old rule) stops at 150; window 100 sees 3/100 at 150 and stops at 200.
    func testHeadWindow() async throws {
        var p = [Double](repeating: 0, count: 1000)
        for i in Array(0..<30) + [60, 70, 80, 160, 170] { p[i] = 1 }
        for (window, want) in [(nil, 150), (100, 200)] as [(Int?, Int)] {
            var params = StreamParams(); params.headSize = 150; params.headChunk = 50; params.headMax = 2000
            params.tailBudget = 0; params.headWindow = window
            var head = -1
            try await streamRounds(n: p.count, params: params, judge: { pos in pos.map { p[$0] } }) { r in head = r.nHead; return false }
            XCTAssertEqual(head, want, "window \(String(describing: window))")
        }
    }

    /// The phone's fast mode (Search.swift params + fastModeWantsMore) on the same replay: the bound shown after an
    /// automatic round 2 must not overclaim either, and the auto round must stop at autoRounds.
    func testPhoneFastModeAutoRoundNeverOverclaims() async throws {
        let url = Bundle.module.url(forResource: "stream", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([StreamCase].self, from: Data(contentsOf: url))
        var params = StreamParams(); params.headSize = 150; params.headChunk = 50; params.headMax = 2000; params.tailBudget = 150
        params.headWindow = 100; params.autoRoundHits = 3; params.autoRounds = 2
        let fixed = params
        var rounds = 0, over = 0, auto = 0
        for c in cases {
            let truth = c.p.filter { $0 >= 0.7 }.count
            for seed in 0..<5 {
                try await streamRounds(n: c.p.count, params: fixed, seed: UInt64(seed), judge: { pos in pos.map { c.p[$0] } }) { r in
                    rounds += 1
                    if r.k > 1 { auto += 1 }
                    XCTAssertLessThanOrEqual(r.k, fixed.autoRounds)
                    if r.certificate.recallLower > Double(r.found.count) / Double(truth) + 1e-9 { over += 1 }
                    return fastModeWantsMore(r, params: fixed)
                }
            }
        }
        print("PHONE FAST rounds \(rounds) (auto round 2: \(auto)), overclaims \(over)")
        XCTAssertEqual(over, 0)
    }

    /// Timed rounds (RESULTS 43) == the Python replay that chose them (eval/tune_stop_time.py sim_timed): same calls in
    /// the same order, same round ends, samples, hits, found and bound, fast and exhaustive, permutation passed in.
    func testTimedMatchesPythonReplay() async throws {
        struct Run: Decodable { let calls, hits, sampled, ntail, found, order: [Int]; let upper: [Double] }
        struct Case: Decodable { let name: String; let p: [Double]; let perm: [Int]; let fast, exhaustive: Run }
        struct Rule: Decodable {
            let head, chunk, window, head_max, tail, round_calls, inter, hits, extra, topup: Int; let rate, a1: Double
            let fast_stop, stop_topup: Int?
        }
        struct Run1: Decodable { let rule: Rule; let cases: [Case] }
        struct File: Decodable { let runs: [Run1] }
        let url = Bundle.module.url(forResource: "stream_timed", withExtension: "json", subdirectory: "Fixtures")!
        let file = try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
        XCTAssertGreaterThanOrEqual(file.runs.count, 2)
        for f in file.runs {
        var p = StreamParams(); p.headSize = f.rule.head; p.headChunk = f.rule.chunk; p.headWindow = f.rule.window
        p.headStopRate = f.rule.rate; p.headMax = f.rule.head_max; p.tailBudget = f.rule.tail; p.roundCalls = f.rule.round_calls
        p.interleave = f.rule.inter; p.autoRoundHits = f.rule.hits; p.autoExtraRounds = f.rule.extra; p.roundTopUp = f.rule.topup
        p.firstRoundAlphaShare = f.rule.a1
        p.fastStopCalls = f.rule.fast_stop ?? 0; p.fastStopTopUp = f.rule.stop_topup ?? 0
        let params = p
        for c in f.cases {
            for (mode, want) in [("fast", c.fast), ("exhaustive", c.exhaustive)] {
                var order: [Int] = [], rounds: [Round] = []
                try await streamTimed(n: c.p.count, perm: c.perm, params: params, judge: { pos in
                    order += pos; return pos.map { c.p[$0] }
                }) { r in rounds.append(r); return mode == "exhaustive" || fastModeWantsMore(r, params: params) }
                let tag = "\(c.name) \(mode)"
                XCTAssertEqual(rounds.map { $0.judged }, want.calls, tag)
                XCTAssertEqual(rounds.map { $0.certificate.tailHits }, want.hits, tag)
                XCTAssertEqual(rounds.map { $0.certificate.tailSampled }, want.sampled, tag)
                XCTAssertEqual(rounds.map { $0.certificate.nTail }, want.ntail, tag)
                XCTAssertEqual(rounds.map { $0.found.count }, want.found, tag)
                XCTAssertEqual(Array(order.prefix(80)), want.order, tag)
                for (r, u) in zip(rounds, want.upper) { XCTAssertEqual(r.certificate.missedUpper, u, accuracy: 1e-4, tag) }
            }
        }
        }
    }

    /// The phone's timed fast mode (Search.swift params) and exhaustive mode on the public replay: no bound shown at any
    /// round may overclaim (recallLower <= found / all judge-yes), and exhaustive's last round has judged everything.
    func testTimedNeverOverclaims() async throws {
        let url = Bundle.module.url(forResource: "stream", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([StreamCase].self, from: Data(contentsOf: url))
        let params = phoneFastParams(accept: 0.7)
        var rounds = 0, over = 0
        for c in cases {
            let truth = c.p.filter { $0 >= 0.7 }.count
            for seed in 0..<5 {
                for exhaustive in [false, true] {
                    var lastFound = 0, lastJudged = 0
                    try await streamRounds(n: c.p.count, params: params, seed: UInt64(seed), judge: { pos in pos.map { c.p[$0] } }) { r in
                        rounds += 1
                        if r.certificate.recallLower > Double(r.found.count) / Double(truth) + 1e-9 { over += 1 }
                        lastFound = r.found.count; lastJudged = r.judged
                        return exhaustive || fastModeWantsMore(r, params: params)
                    }
                    if exhaustive { XCTAssertEqual(lastFound, truth); XCTAssertEqual(lastJudged, c.p.count) }
                }
            }
        }
        print("TIMED rounds \(rounds), overclaims \(over)")
        XCTAssertEqual(over, 0)
    }

    /// Fast mode's call budget (RESULTS 44): the run ends with the first round that reaches fastStopCalls; every round
    /// before it is the same round (calls, sample, hits, bound) the unbudgeted run shows, and the last one has the same
    /// head with its random sample topped up to fastStopTopUp (or the whole rest).
    func testFastStopEndsAfterTheSameRounds() async throws {
        let url = Bundle.module.url(forResource: "stream", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([StreamCase].self, from: Data(contentsOf: url))
        let budgeted = phoneFastParams(accept: 0.7)
        XCTAssertEqual(budgeted.fastStopCalls, 480); XCTAssertEqual(budgeted.fastStopTopUp, 150)
        var free = budgeted; free.fastStopCalls = 0; free.fastStopTopUp = 0
        let unbudgeted = free
        var cut = 0
        for c in cases {
            for seed in 0..<5 {
                var a: [Round] = [], b: [Round] = []
                try await streamRounds(n: c.p.count, params: budgeted, seed: UInt64(seed), judge: { pos in pos.map { c.p[$0] } }) { r in
                    a.append(r); return fastModeWantsMore(r, params: budgeted)
                }
                try await streamRounds(n: c.p.count, params: unbudgeted, seed: UInt64(seed), judge: { pos in pos.map { c.p[$0] } }) { r in
                    b.append(r); return fastModeWantsMore(r, params: unbudgeted)
                }
                XCTAssertLessThanOrEqual(a.count, b.count)
                for (x, y) in zip(a.dropLast(), b) {
                    XCTAssertEqual(x.judged, y.judged); XCTAssertEqual(x.certificate.tailHits, y.certificate.tailHits)
                    XCTAssertEqual(x.certificate.missedUpper, y.certificate.missedUpper, accuracy: 1e-9)
                }
                XCTAssertTrue(a.dropLast().allSatisfy { $0.judged < 480 }, "\(c.name): only the last round may reach the budget")
                let x = a.last!, y = b[a.count - 1]
                XCTAssertEqual(x.nHead, y.nHead)
                if x.judged >= 480 {
                    XCTAssertGreaterThanOrEqual(x.certificate.tailSampled, min(150, x.certificate.nTail), c.name)
                    if a.count < b.count { cut += 1 }
                }
            }
        }
        print("FAST STOP: \(cut) of \(cases.count * 5) runs ended by the call budget")
        XCTAssertGreaterThan(cut, 0)
    }

    /// The prefetch hint (RESULTS 44): asking for it changes nothing about the run, its first position is always the
    /// one about to be judged, and the next position judged was in the previous hint for nearly every call.
    func testLookaheadHintIsRightAndChangesNothing() async throws {
        struct Case: Decodable { let name: String; let p: [Double]; let perm: [Int] }
        struct Run1: Decodable { let cases: [Case] }
        struct File: Decodable { let runs: [Run1] }
        let url = Bundle.module.url(forResource: "stream_timed", withExtension: "json", subdirectory: "Fixtures")!
        let file = try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
        let params = phoneFastParams(accept: 0.7)
        var calls = 0, predicted = 0
        for c in file.runs[0].cases {
            for exhaustive in [false, true] {
                var plain: [Int] = [], hinted: [Int] = [], hints: [[Int]] = []
                var ra: [Int] = [], rb: [Int] = []
                try await streamTimed(n: c.p.count, perm: c.perm, params: params, judge: { pos in
                    plain += pos; return pos.map { c.p[$0] }
                }) { r in ra.append(r.judged); return exhaustive || fastModeWantsMore(r, params: params) }
                try await streamTimed(n: c.p.count, perm: c.perm, params: params, lookahead: 2, upcoming: { hints.append($0) },
                                      judge: { pos in hinted += pos; return pos.map { c.p[$0] } }) { r in
                    rb.append(r.judged); return exhaustive || fastModeWantsMore(r, params: params)
                }
                XCTAssertEqual(plain, hinted, c.name); XCTAssertEqual(ra, rb, c.name)
                XCTAssertEqual(hints.count, hinted.count)
                for (i, h) in hints.enumerated() {
                    XCTAssertEqual(h.first, hinted[i]); XCTAssertLessThanOrEqual(h.count, 3)
                    XCTAssertEqual(Set(h).count, h.count)
                    if i + 1 < hinted.count { calls += 1; if h.dropFirst().contains(hinted[i + 1]) { predicted += 1 } }
                }
            }
        }
        let rate = Double(predicted) / Double(max(calls, 1))
        print("LOOKAHEAD next photo predicted \(predicted) of \(calls) (\(rate))")
        XCTAssertGreaterThan(rate, 0.95)
    }
}
