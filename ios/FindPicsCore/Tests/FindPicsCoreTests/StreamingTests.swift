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
}
