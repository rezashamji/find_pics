import XCTest
@testable import FindPicsCore

final class ThermalPaceTests: XCTestCase {
    func testDecisionPerState() {
        XCTAssertEqual(thermalPace(.nominal, secondsSinceCheck: 0), .judge)
        XCTAssertEqual(thermalPace(.fair, secondsSinceCheck: 500), .judge)
        XCTAssertEqual(thermalPace(.serious, secondsSinceCheck: 0), .wait(seconds: 20))       // just read: full poll
        XCTAssertEqual(thermalPace(.serious, secondsSinceCheck: 15), .wait(seconds: 5))       // read 15 s ago
        XCTAssertEqual(thermalPace(.serious, secondsSinceCheck: 90), .wait(seconds: 1))       // overdue: re-read soon
        XCTAssertEqual(thermalPace(.critical, secondsSinceCheck: 0), .stop)
        XCTAssertEqual(thermalPace(.serious, secondsSinceCheck: 0, poll: 30), .wait(seconds: 30))
    }

    /// A session: cool -> hot (pause, polled every 20 s for 60 s) -> cool again -> hot -> critical (stop).
    func testPacerCountsPausesAndTransitions() {
        var p = ThermalPacer(poll: 20)
        var t = 0.0
        XCTAssertEqual(p.step(.nominal, now: t), .judge); t += 3
        XCTAssertEqual(p.step(.fair, now: t), .judge); t += 3
        XCTAssertEqual(p.step(.serious, now: t), .wait(seconds: 17))     // last read 3 s ago
        for _ in 0..<3 { t += 20; _ = p.step(.serious, now: t) }
        XCTAssertEqual(p.pausedNow(t), 60, accuracy: 1e-9)
        t += 20
        XCTAssertEqual(p.step(.fair, now: t), .judge)                      // cooled: back to judging
        XCTAssertEqual(p.pauses, 1); XCTAssertEqual(p.pausedSeconds, 80, accuracy: 1e-9)
        t += 3; _ = p.step(.serious, now: t); t += 20
        XCTAssertEqual(p.step(.critical, now: t), .stop)
        XCTAssertEqual(p.pauses, 2); XCTAssertEqual(p.pausedSeconds, 100, accuracy: 1e-9)
        XCTAssertEqual(p.transitions.map { $0.level }, [.nominal, .fair, .serious, .fair, .serious, .critical])
        XCTAssertEqual(p.transitions.map { $0.at }, [0, 3, 6, 86, 89, 109])
    }

    /// The first answer is the round where fast mode would stop; the continuation that follows runs exhaustive mode's
    /// rounds, so its bounds are the ones testTimedNeverOverclaims checks.
    func testFirstAnswerIsWhereFastModeStops() async throws {
        let url = Bundle.module.url(forResource: "stream", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([StreamCase].self, from: Data(contentsOf: url))
        let params = phoneFastParams(accept: 0.7)
        for c in cases.prefix(3) {
            var fast: [Int] = [], cont: [Int] = [], firstAt: Int? = nil
            try await streamRounds(n: c.p.count, params: params, seed: 1, judge: { pos in pos.map { c.p[$0] } }) { r in
                fast.append(r.judged); return fastModeWantsMore(r, params: params)
            }
            try await streamRounds(n: c.p.count, params: params, seed: 1, judge: { pos in pos.map { c.p[$0] } }) { r in
                cont.append(r.judged)
                if firstAt == nil && firstAnswerDone(r, params: params) { firstAt = cont.count }
                return true
            }
            XCTAssertEqual(firstAt, fast.count, c.name)
            XCTAssertEqual(Array(cont.prefix(fast.count)), fast, c.name)       // same rounds up to the first answer
            XCTAssertEqual(cont.last, c.p.count, c.name)                          // the continuation judges everything
        }
    }
}
