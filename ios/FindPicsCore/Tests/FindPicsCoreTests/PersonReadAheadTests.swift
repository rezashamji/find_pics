import XCTest
@testable import FindPicsCore

/// The app reads person crops ahead along `lookWalkOrder` (PersonSearch.swift, 11-22) while streamPersonLook decides
/// what is judged. The hint is only useful if both walk the SAME order with the same arguments; if they ever drift
/// apart, every read-ahead is wasted and person searches silently fall back to one read at a time.
final class PersonReadAheadTests: XCTestCase {
    func testReadAheadWalkIsTheJudgedOrder() async throws {
        let ids = (0..<60).map { "p\($0)" }
        let eventOf = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, "e\($0 / 7)") })
        let seed = Int(truncatingIfNeeded: fnv1a64("Reza"))
        let walk = lookWalkOrder(ids: ids, eventOf: eventOf, seed: seed).map { ids[$0] }
        var judged = [String]()
        try await streamPersonLook(ids: ids, eventOf: eventOf, questions: 1, seed: seed, judge: { id, _ in
            judged.append(id); return 0.5
        }, onUpdate: { _ in true })
        XCTAssertEqual(judged, walk)
        XCTAssertEqual(Set(judged).count, ids.count)
    }
}
