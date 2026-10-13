import Foundation
import XCTest
@testable import FindPicsCore

struct WinCase: Decodable { let anchors: [Int]; let window: String?; let rows: [Int] }
struct WinFixture: Decodable { let taken: [Double?]; let place: [String]; let events: [Int]; let cases: [WinCase] }

final class WindowsTests: XCTestCase {
    /// Same events and window rows as agent.events / agent.window_rows (900 cases, 15 window kinds).
    func testMatchesPython() throws {
        let url = Bundle.module.url(forResource: "windows", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(WinFixture.self, from: Data(contentsOf: url))
        let ev = events(f.taken)
        // event ids may be numbered differently only if the grouping differs: compare the partitions
        XCTAssertEqual(Dictionary(grouping: f.taken.indices, by: { ev[$0] }).values.map { Set($0) }.sorted { $0.min()! < $1.min()! },
                       Dictionary(grouping: f.taken.indices, by: { f.events[$0] }).values.map { Set($0) }.sorted { $0.min()! < $1.min()! })
        var bad = 0
        for c in f.cases where windowRows(taken: f.taken, place: f.place, anchors: c.anchors, window: c.window) != c.rows {
            bad += 1; if bad <= 5 { XCTFail("\(c.window ?? "nil") anchors \(c.anchors)") }
        }
        XCTAssertEqual(bad, 0)
    }

    /// agent.one_day_anchors (tests/test_agent.py::test_one_day_anchors_keep_the_best_day_only_when_it_stands_out)
    func testOneDayAnchors() {
        func sec(_ iso: String, _ h: Double) -> Double { Double(Day(iso: iso)!.serial) * 86400 + h * 3600 }
        let t: [Double?] = [sec("2011-02-18", 10), sec("2011-02-18", 12), sec("2010-03-08", 10), sec("2010-03-13", 10), nil]
        XCTAssertEqual(oneDayAnchors(taken: t, anchors: [0, 2, 3], scores: [0.98, 0.84, 0.80], window: "same_day"), [0])
        XCTAssertEqual(oneDayAnchors(taken: t, anchors: [2, 0, 1, 3], scores: [0.84, 0.98, 0.70, 0.95], window: "same_day"), [0, 1, 3])
        XCTAssertEqual(oneDayAnchors(taken: t, anchors: [0, 2], scores: [0.99, 0.99], window: "same_day"), [0, 2])
        XCTAssertEqual(oneDayAnchors(taken: t, anchors: [0, 2, 3], scores: [0.98, 0.84, 0.80], window: "same_event"), [0, 2, 3])
        XCTAssertEqual(oneDayAnchors(taken: t, anchors: [0, 1], scores: [0.98, 0.5], window: "same_day"), [0, 1])
        XCTAssertEqual(oneDayAnchors(taken: t, anchors: [0, 4, 2], scores: [0.98, 0.99, 0.84], window: "same_day"), [0])
        XCTAssertEqual(oneDayAnchors(taken: t, anchors: [0], scores: [0.98], window: "same_day"), [0])
    }
}
