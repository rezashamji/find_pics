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
}
