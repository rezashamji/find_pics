import Foundation
import XCTest
@testable import FindPicsCore

/// RESULTS 61d: repeat-stop for the facts writer. Golden answers from findpics.facts_line on hand cases + real 2B-student
/// lines on public dev photos (eval/facts_writer/facts_line_fixture.py regenerates the file).
private struct FLFixture: Decodable {
    struct Case: Decodable { let version: Int; let text: String; let scan: [Int]?; let kept: String; let stopped: Bool; let streamStop: Int? }
    let cases: [Case]
}

final class FactsLineTests: XCTestCase {
    private func fixture() throws -> FLFixture {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "facts_line", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(FLFixture.self, from: Data(contentsOf: url))
    }

    func testScanMatchesPython() throws {
        let f = try fixture()
        XCTAssertGreaterThan(f.cases.count, 500)
        var bad = 0
        for c in f.cases {
            let s = FactsLine.repeatScan(c.text, version: c.version)
            let got = s.map { [$0.cut, $0.end] }
            if got != c.scan { bad += 1; if bad < 5 { XCTFail("scan \(String(describing: got)) != \(String(describing: c.scan)) for \(c.text)") } }
            let (kept, stopped) = FactsLine.applyRepeatStop(c.text, version: c.version)
            XCTAssertEqual(kept, c.kept, c.text)
            XCTAssertEqual(stopped, c.stopped, c.text)
        }
        XCTAssertEqual(bad, 0)
    }

    /// Streaming: checking every prefix and stopping at the first hit gives the same line as cutting the full output.
    func testStreamedStopMatchesPython() throws {
        for c in try fixture().cases {
            let u = Array(c.text.unicodeScalars)
            var first: Int? = nil
            for n in 0...u.count where FactsLine.shouldStop(String(String.UnicodeScalarView(u[..<n])), version: c.version) { first = n; break }
            XCTAssertEqual(first, c.streamStop, c.text)
            if let n = first {
                XCTAssertEqual(FactsLine.applyRepeatStop(String(String.UnicodeScalarView(u[..<n])), version: c.version).line, c.kept, c.text)
            }
        }
    }

    func testHandCases() {
        XCTAssertEqual(FactsLine.applyRepeatStop("scene: x; objects: kilt, kilt, kilt, kil").line, "scene: x; objects: kilt")
        XCTAssertFalse(FactsLine.applyRepeatStop("objects: a, b; colours: a, b").stopped)   // a new field resets
        XCTAssertFalse(FactsLine.shouldStop("objects: kilt, kilt"))                            // not completed yet
        XCTAssertTrue(FactsLine.shouldStop("objects: kilt, kilt,"))
        // v2: an unnamed ";" list is one field; the 3rd occurrence stops
        XCTAssertEqual(FactsLine.applyRepeatStop("a; mask; gloves; mask; gloves; mask; x", version: 2).line, "a; mask; gloves; mask; gloves")
        XCTAssertFalse(FactsLine.shouldStop("objects: kilt, kilt,", version: 2))
    }
}
