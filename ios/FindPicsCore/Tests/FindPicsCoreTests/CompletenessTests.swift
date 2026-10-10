import Foundation
import XCTest
@testable import FindPicsCore

struct CertCase: Decodable { let hits: Int; let n: Int; let alpha: Double; let cp: Double; let nTail: Int; let found: Int
    let recall_lower: Double?; let missed_upper: Double? }

final class CompletenessTests: XCTestCase {
    /// Same Clopper-Pearson bound and certificate as audit.py (scipy) on 300 random cases.
    func testMatchesScipy() throws {
        let url = Bundle.module.url(forResource: "certify", withExtension: "json", subdirectory: "Fixtures")!
        for c in try JSONDecoder().decode([CertCase].self, from: Data(contentsOf: url)) {
            XCTAssertEqual(cpUpper(hits: c.hits, n: c.n, alpha: c.alpha), c.cp, accuracy: 1e-9, "\(c.hits)/\(c.n) a=\(c.alpha)")
            let cert = certify(found: c.found, nTail: c.nTail, tailLabels: [Bool](repeating: true, count: c.hits) + [Bool](repeating: false, count: c.n - c.hits), alpha: c.alpha)
            if let m = c.missed_upper { XCTAssertEqual(cert.missedUpper, m, accuracy: max(1e-6, m * 1e-9)) }
            if let r = c.recall_lower, !r.isNaN { XCTAssertEqual(cert.recallLower, r, accuracy: 1e-9) }
        }
    }
}

final class CompletenessNoteTests: XCTestCase {
    func testPlainWords() {
        let all = certify(found: 12, nTail: 0, tailLabels: [])
        XCTAssertTrue(completenessNote(all, inScope: 300).hasPrefix("Checked all 300 photos"))
        let none = certify(found: 21, nTail: 200, tailLabels: Array(repeating: false, count: 200))
        XCTAssertTrue(completenessNote(none, inScope: 500).contains("Checked the 300 most likely of 500 photos"))
        let some = certify(found: 21, nTail: 16_515, tailLabels: Array(repeating: false, count: 150))
        let s = completenessNote(some, inScope: 16_965)
        XCTAssertTrue(s.contains("Checked the 450 most likely of 16,965 photos"), s)
        XCTAssertTrue(s.contains("more could be among the other 16,515 (95% sure)"), s)
        XCTAssertFalse(s.contains("%  of matches"))
    }
}
