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
