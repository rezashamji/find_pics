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
}
