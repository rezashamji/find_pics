import Foundation
import XCTest
@testable import FindPicsCore

struct AlignCase: Decodable { let lmk: [[Double]]; let M: [[Double]] }

final class FaceAlignTests: XCTestCase {
    /// Same transform as insightface estimate_norm (skimage SimilarityTransform) on 200 random rotated/scaled faces.
    func testMatchesInsightface() throws {
        let url = Bundle.module.url(forResource: "align", withExtension: "json", subdirectory: "Fixtures")!
        for c in try JSONDecoder().decode([AlignCase].self, from: Data(contentsOf: url)) {
            let M = similarityTransform(from: c.lmk.map { ($0[0], $0[1]) })
            for i in 0..<2 { for j in 0..<3 { XCTAssertEqual(M[i][j], c.M[i][j], accuracy: 1e-3 * max(1, abs(c.M[i][j]))) } }
        }
    }
}
