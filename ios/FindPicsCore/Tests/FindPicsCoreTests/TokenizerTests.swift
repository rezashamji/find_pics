import Foundation
import XCTest
@testable import FindPicsCore

struct TokCase: Decodable { let text: String; let ids: [Int32] }

final class TokenizerTests: XCTestCase {
    /// Same token ids as open_clip's SimpleTokenizer (PE-Core text tower input) on every fuzz-corpus request.
    func testMatchesOpenClip() throws {
        let url = Bundle.module.url(forResource: "tokens", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([TokCase].self, from: Data(contentsOf: url))
        let tok = try ClipTokenizer()
        var bad = 0
        for c in cases where tok(c.text) != c.ids {
            bad += 1
            if bad <= 8 { XCTFail("\(c.text.prefix(60)) -> \(tok(c.text).prefix(12)) vs \(c.ids.prefix(12))") }
        }
        XCTAssertEqual(bad, 0, "of \(cases.count)")
    }
}
