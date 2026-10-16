import Foundation
import XCTest
@testable import FindPicsCore

/// RESULTS 68 (4): BertTokenizer = the Hugging Face fast tokenizer of all-MiniLM-L12-v2 (add_special_tokens=False) on 1,000
/// dev facts lines, every recorded dev scorer question and hand cases (Fixtures/bert_tokens.json, bert_vocab.txt).
final class BertTokenizerTests: XCTestCase {
    struct Case: Decodable { let text: String; let ids: [Int] }
    struct File: Decodable { let cases: [Case] }

    func testMatchesHuggingFace() throws {
        let vurl = Bundle.module.url(forResource: "bert_vocab", withExtension: "txt", subdirectory: "Fixtures")!
        let tok = try BertTokenizer(vocabURL: vurl)
        let url = Bundle.module.url(forResource: "bert_tokens", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
        var bad = 0, tokens = 0
        for c in f.cases {
            let got = tok.ids(c.text)
            tokens += c.ids.count
            if got != c.ids {
                bad += 1
                if bad <= 5 { print("BERT_TOK mismatch:", c.text.prefix(80), got.prefix(20), c.ids.prefix(20)) }
            }
        }
        print("BERT_TOK \(f.cases.count) texts, \(tokens) tokens: \(bad) texts differ")
        XCTAssertEqual(bad, 0)
        XCTAssertEqual(tok.factsInput("a dog").first, tok.cls)
        XCTAssertEqual(tok.questionInput(String(repeating: "dog ", count: 100)).count, 66)
    }
}
