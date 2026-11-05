import Foundation
import XCTest
@testable import FindPicsCore

/// 11-05: the phone's facts text encoder path, everything that needs no MLX. Fixtures/facts_encoder_golden.json comes from
/// eval/cascade_port/encoder_golden.py (public dev photos, product writer lines + repeat-stop v2, recorded dev scorer
/// questions, edge cases); the Mac's #factsparity reads the same file and compares the device's MLX vectors with it.
///  - token ids: BertTokenizer built from the vocab.txt the app bundles next to towervec_encoder_fp16 (= what FactsWriter
///    loads), factsInput / questionInput = the scorers' Hugging Face ids, incl. the 254 / 64 word-piece caps
///  - factsPrefix(place:taken:) = the cluster's prefix (data.py --meta=1), and prefix + line = the text that was encoded
///  - the app's towervec head (ScorerHead and the per-search FoldedScorerHead) on the fixture's fp32 reference vectors =
///    the PyTorch fp32 P
///  - the fixture's second reference (MLX, fp16 export) is itself inside the bar the device is held to (cosine >= .9999)
final class FactsEncoderGoldenTests: XCTestCase {
    struct Fact: Decodable {
        let name: String; let kind: String; let place: String?; let taken: Double?; let prefix: String; let line: String
        let text: String; let ids: [Int]; let fp32: [Float]; let mlx_fp16: [Float]?; let v: [Float]?
    }
    struct Question: Decodable {
        let name: String; let question: String; let ids: [Int]; let fp32: [Float]; let mlx_fp16: [Float]?; let t: [Float]?
    }
    struct Pair: Decodable { let fact: Int; let question: Int; let p: Double; let p_mlx: Double? }
    struct File: Decodable {
        let head_file: String; let head_bytes: Int; let encoder_safetensors_bytes: Int
        let facts: [Fact]; let questions: [Question]; let pairs: [Pair]
    }

    /// The app's bundled facts models (ios/FindPicsApp.swiftpm/Sources/Models/facts), found from this file's location.
    static let models = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("FindPicsApp.swiftpm/Sources/Models/facts")

    func golden() throws -> File {
        let url = Bundle.module.url(forResource: "facts_encoder_golden", withExtension: "json", subdirectory: "Fixtures")!
        return try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var ab = 0.0, aa = 0.0, bb = 0.0
        for (x, y) in zip(a, b) { ab += Double(x) * Double(y); aa += Double(x) * Double(x); bb += Double(y) * Double(y) }
        return ab / (aa.squareRoot() * bb.squareRoot())
    }

    func testTokenIdsMatchPython() throws {
        let g = try golden()
        let tok = try BertTokenizer(vocabURL: Self.models.appendingPathComponent("towervec_encoder_fp16/vocab.txt"))
        var bad = 0
        for f in g.facts {
            let got = tok.factsInput(f.text)
            if got != f.ids {
                bad += 1
                print("FACTS_GOLDEN ids mismatch [\(f.name)] text \(f.text.debugDescription)\n  swift  \(got)\n  python \(f.ids)")
            }
        }
        for q in g.questions {
            let got = tok.questionInput(q.question)
            if got != q.ids {
                bad += 1
                print("FACTS_GOLDEN ids mismatch [\(q.name)] question \(q.question.debugDescription)\n  swift  \(got)\n  python \(q.ids)")
            }
        }
        print("FACTS_GOLDEN ids: \(g.facts.count) facts texts + \(g.questions.count) questions, \(bad) differ")
        XCTAssertEqual(bad, 0)
        // the fixture must exercise both caps and the empty text
        XCTAssertTrue(g.facts.contains { $0.ids.count == 256 }, "no facts text reaches the 254 word-piece cap")
        XCTAssertTrue(g.questions.contains { $0.ids.count == 66 }, "no question reaches the 64 word-piece cap")
        XCTAssertTrue(g.facts.contains { $0.ids == [tok.cls, tok.sep] }, "no empty facts text")
        XCTAssertTrue(g.facts.contains { $0.text.unicodeScalars.contains("\u{FFFD}") }, "no U+FFFD case")
    }

    func testPrefixesMatchPython() throws {
        let g = try golden()
        var bad = 0
        for f in g.facts {
            let got = factsPrefix(place: f.place, taken: f.taken.map { Date(timeIntervalSince1970: $0) })
            if got != f.prefix || got + f.line != f.text {
                bad += 1
                print("FACTS_GOLDEN prefix mismatch [\(f.name)] place \(f.place ?? "nil") taken \(f.taken.map { "\($0)" } ?? "nil")\n" +
                      "  swift  \(got.debugDescription)\n  python \(f.prefix.debugDescription)")
            }
        }
        print("FACTS_GOLDEN prefixes: \(g.facts.count) texts, \(bad) differ")
        XCTAssertEqual(bad, 0)
        XCTAssertTrue(g.facts.contains { $0.place == nil && $0.taken == nil }, "no case without place and date")
    }

    func testTowervecHeadOnReferenceVectors() throws {
        let g = try golden()
        let url = Self.models.appendingPathComponent("towervec_head.fphead")
        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.count, g.head_bytes, "the app's towervec head is not the one the fixture was made from")
        let head = try ScorerHead(data: data)
        XCTAssertEqual(head.kind, "towervec")
        var dFold = 0.0, dFull = 0.0, dMLX = 0.0, nMLX = 0
        var folds = [Int: FoldedScorerHead](), foldsMLX = [Int: FoldedScorerHead]()    // one fold per question, as per search
        for (k, p) in g.pairs.enumerated() {
            let f = g.facts[p.fact], q = g.questions[p.question]
            guard let v = f.v, let t = q.t else { XCTFail("pair \(k): fixture lacks v or t"); continue }
            XCTAssertEqual(f.fp32.count, 384); XCTAssertEqual(q.fp32.count, 384)
            XCTAssertEqual(v.count, head.dImg); XCTAssertEqual(t.count, head.dImg)
            let folded = folds[p.question] ?? head.fold(q: q.fp32, t: t)
            folds[p.question] = folded
            let pf = Double(folded.p(f: f.fp32, v: v)), pu = Double(head.p(f: f.fp32, q: q.fp32, v: v, t: t))
            dFold = max(dFold, abs(pf - p.p)); dFull = max(dFull, abs(pu - p.p))
            XCTAssertEqual(pf, p.p, accuracy: 1e-4, "pair \(k) [\(f.name)] x [\(q.question)]")
            if let fm = f.mlx_fp16, let qm = q.mlx_fp16, let pm = p.p_mlx {
                let fm16 = foldsMLX[p.question] ?? head.fold(q: qm, t: t)
                foldsMLX[p.question] = fm16
                let x = Double(fm16.p(f: fm, v: v))
                dMLX = max(dMLX, abs(x - pm)); nMLX += 1
                XCTAssertEqual(x, pm, accuracy: 1e-4, "pair \(k) on the MLX vectors")
            }
        }
        let ps = g.pairs.map { $0.p }
        print("FACTS_GOLDEN head: \(g.pairs.count) pairs (P \(String(format: "%.3f", ps.min() ?? 0))-\(String(format: "%.3f", ps.max() ?? 0))); " +
              "max |P swift - P torch fp32|: folded \(dFold), unfolded \(dFull); on MLX vectors \(dMLX) (\(nMLX) pairs)")
        XCTAssertLessThanOrEqual(dFull, 1e-4)
    }

    func testSecondReferenceWithinDeviceBar() throws {
        let g = try golden()
        let st = Self.models.appendingPathComponent("towervec_encoder_fp16/model.safetensors")
        let size = try st.resourceValues(forKeys: [.fileSizeKey]).fileSize
        XCTAssertEqual(size, g.encoder_safetensors_bytes, "the app's encoder is not the one the fixture was made from")
        var lo = 1.0, n = 0
        for (name, a, b) in g.facts.map({ ($0.name, $0.fp32, $0.mlx_fp16) }) + g.questions.map({ ($0.name, $0.fp32, $0.mlx_fp16) }) {
            XCTAssertEqual(a.count, 384, name)
            guard let b = b else { continue }
            let c = Self.cosine(a, b)
            lo = min(lo, c); n += 1
            XCTAssertGreaterThanOrEqual(c, 0.9999, "MLX fp16 vs fp32 [\(name)]")
        }
        print("FACTS_GOLDEN second reference: \(n) texts with MLX fp16 vectors, min cosine vs fp32 \(lo)")
    }
}
