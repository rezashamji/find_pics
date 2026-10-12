import Foundation
import XCTest
@testable import FindPicsCore

struct HeadGoldenQuery: Decodable {
    let name: String; let question: String; let text: [Float]; let looks: [[Float]]; let avoid: [[Float]]
    let cosine: [Double]; let head_logit: [Double]; let blend: [Double]; let top200: [Int]
}
struct HeadGolden: Decodable {
    let head_file: String; let head_bytes: Int; let n: Int; let dim: Int; let vectors_f16_b64: String
    let queries: [HeadGoldenQuery]
}

/// RESULTS 48: the shipped learned head (ios/FindPicsApp.swiftpm/Sources/Models/<head_file>) on 3 audit4 searches x
/// 500 public DISBench photos must give the Python reference's scores (eval/learned_head/head_ship.py fixture).
final class HeadRankerTests: XCTestCase {
    func golden() throws -> HeadGolden {
        let url = Bundle.module.url(forResource: "head_golden", withExtension: "json", subdirectory: "Fixtures")!
        return try JSONDecoder().decode(HeadGolden.self, from: Data(contentsOf: url))
    }

    /// The app's copy of the head (the file the phone ships), found from this source file's location.
    func shippedHeadURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("FindPicsApp.swiftpm/Sources/Models/" + name)
    }

    static func order(_ s: [Float]) -> [Int] {        // best first; exact ties by index (= numpy stable argsort)
        s.indices.sorted { s[$0] != s[$1] ? s[$0] > s[$1] : $0 < $1 }
    }

    func testGoldenParityWithPython() throws {
        let g = try golden()
        let url = shippedHeadURL(g.head_file)
        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.count, g.head_bytes, "the app's head file is not the one the fixture was made from")
        let head = try HeadRanker(data: data)
        XCTAssertEqual(head.dim, g.dim)
        let bits = Data(base64Encoded: g.vectors_f16_b64)!
        XCTAssertEqual(bits.count, g.n * g.dim * 2)
        var vec = [Float](repeating: 0, count: g.n * g.dim)
        bits.withUnsafeBytes { src in vec.withUnsafeMutableBufferPointer { MatrixMath.halfToFloat(src.baseAddress!, $0.baseAddress!, g.n * g.dim) } }
        let units = FlatRows(data: vec, dim: g.dim)
        let unitItem = Array(0..<g.n)
        for q in g.queries {
            let cos = lookScores(units: units, unitItem: unitItem, nItems: g.n, looks: q.looks, avoid: q.avoid)
            let folded = head.fold(question: q.text)
            let hl = folded.itemLogits(units: units, unitItem: unitItem, nItems: g.n, block: 128)
            let bl = blendScores(cosine: cos, headLogits: hl)
            let dc = zip(cos, q.cosine).map { abs(Double($0) - $1) }.max()!
            let dh = zip(hl, q.head_logit).map { abs(Double($0) - $1) }.max()!
            let db = zip(bl, q.blend).map { abs(Double($0) - $1) }.max()!
            print("HEAD PARITY \(q.name): max |d cosine| \(dc), |d head logit| \(dh), |d blend| \(db)")
            XCTAssertLessThan(dc, 1e-5, q.name)
            XCTAssertLessThan(dh, 1e-3, q.name)
            XCTAssertLessThan(db, 1e-3, q.name)
            XCTAssertEqual(Array(Self.order(bl).prefix(200)), q.top200, "\(q.name): top-200 order differs from Python")
        }
    }

    /// A tiny head written in the file format, with the |img - txt| block (B = 4): the folded per-search computation and
    /// the custom-matmul seam give the unfolded network's logit.
    func testFoldedEqualsUnfoldedWithAbsBlockAndCustomGemm() throws {
        let D = 4, B = 4, H1 = 3, H2 = 2
        var rng = SystemRandomNumberGenerator()
        func r() -> Float { Float.random(in: -1...1, using: &rng) }
        let w1 = (0..<(H1 * B * D)).map { _ in Float(Int(r() * 64)) / 64 }       // exact in fp16
        let b1 = (0..<H1).map { _ in r() }, w2 = (0..<(H2 * H1)).map { _ in Float(Int(r() * 64)) / 64 }
        let b2 = (0..<H2).map { _ in r() }, w3 = (0..<H2).map { _ in r() }, b3 = r()
        var file = Data("FPHEAD01".utf8)
        let meta = Data(#"{"test":true}"#.utf8)
        for v in [UInt32(1), UInt32(D), UInt32(B), UInt32(H1), UInt32(H2), UInt32(meta.count)] { withUnsafeBytes(of: v.littleEndian) { file.append(contentsOf: $0) } }
        file.append(meta)
        file.append(Data(count: (16 - file.count % 16) % 16))
        for v in w1 { withUnsafeBytes(of: MatrixMath.floatToHalfBits(v).littleEndian) { file.append(contentsOf: $0) } }
        for v in b1 { withUnsafeBytes(of: v.bitPattern.littleEndian) { file.append(contentsOf: $0) } }
        for v in w2 { withUnsafeBytes(of: MatrixMath.floatToHalfBits(v).littleEndian) { file.append(contentsOf: $0) } }
        for v in b2 + w3 + [b3] { withUnsafeBytes(of: v.bitPattern.littleEndian) { file.append(contentsOf: $0) } }
        let head = try HeadRanker(data: file)
        XCTAssertEqual(head.meta, #"{"test":true}"#)
        XCTAssertThrowsError(try HeadRanker(data: file.dropLast()))
        let t: [Float] = [0.5, -0.5, 0.5, 0.5]
        let photos: [[Float]] = (0..<7).map { _ in (0..<D).map { _ in r() } }
        func unfolded(_ x: [Float]) -> Float {
            var f = [Float]()
            f += x; f += t
            for j in 0..<D { f.append(x[j] * t[j]) }
            for j in 0..<D { f.append(abs(x[j] - t[j])) }
            var h = [Float](repeating: 0, count: H1), g = [Float](repeating: 0, count: H2)
            for j in 0..<H1 {
                var s: Float = b1[j]
                for k in 0..<(B * D) { s += w1[j * B * D + k] * f[k] }
                h[j] = max(0, s)
            }
            for k in 0..<H2 {
                var s: Float = b2[k]
                for j in 0..<H1 { s += w2[k * H1 + j] * h[j] }
                g[k] = max(0, s)
            }
            var out: Float = b3
            for k in 0..<H2 { out += w3[k] * g[k] }
            return out
        }
        final class Count: @unchecked Sendable { var n = 0 }
        let calls = Count()
        let counting: GemmNT = { a, m, b, n, d, c in calls.n += 1; MatrixMath.gemmNT(a, m: m, b, n: n, d: d, c) }
        let folded = head.fold(question: t)
        // items: 0 = photos 0-2 (a video: best frame), 1 = photo 3, 2 = no vector, 3 = photos 4-6
        let unitItem = [0, 0, 0, 1, 3, 3, 3]
        let got = folded.itemLogits(units: photos, unitItem: unitItem, nItems: 4, block: 3, gemm: counting)
        let want = [photos[0...2].map(unfolded).max()!, unfolded(photos[3]), -.infinity, photos[4...6].map(unfolded).max()!]
        for k in [0, 1, 3] { XCTAssertEqual(got[k], want[k], accuracy: 1e-5) }
        XCTAssertEqual(got[2], -.infinity)
        XCTAssertEqual(calls.n, 3 * 3, "3 blocks x (folded layer 1 + |img-txt| block + layer 2)")
    }

    func testBlendZScores() {
        let z = zScores([1, 2, 3, -.infinity])
        XCTAssertEqual(z[0], -1.2247449, accuracy: 1e-6); XCTAssertEqual(z[1], 0, accuracy: 1e-6)
        XCTAssertEqual(z[3], -.infinity)
        XCTAssertEqual(zScores([5, 5]), [0, 0])                                        // std 0 -> 1
        let b = blendScores(cosine: [0.1, 0.3, 0.2, -.infinity], headLogits: [40, -3, 0, 2])
        let zc = zScores([0.1, 0.3, 0.2]), zh = zScores([headLogitClip, -3, 0])    // logit clipped; no-vector photo out
        for k in 0..<3 { XCTAssertEqual(b[k], zc[k] + zh[k], accuracy: 1e-6) }
        XCTAssertEqual(b[3], -.infinity)
        XCTAssertEqual(HeadRankerTests.order(b).last, 3)
    }
}
