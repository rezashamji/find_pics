import Foundation
import XCTest
@testable import FindPicsCore

/// RESULTS 68 (3): ScorerHead / FoldedScorerHead = scorers.py Head in PyTorch fp32 (CPU).
/// Always: tiny random heads (Fixtures/heads.json from eval/cascade_port/heads.py export).
/// FP_HEADS_DIR=<data/public/cascade_port/heads> + FP_HEAD_MODELS=<eval/cascade_port/models>: the shipped towervec head and
/// vecB on every (photo, question) two dev libraries' recordings scored; then the cascade replayed on those libraries'
/// recorded requests with Swift scores vs PyTorch-fp32 scores (bar: identical outputs and judged sets) and vs the
/// recorded bf16-GPU scores (reported).
final class FactsScorerTests: XCTestCase {
    struct Tiny: Decodable { let kind: String; let head: String; let f: [[Float]]?; let q: [[Float]]?; let v: [[Float]]
        let t: [[Float]]; let logit: [Float]; let p: [Float] }
    struct TinyFile: Decodable { let cases: [Tiny] }

    func testTinyHeads() throws {
        let url = Bundle.module.url(forResource: "heads", withExtension: "json", subdirectory: "Fixtures")!
        let F = try JSONDecoder().decode(TinyFile.self, from: Data(contentsOf: url))
        for c in F.cases {
            let h = try ScorerHead(data: Data(base64Encoded: c.head)!)
            var maxL: Float = 0, maxP: Float = 0, maxFold: Float = 0
            var groups = [[Float]: [Int]]()
            for i in c.v.indices {
                let lg = h.logit(f: c.f?[i], q: c.q?[i], v: c.v[i], t: c.t[i])
                maxL = max(maxL, abs(lg - c.logit[i])); maxP = max(maxP, abs(ScorerHead.sigmoid(lg) - c.p[i]))
                let fh = h.fold(q: c.q?[i], t: c.t[i])
                maxFold = max(maxFold, abs(fh.p(f: c.f?[i], v: c.v[i]) - c.p[i]))
                groups[c.t[i], default: []].append(i)
            }
            print("HEADS tiny \(c.kind): n \(c.v.count) max |logit diff| \(maxL) max |P diff| \(maxP) folded \(maxFold)")
            let big = c.logit.map { abs($0) }.max() ?? 1
            XCTAssertLessThan(maxL, 1e-5 * max(1, big)); XCTAssertLessThan(maxP, 1e-5); XCTAssertLessThan(maxFold, 1e-5)
            // batch path = per-photo folded path
            let fh = h.fold(q: c.q?[0], t: c.t[0])
            let b = fh.p(batch: c.f, vs: c.v)
            for i in c.v.indices { XCTAssertEqual(b[i], fh.p(f: c.f?[i], v: c.v[i]), accuracy: 1e-6) }
        }
    }

    struct Inputs: Decodable { let user: String; let rows: [Int]; let qs: [String]; let n: Int; let m: Int; let pairs: [[Double]] }

    static func floats(_ d: Data, _ off: Int, _ count: Int) -> [Float] {
        var a = [Float](repeating: 0, count: count)
        a.withUnsafeMutableBytes { dst in d.withUnsafeBytes { src in
            dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[(4 * off)..<(4 * (off + count))])) } }
        return a
    }

    func testRealHeadsAndDecisions() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let dir = env["FP_HEADS_DIR"], let mdir = env["FP_HEAD_MODELS"], let full = env["FP_CASCADE_FULL"] else { return }
        let tvH = try ScorerHead(contentsOf: URL(fileURLWithPath: mdir + "/towervec_head.fphead"))
        let vbH = try ScorerHead(contentsOf: URL(fileURLWithPath: mdir + "/vecB.fphead"))
        let files = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".json") }.sorted()
        for fn in files {
            let I = try JSONDecoder().decode(Inputs.self, from: Data(contentsOf: URL(fileURLWithPath: dir + "/" + fn)))
            let bin = try Data(contentsOf: URL(fileURLWithPath: dir + "/" + fn.replacingOccurrences(of: ".json", with: ".bin")))
            let n = I.n, m = I.m
            let Fm = (0..<n).map { Self.floats(bin, $0 * 384, 384) }
            let Vm = (0..<n).map { Self.floats(bin, n * 384 + $0 * 1024, 1024) }
            let Qm = (0..<m).map { Self.floats(bin, n * 1408 + $0 * 384, 384) }
            let Tm = (0..<m).map { Self.floats(bin, n * 1408 + m * 384 + $0 * 1024, 1024) }
            var byQ = [Int: [(Int, Double, Double)]]()
            for p in I.pairs { byQ[Int(p[1]), default: []].append((Int(p[0]), p[2], p[3])) }
            var swiftTV = [String: [String: Double]](), swiftVB = [String: [String: Double]]()
            var pyTV = [String: [String: Double]](), pyVB = [String: [String: Double]]()
            var dTV: Double = 0, dVB: Double = 0, nPairs = 0, photos = Set<Int>()
            let t0 = Date()
            for (j, ps) in byQ {
                let ft = tvH.fold(q: Qm[j], t: Tm[j]), fv = vbH.fold(q: nil, t: Tm[j])
                let pt = ft.p(batch: ps.map { Fm[$0.0] }, vs: ps.map { Vm[$0.0] })
                let pv = fv.p(batch: nil, vs: ps.map { Vm[$0.0] })
                let q = I.qs[j]
                for (k, (i, a, b)) in ps.enumerated() {
                    let row = String(I.rows[i])
                    swiftTV[q, default: [:]][row] = Double(pt[k]); swiftVB[q, default: [:]][row] = Double(pv[k])
                    pyTV[q, default: [:]][row] = a; pyVB[q, default: [:]][row] = b
                    dTV = max(dTV, abs(Double(pt[k]) - a)); dVB = max(dVB, abs(Double(pv[k]) - b))
                    nPairs += 1; photos.insert(i)
                }
            }
            print("HEADS real \(I.user): \(photos.count) photos, \(m) questions, \(nPairs) pairs; max |P diff| vs torch fp32: " +
                  "towervec \(dTV), vecB \(dVB); \(String(format: "%.1f", Date().timeIntervalSince(t0))) s (Linux, pure-Swift loops)")
            XCTAssertLessThanOrEqual(dTV, 1e-5); XCTAssertLessThanOrEqual(dVB, 1e-5)
            // decisions downstream: the recorded requests of this library, replayed with each score source
            let L = try JSONDecoder().decode(CascadeRouteTests.LibFile.self,
                                             from: Data(contentsOf: URL(fileURLWithPath: full + "/dev/" + I.user + ".json")))
            var sameSP = 0, diffSP = 0, sameSR = 0, diffSR = 0, judgedSP = 0, judgedSR = 0, missS = 0, missR = 0, outs = 0
            var callsSR = 0, cutSP = 0, cutSR = 0, maxSP: Float = 0
            for r in L.requests {
                let rec = try await CascadeRouteTests.run(r, places: L.place, tv: r.tv, vec: r.vec)
                let sw = try await CascadeRouteTests.run(r, places: L.place, tv: swiftTV, vec: swiftVB)
                let py = try await CascadeRouteTests.run(r, places: L.place, tv: pyTV, vec: pyVB)
                for (a, b) in zip(sw.outs, py.outs) { for (x, y) in zip(a, b) { if x.bitPattern == y.bitPattern { sameSP += 1 } else { diffSP += 1; maxSP = max(maxSP, abs(x - y)) }
                    if (Double(x) >= 0.7) != (Double(y) >= 0.7) { cutSP += 1 } } }
                for (a, b) in zip(sw.outs, rec.outs) {
                    var d = 0
                    for (x, y) in zip(a, b) { if x.bitPattern == y.bitPattern { sameSR += 1 } else { diffSR += 1; d += 1 }
                        if (Double(x) >= 0.7) != (Double(y) >= 0.7) { cutSR += 1 } }
                    if d > 0 { callsSR += 1 }
                    outs += a.count
                }
                judgedSP += sw.judged == py.judged ? 0 : 1
                judgedSR += sw.judged == rec.judged ? 0 : 1
                missS += sw.missingJudge + sw.missingScore; missR += rec.missingJudge + rec.missingScore
                XCTAssertEqual(sw.outs.count, py.outs.count)
            }
            print("HEADS decisions \(I.user): \(L.requests.count) (request, config), \(outs) photo answers. Swift vs torch fp32: " +
                  "\(diffSP) answers not bit-equal (max |diff| \(maxSP)), \(cutSP) on the other side of P = .7, judged sets differ in \(judgedSP) requests. Swift fp32 vs recorded bf16-GPU scores: " +
                  "\(diffSR) answers not bit-equal (in \(callsSR) calls), \(cutSR) on the other side of P = .7, judged sets differ in \(judgedSR) requests; judge answers " +
                  "the fp32 runs need that the recording lacks: \(missS) (recorded scores: \(missR))")
            // pre-registered as bit-equal outputs: impossible for scorer-routed answers (they ARE the scores, ~1e-6 apart);
            // reported in RESULTS 68 as failed-as-worded; what is asserted is the decision layer
            XCTAssertEqual(judgedSP, 0); XCTAssertEqual(cutSP, 0)
        }
    }
}
