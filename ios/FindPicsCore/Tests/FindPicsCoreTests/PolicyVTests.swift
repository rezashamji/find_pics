// policyVScores: every photo gets the right head's P, and - the part that can go wrong without ever looking wrong -
// its OWN score. A scatter bug gives every photo a plausible number belonging to another photo, which no screenshot
// and no AUC over the whole set would catch, so it is checked here against the one-photo-at-a-time path.
import Foundation
import XCTest
@testable import FindPicsCore

final class PolicyVTests: XCTestCase {
    /// the shipped heads, read from the app's own resources (same files the phone loads)
    static func heads() throws -> (towervec: ScorerHead, vec: ScorerHead) {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("FindPicsApp.swiftpm/Sources/Models/facts")
        return (try ScorerHead(contentsOf: dir.appendingPathComponent("towervec_head.fphead")),
                try ScorerHead(contentsOf: dir.appendingPathComponent("vecB.fphead")))
    }

    struct LCG { var s: UInt64; mutating func f() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: s >> 33)) / Float(Int32.max) } }

    func testEveryPhotoGetsItsOwnScoreFromItsOwnHead() throws {
        let (tw, vb) = try Self.heads()
        var r = LCG(s: 20261105)
        let photos = 37
        // every third photo has no image vector at all; of the rest, every other one has a facts line
        let photoRow = (0..<photos).filter { $0 % 3 != 0 }
        let facts = Dictionary(uniqueKeysWithValues: photoRow.enumerated().filter { $0.offset % 2 == 0 }
            .map { ($0.element, (0..<tw.dTxt).map { _ in r.f() }) })
        let vs = photoRow.map { _ in (0..<vb.dImg).map { _ in r.f() } }
        let q = (0..<tw.dTxt).map { _ in r.f() }, t = (0..<vb.dImg).map { _ in r.f() }
        let ft = tw.fold(q: q, t: t), fv = vb.fold(q: nil, t: t)

        // several block sizes, including ones that do not divide the row count and one bigger than it
        for block in [1, 4, 7, 16, 1024] {
            let got = policyVScores(rows: vs, photoRow: photoRow, photos: photos, towervec: ft,
                                    factsVector: { facts[$0] }, vec: fv, block: block)
            XCTAssertEqual(got.count, photos)
            for (k, photo) in photoRow.enumerated() {
                let want = facts[photo] == nil ? fv.p(f: nil, v: vs[k]) : ft.p(f: facts[photo]!, v: vs[k])
                XCTAssertEqual(got[photo], want, accuracy: 1e-5, "photo \(photo), block \(block)")
            }
            for photo in 0..<photos where photo % 3 == 0 {
                XCTAssertEqual(got[photo], 0, "photo \(photo) has no vector, so nothing may score it")
            }
        }
    }

    /// no towervec head (or no facts encoder on the device): every photo falls to vecB, none is skipped
    func testWithoutTheTowervecHeadEveryPhotoStillScores() throws {
        let (tw, vb) = try Self.heads()
        var r = LCG(s: 7)
        let photos = 20
        let photoRow = Array(0..<photos).reversed().map { $0 }      // deliberately not in photo order
        let vs = photoRow.map { _ in (0..<vb.dImg).map { _ in r.f() } }
        let t = (0..<vb.dImg).map { _ in r.f() }
        let fv = vb.fold(q: nil, t: t)
        let facts = [Int: [Float]](uniqueKeysWithValues: (0..<photos).map { ($0, (0..<tw.dTxt).map { _ in r.f() }) })
        let got = policyVScores(rows: vs, photoRow: photoRow, photos: photos, towervec: nil,
                                factsVector: { facts[$0] }, vec: fv, block: 6)
        for (k, photo) in photoRow.enumerated() {
            XCTAssertEqual(got[photo], fv.p(f: nil, v: vs[k]), accuracy: 1e-5, "photo \(photo)")
        }
    }

    func testEmptyAndMismatchedInputsScoreNothingRatherThanCrash() throws {
        let (_, vb) = try Self.heads()
        var r = LCG(s: 3)
        let t = (0..<vb.dImg).map { _ in r.f() }
        let fv = vb.fold(q: nil, t: t)
        let vs = [[Float]]()
        XCTAssertEqual(policyVScores(rows: vs, photoRow: [], photos: 5, towervec: nil,
                                     factsVector: { _ in nil }, vec: fv), [Float](repeating: 0, count: 5))
        let one = [(0..<vb.dImg).map { _ in r.f() }]
        XCTAssertEqual(policyVScores(rows: one, photoRow: [0, 1], photos: 5, towervec: nil,
                                     factsVector: { _ in nil }, vec: fv), [Float](repeating: 0, count: 5))
    }
}
