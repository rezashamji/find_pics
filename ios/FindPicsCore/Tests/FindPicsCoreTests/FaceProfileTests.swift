import Foundation
import XCTest
@testable import FindPicsCore

final class FaceProfileTests: XCTestCase {
    /// The Swift profile table equals src/findpics/face_profiles.py (fixture written by eval/face_fixtures.py golden).
    func testTableMatchesPython() throws {
        let url = Bundle.module.url(forResource: "face_profiles", withExtension: "json", subdirectory: "Fixtures")!
        let py = try JSONDecoder().decode([String: FaceProfile].self, from: Data(contentsOf: url))
        XCTAssertEqual(Set(py.keys), Set(FaceProfile.all.map(\.id)))
        for p in FaceProfile.all { XCTAssertEqual(py[p.id], p, p.id) }
        XCTAssertEqual(FaceProfile.shipped.id, "auraface_flip")          // Reza 10-07: the app ships AuraFace + flip
        XCTAssertEqual(FaceProfile.named("buffalo_l"), .buffaloL)
        XCTAssertNil(FaceProfile.named("nope"))
    }

    /// Old entries (no recorded model) are buffalo_l; an entry without faces never mixes anything.
    func testWhichVectorsMayBeCompared() {
        XCTAssertFalse(faceVectorsCurrent(model: nil, hasFaces: true, current: "auraface_flip"))
        XCTAssertTrue(faceVectorsCurrent(model: nil, hasFaces: true, current: "buffalo_l"))
        XCTAssertTrue(faceVectorsCurrent(model: nil, hasFaces: false, current: "auraface_flip"))
        XCTAssertTrue(faceVectorsCurrent(model: "auraface_flip", hasFaces: true, current: "auraface_flip"))
        XCTAssertFalse(faceVectorsCurrent(model: "buffalo_l", hasFaces: true, current: "auraface_flip"))
    }
}

final class FaceMigrationTests: XCTestCase {
    func v(_ x: Float, _ y: Float) -> [Float] { [x, y, 0] }

    func testLocateRefsExactThenNear() {
        let faces: [[Float]] = [v(1, 0), v(0, 1), v(0.6, 0.8)]
        let at = locateRefs([v(0, 1), v(0.6, 0.8), v(0.6000001, 0.8), v(0.7, 0.7)], in: faces)
        XCTAssertEqual(at[0], 1); XCTAssertEqual(at[1], 2)
        XCTAssertEqual(at[2], 2)                      // not bit-identical, cosine >= 0.9999
        XCTAssertNil(at[3])                           // a different face
    }

    func testSourceFaceFoundByBoxAtAnySize() {
        let s = FaceSource(id: "p", t: nil, box: [100, 100, 200, 220], imageW: 1000, imageH: 800)
        // the same photo read at half size: same face, half the pixels; plus another face
        let cands: [(box: [Double], imageW: Double, imageH: Double)] = [([300, 50, 350, 110], 500, 400), ([51, 49, 101, 111], 500, 400)]
        XCTAssertEqual(matchSourceFace(s, candidates: cands), 1)
        XCTAssertNil(matchSourceFace(s, candidates: [cands[0]]))
        XCTAssertEqual(boxIoU([0, 0, 10, 10], [0, 0, 10, 10]), 1, accuracy: 1e-9)
        XCTAssertEqual(boxIoU([0, 0, 10, 10], [5, 0, 15, 10]), 1.0 / 3, accuracy: 1e-9)
        XCTAssertEqual(boxIoU([0, 0, 1, 1], [2, 2, 3, 3]), 0)
    }

    func testKeepOnlyWhenMostFacesFoundAgain() {
        XCTAssertTrue(keepAfterRederive(found: 10, total: 10))
        XCTAssertTrue(keepAfterRederive(found: 2, total: 4))
        XCTAssertFalse(keepAfterRederive(found: 1, total: 3))
        XCTAssertFalse(keepAfterRederive(found: 0, total: 0))
    }

    /// people.json / index.json written before 10-07 decode with the new fields absent.
    func testFaceSourceCodable() throws {
        let s = FaceSource(id: "a", t: 2.5, box: [1, 2, 3, 4], imageW: 10, imageH: 20)
        XCTAssertEqual(try JSONDecoder().decode(FaceSource.self, from: JSONEncoder().encode(s)), s)
    }
}
