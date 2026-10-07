import Foundation
import XCTest
@testable import FindPicsCore

final class FaceUpgradeTests: XCTestCase {
    func testEffectiveFaceSide() {
        // recorded: used as is
        XCTAssertEqual(effectiveFaceSide(stored: 448, isVideo: false, imageVersion: 2, lowRes: nil), 448)
        XCTAssertEqual(effectiveFaceSide(stored: 1280, isVideo: false, imageVersion: 2, lowRes: nil), 1280)
        // not recorded: before 8ded039 (imageVersion nil = 1) the read was 1280 -> checked
        XCTAssertTrue(facesFromFullRead(stored: nil, isVideo: false, imageVersion: nil, lowRes: nil))
        XCTAssertTrue(facesFromFullRead(stored: nil, isVideo: false, imageVersion: 1, lowRes: false))
        // ... unless it was a smaller stand-in
        XCTAssertFalse(facesFromFullRead(stored: nil, isVideo: false, imageVersion: nil, lowRes: true))
        // imageVersion 2 was only written by builds reading at 448
        XCTAssertFalse(facesFromFullRead(stored: nil, isVideo: false, imageVersion: 2, lowRes: nil))
        XCTAssertEqual(effectiveFaceSide(stored: nil, isVideo: false, imageVersion: 2, lowRes: nil), indexReadSide)
        // video frames are read at 1280 whatever the photo read size
        XCTAssertTrue(facesFromFullRead(stored: nil, isVideo: true, imageVersion: 2, lowRes: nil))
        XCTAssertFalse(facesFromFullRead(stored: 448, isVideo: false, imageVersion: nil, lowRes: nil))
        XCTAssertTrue(facesFromFullRead(stored: faceReadSide, isVideo: false, imageVersion: 2, lowRes: nil))
    }

    func testRecordedFaceSide() {
        XCTAssertEqual(recordedFaceSide(full: true, requested: 448, gotLongSide: 486), 448)
        XCTAssertEqual(recordedFaceSide(full: true, requested: 1280, gotLongSide: 800), 1280)   // whole small original
        XCTAssertEqual(recordedFaceSide(full: false, requested: 1280, gotLongSide: 480), 480)  // stand-in
    }

    func testUpgradeWorkNewestFirst() {
        let items: [(id: String, taken: Double?, needsUpgrade: Bool)] = [
            ("old", 100, true), ("new", 300, true), ("done", 400, false), ("undated", nil, true), ("mid", 200, true),
            ("tieB", 250, true), ("tieA", 250, true),
        ]
        XCTAssertEqual(faceUpgradeWork(items), ["new", "tieA", "tieB", "mid", "old", "undated"])
        XCTAssertEqual(faceUpgradeWork(items, skip: ["new", "mid"]), ["tieA", "tieB", "old", "undated"])
        XCTAssertEqual(faceUpgradeWork([]), [])
    }

    func testReadPolicy() {
        XCTAssertEqual(readPolicy(.indexForeground, side: indexReadSide), ReadPolicy(fastResize: true, localCopyIsFinal: true))
        XCTAssertEqual(readPolicy(.indexBackground, side: indexReadSide), ReadPolicy(fastResize: true, localCopyIsFinal: true))
        // the face upgrade: must not settle for the local ~480 px copy
        XCTAssertEqual(readPolicy(.indexBackground, side: faceReadSide), ReadPolicy(fastResize: false, localCopyIsFinal: false))
        XCTAssertEqual(readPolicy(.indexForeground, side: faceReadSide), ReadPolicy(fastResize: false, localCopyIsFinal: false))
        XCTAssertEqual(readPolicy(.judge, side: 448), ReadPolicy(fastResize: false, localCopyIsFinal: false))
        XCTAssertEqual(readPolicy(.localOnly, side: 448), ReadPolicy(fastResize: false, localCopyIsFinal: false))
    }

    func testNotes() {
        XCTAssertNil(uncheckedFacesNote(0))
        XCTAssertEqual(uncheckedFacesNote(1), "1 photo not checked for faces yet (improving overnight).")
        XCTAssertEqual(uncheckedFacesNote(42), "42 photos not checked for faces yet (improving overnight).")
        XCTAssertEqual(faceUpgradeLine(checked: 3, total: 10), "Improving faces: 3 of 10")
    }

    private func unit(_ v: [Float]) -> [Float] { let n = sqrt(v.map { $0 * $0 }.reduce(0, +)); return v.map { $0 / n } }

    /// Small-read faces decide nothing: a matching one is "unchecked" (counted), not a member; a stranger's is
    /// unchecked too (not silently dropped); out-of-scope items are neither.
    func testUncheckedFacesAreNotMembers() {
        let me = unit([1, 0.1, 0, 0]), stranger: [Float] = [0, 0, 1, 0]
        let faces = [me, me, stranger, stranger, me, me]
        let checked = [true, false, true, false, true, false]
        let inScope = [true, true, true, true, false, false]
        let m = matchPerson(faces: faces, faceItem: [0, 1, 2, 3, 4, 5], faceChecked: checked, nItems: 6, inScope: inScope,
                            refs: [[1, 0, 0, 0]], expandRounds: 0)
        XCTAssertEqual(m.members, [0])
        XCTAssertEqual(m.unchecked, [1, 3])
        XCTAssertEqual(m.best, [0, -1, 2, -1, 4, -1])
        XCTAssertEqual(m.scores[1], -1)
    }

    /// Query-time expansion must not walk through an unchecked face: u bridges the person to c.
    func testExpansionIgnoresUncheckedFaces() {
        let ref: [Float] = [1, 0, 0, 0]
        let near = unit([1, 0.1, 0, 0]), u = unit([1, 1, 0, 0]), c: [Float] = [0, 1, 0, 0]
        let faces = [near, u, c]
        let all = matchPerson(faces: faces, faceItem: [0, 1, 2], faceChecked: [true, true, true], nItems: 3,
                              inScope: [true, true, true], refs: [ref])
        XCTAssertEqual(all.members, [0, 1, 2])       // u checked: the chain reaches c (cos 0.707 to u)
        let safe = matchPerson(faces: faces, faceItem: [0, 1, 2], faceChecked: [true, false, true], nItems: 3,
                               inScope: [true, true, true], refs: [ref])
        XCTAssertEqual(safe.members, [0])
        XCTAssertEqual(safe.unchecked, [1])
    }

    /// With every face checked, matchPerson is exactly the people.py chain (fixture of FaceMatchTests).
    func testAllCheckedEqualsChain() throws {
        let url = Bundle.module.url(forResource: "faces", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(FaceFixture.self, from: Data(contentsOf: url))
        let pr = FaceProfile.shipped
        let g = faceGroups(faces: f.faces, faceItem: f.faceItem, facePx: f.facePx, det: f.det, accept: pr.group)
        let groups = g.map { $0.faces.map { f.faces[$0] } }
        let exp = expandRefs(f.faces, refs: f.refs, accept: pr.expand, rounds: 3)
        let other = otherIdentities(groups: groups, refs: exp, accept: pr.other)
        let (s, b) = itemPersonScores(faces: f.faces, faceItem: f.faceItem, nItems: f.nItems, refs: exp, others: other)
        let m = matchPerson(faces: f.faces, faceItem: f.faceItem, faceChecked: f.faces.map { _ in true }, nItems: f.nItems,
                            inScope: (0..<f.nItems).map { _ in true }, refs: f.refs, others: groups, profile: pr)
        XCTAssertEqual(m.scores, s); XCTAssertEqual(m.best, b)
        XCTAssertEqual(m.members, (0..<f.nItems).filter { s[$0] >= pr.accept })
        XCTAssertEqual(m.unchecked, [])
        XCTAssertFalse(m.members.isEmpty)
    }
}
