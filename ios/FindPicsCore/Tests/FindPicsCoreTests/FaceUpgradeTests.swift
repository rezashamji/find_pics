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

    // MARK: saved people after the upgrade

    private func cand(_ box: [Double], _ w: Double, _ h: Double, _ e: [Float]) -> FaceCandidate {
        FaceCandidate(box: box, imageW: w, imageH: h, embedding: e)
    }

    /// A reference from a 448 read whose photo was re-read at 1280: the face with the same box (scaled) replaces it.
    func testRefsRederivedFromUpgradedFaces() {
        let old: [Float] = [1, 0, 0], new: [Float] = [0.9, 0.1, 0], other: [Float] = [0, 0, 1]
        let src = FaceSource(id: "p", t: nil, box: [100, 50, 140, 100], imageW: 448, imageH: 336, side: 448)
        // same face at 1280 x 960 (x 2.857), plus a stranger elsewhere
        let now = SourcePhotoFaces(side: 1280, faces: [cand([900, 600, 1000, 700], 1280, 960, other),
                                                      cand([286, 143, 400, 286], 1280, 960, new)])
        let r = refsAfterUpgrade(refs: [old], sources: [src], now: [now])!
        XCTAssertEqual(r.refs, [new]); XCTAssertEqual(r.rederived, 1); XCTAssertEqual(r.lost, 0); XCTAssertTrue(r.keep)
        XCTAssertEqual(r.sources[0].box, [286, 143, 400, 286]); XCTAssertEqual(r.sources[0].imageW, 1280)
        XCTAssertEqual(r.sources[0].side, 1280)
    }

    func testRefsUntouchedCases() {
        let e: [Float] = [1, 0, 0]
        let s448 = FaceSource(id: "a", t: nil, box: [0, 0, 10, 10], imageW: 448, imageH: 448, side: 448)
        let small = SourcePhotoFaces(side: 448, faces: [cand([0, 0, 10, 10], 448, 448, [0, 1, 0])])
        XCTAssertNil(refsAfterUpgrade(refs: [e], sources: [s448], now: [small]))          // photo not re-read yet
        XCTAssertNil(refsAfterUpgrade(refs: [e], sources: [s448], now: [nil]))            // not in the index
        let picked = FaceSource(id: "a", t: nil, box: [0, 0, 10, 10], imageW: 1280, imageH: 1280, side: 1280)
        let full = SourcePhotoFaces(side: 1280, faces: [cand([0, 0, 30, 30], 1280, 1280, [0, 1, 0])])
        XCTAssertNil(refsAfterUpgrade(refs: [e], sources: [picked], now: [full]))         // already from a full-size read
        let video = FaceSource(id: "v", t: 2.0, box: [0, 0, 10, 10], imageW: 448, imageH: 448)
        XCTAssertNil(refsAfterUpgrade(refs: [e], sources: [video], now: [full]))          // video frames: not upgraded
        XCTAssertNil(refsAfterUpgrade(refs: [e, e], sources: [s448], now: [full]))        // refs / sources do not pair
        // unknown side, vector still the photo's face exactly: kept, side recorded
        let legacy = FaceSource(id: "a", t: nil, box: [0, 0, 30, 30], imageW: 1280, imageH: 1280)
        let same = SourcePhotoFaces(side: 1280, faces: [cand([0, 0, 30, 30], 1280, 1280, e)])
        let r = refsAfterUpgrade(refs: [e], sources: [legacy], now: [same])!
        XCTAssertEqual(r.refs, [e]); XCTAssertEqual(r.rederived, 0); XCTAssertEqual(r.sources[0].side, 1280); XCTAssertTrue(r.keep)
    }

    /// At least half of ALL references must survive (not-yet-re-read ones count as found); else ask again.
    func testRefsReaskRule() {
        let e: [Float] = [1, 0, 0]
        func s(_ id: String) -> FaceSource { FaceSource(id: id, t: nil, box: [0, 0, 10, 10], imageW: 448, imageH: 448, side: 448) }
        let gone = SourcePhotoFaces(side: 1280, faces: [cand([1000, 1000, 1100, 1100], 1280, 1280, [0, 1, 0])])  // no box overlap
        let notYet = SourcePhotoFaces(side: 448, faces: [])
        // 2 of 4 lost, 2 not re-read yet: 2 >= 4/2 -> kept, the 2 lost ones dropped
        let a = refsAfterUpgrade(refs: [e, e, e, e], sources: [s("1"), s("2"), s("3"), s("4")], now: [gone, gone, notYet, notYet])!
        XCTAssertTrue(a.keep); XCTAssertEqual(a.refs.count, 2); XCTAssertEqual(a.lost, 2); XCTAssertEqual(a.sources.map(\.id), ["3", "4"])
        // 3 of 4 lost -> ask again
        let b = refsAfterUpgrade(refs: [e, e, e, e], sources: [s("1"), s("2"), s("3"), s("4")], now: [gone, gone, gone, notYet])!
        XCTAssertFalse(b.keep); XCTAssertEqual(b.found, 1); XCTAssertEqual(b.total, 4)
    }

    func testFaceSourceDecodesWithoutSide() throws {
        let old = #"{"id":"p","box":[1,2,3,4],"imageW":10,"imageH":20}"#
        let s = try JSONDecoder().decode(FaceSource.self, from: Data(old.utf8))
        XCTAssertNil(s.side); XCTAssertNil(s.t)
    }

    /// Groups from checked faces when they form any; all faces otherwise. Indices are into the full list.
    func testGroupsPreferChecked() {
        let a: [Float] = [1, 0, 0], b: [Float] = [0, 1, 0]
        // faces 0-2: person A, small read; faces 3-5: person B, checked
        let faces = [a, a, a, b, b, b], item = [0, 1, 2, 3, 4, 5]
        let px = [Float](repeating: 80, count: 6), det = [Float](repeating: 0.9, count: 6)
        let g = faceGroupsPreferChecked(faces: faces, faceItem: item, facePx: px, det: det,
                                        checked: [false, false, false, true, true, true])
        XCTAssertEqual(g.count, 1); XCTAssertEqual(Set(g[0].faces), [3, 4, 5]); XCTAssertEqual(g[0].items, [3, 4, 5])
        XCTAssertTrue([3, 4, 5].contains(g[0].rep))
        // nothing checked yet: all faces (both people)
        let all = faceGroupsPreferChecked(faces: faces, faceItem: item, facePx: px, det: det, checked: [Bool](repeating: false, count: 6))
        XCTAssertEqual(all, faceGroups(faces: faces, faceItem: item, facePx: px, det: det))
        XCTAssertEqual(all.count, 2)
        // one checked face forms no group (a group needs a face with 2+ neighbours incl. itself): fall back to all
        let few = faceGroupsPreferChecked(faces: faces, faceItem: item, facePx: px, det: det,
                                          checked: [false, false, false, true, false, false])
        XCTAssertEqual(few.count, 2)
    }
}
