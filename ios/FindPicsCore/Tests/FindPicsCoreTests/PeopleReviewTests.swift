import XCTest
@testable import FindPicsCore

/// [F8]: review a face group before confirming it; excluded faces never come back.
final class PeopleReviewTests: XCTestCase {
    /// A unit vector near axis `a` (d = 8), nudged by `j` along axis 7 so faces of one person differ slightly.
    func v(_ a: Int, _ j: Float = 0) -> [Float] {
        var x = [Float](repeating: 0, count: 8); x[a] = 1; x[7] += j
        let n = sqrt(x.reduce(0) { $0 + $1 * $1 }); return x.map { $0 / n }
    }
    func src(_ id: String) -> FaceSource { FaceSource(id: id, t: nil, box: [10, 10, 50, 50], imageW: 100, imageH: 100, side: 1280) }

    func testPickFromGroupKeepsOnlyUnexcluded() {
        let vecs = [v(0, 0.1), v(0, 0.2), v(1), v(0, 0.3)]
        let srcs = ["a", "b", "c", "d"].map(src)
        let p = pickFromGroup(vectors: vecs, sources: srcs, excluded: [2])!
        XCTAssertEqual(p.refs, [vecs[0], vecs[1], vecs[3]])
        XCTAssertEqual(p.sources.map(\.id), ["a", "b", "d"])
        XCTAssertEqual(p.negatives, [vecs[2]])
        XCTAssertEqual(p.negativeSources.map(\.id), ["c"])
        XCTAssertNil(pickFromGroup(vectors: vecs, sources: srcs, excluded: [0, 1, 2, 3]), "nothing kept: no pick")
        XCTAssertNil(pickFromGroup(vectors: vecs, sources: Array(srcs.prefix(3)), excluded: []), "unpaired: no pick")
    }

    func testNegativesMergeAndAKeptFaceLeavesThem() {
        let a = v(1), b = v(2)
        let first = pickFromGroup(vectors: [v(0), a, b], sources: ["x", "a", "b"].map(src), excluded: [1, 2])!
        XCTAssertEqual(first.negatives.count, 2)
        // picked again: `a` is kept this time (it is the person after all), `b` stays a negative, no duplicates
        let again = pickFromGroup(vectors: [v(0), a, b], sources: ["x", "a", "b"].map(src), excluded: [2], earlier: first)!
        XCTAssertEqual(again.negatives, [b])
        XCTAssertEqual(again.negativeSources.map(\.id), ["b"])
        XCTAssertEqual(again.refs, [v(0), a])
    }

    func testRemoveFromPickAndPhotos() {
        let p = PersonPick(refs: [v(0), v(1), v(0, 0.2)], sources: ["a", "b", "c"].map(src))
        let q = removeFromPick(p, remove: [1])!
        XCTAssertEqual(q.refs, [v(0), v(0, 0.2)]); XCTAssertEqual(q.negatives, [v(1)])
        XCTAssertNil(removeFromPick(p, remove: [0, 1, 2]))
        XCTAssertNil(removeFromPick(PersonPick(refs: [v(0)], sources: []), remove: []), "old pick without sources")
        let r = pickFromPhotos(refs: [v(3)], sources: [src("p")], earlier: q)!
        XCTAssertEqual(r.refs, [v(3)]); XCTAssertEqual(r.negatives, [v(1)], "earlier 'not me' still holds")
        XCTAssertEqual(alreadyExcluded(vectors: [v(0), v(1), v(2)], negatives: r.negatives), [1])
    }

    /// The defect itself: a group that holds someone else. Before, expandRefs pulled the intruder (and their other
    /// photos) back in. With the intruder excluded, neither it nor faces closer to it are re-added or counted.
    func testExcludedFacesNeverReenterViaExpandRefs() {
        let me = (0..<6).map { v(0, Float($0) * 0.05) }
        // the intruder: close enough to "me" to have joined the cluster (cos ~0.83), and their second photo
        let intruder = unit([0.83, 0.557, 0, 0, 0, 0, 0, 0]), intruder2 = unit([0.80, 0.60, 0, 0, 0, 0, 0, 0])
        let library = me + [intruder, intruder2]
        let accept: Float = 0.5
        let plain = expandRefs(library, refs: [me[0]], accept: accept, rounds: 3)
        XCTAssertTrue(plain.contains(intruder), "the old path: the intruder joins the references")
        let pick = pickFromGroup(vectors: me + [intruder], sources: (0..<7).map { src("p\($0)") }, excluded: [6])!
        let refs = expandRefs(library, refs: pick.refs, accept: accept, rounds: 3, negatives: pick.negatives)
        XCTAssertFalse(refs.contains(intruder))
        XCTAssertFalse(refs.contains(intruder2), "closer to the excluded face than to the person: not added either")
        XCTAssertTrue(me.dropFirst().allSatisfy { m in refs.contains(m) }, "the person's own faces are still added")
        // and the photos: item 6 = intruder, item 7 = intruder2, items 0-5 = me
        let (s, _) = itemPersonScores(faces: library, faceItem: Array(0..<8), nItems: 8, refs: refs, negatives: pick.negatives)
        XCTAssertEqual(s[6], -1); XCTAssertEqual(s[7], -1)
        XCTAssertTrue((0..<6).allSatisfy { s[$0] >= accept })
        let m = matchPerson(faces: library, faceItem: Array(0..<8), faceChecked: Array(repeating: true, count: 8), nItems: 8,
                            inScope: Array(repeating: true, count: 8), refs: pick.refs, profile: .shipped, negatives: pick.negatives)
        XCTAssertFalse(m.members.contains(6)); XCTAssertFalse(m.members.contains(7))
        let m0 = matchPerson(faces: library, faceItem: Array(0..<8), faceChecked: Array(repeating: true, count: 8), nItems: 8,
                             inScope: Array(repeating: true, count: 8), refs: pick.refs, profile: .shipped)
        XCTAssertTrue(m0.members.contains(6), "without negatives the intruder's photo is in the person's album")
    }

    /// The guard stays: unchecked (small-read) faces decide nothing, with or without negatives.
    func testUncheckedFacesStillDecideNothing() {
        let lib = [v(0), v(0, 0.05), v(1)]
        let m = matchPerson(faces: lib, faceItem: [0, 1, 2], faceChecked: [true, false, true], nItems: 3, inScope: [true, true, true],
                            refs: [v(0, 0.02)], profile: .shipped, negatives: [v(1)])
        XCTAssertFalse(m.members.contains(1)); XCTAssertEqual(m.unchecked, [1])
    }

    func testLeastAlikeFirst() {
        let g = [v(0), v(0, 0.05), v(1), v(0, 0.1)]
        XCTAssertEqual(leastAlikeFirst(g).first, 2)
        XCTAssertEqual(leastAlikeFirst([]), [])
    }

    func testFaceCropRect() {
        // centred, padded square
        let r = faceCropRect(box: [40, 40, 60, 60], imageW: 200, imageH: 100, pad: 0.5)
        XCTAssertEqual(r, Rect(30.0 / 200, 30.0 / 100, 70.0 / 200, 70.0 / 100))
        // at the corner: moved inside, not shrunk
        let c = faceCropRect(box: [0, 0, 20, 20], imageW: 200, imageH: 100, pad: 0.5)
        XCTAssertEqual(c, Rect(0, 0, 40.0 / 200, 40.0 / 100))
        // bigger than the image: the whole short side
        let b = faceCropRect(box: [0, 0, 100, 100], imageW: 200, imageH: 100, pad: 0.5)
        XCTAssertEqual(b.y2 - b.y1, 1, accuracy: 1e-9)
        // a 40 px face in a 1280 px read needs a bigger read than a 400 px face; clamped
        let small = faceCropReadSide(box: [0, 0, 40, 40], imageW: 1280, imageH: 960)
        let big = faceCropReadSide(box: [0, 0, 400, 400], imageW: 1280, imageH: 960)
        XCTAssertGreaterThan(small, big); XCTAssertLessThanOrEqual(small, 1600); XCTAssertGreaterThanOrEqual(big, 300)
    }

    func testWording() {
        XCTAssertEqual(pickPhotosPrompt("me"), "Most accurate: pick 1-3 photos of yourself")
        XCTAssertEqual(pickPhotosPrompt("Mom"), "Most accurate: pick 1-3 photos of Mom")
        XCTAssertEqual(confirmGroupLabel("me", kept: 9, total: 9), "Yes, these 9 faces are me")
        XCTAssertEqual(confirmGroupLabel("Mom", kept: 8, total: 9), "Yes, these 8 of 9 faces are Mom")
    }
}
