import XCTest
@testable import FindPicsCore

final class LibraryTests: XCTestCase {
    func testDownloadPolicy() {
        let wifi = NetworkPath(connected: true, expensive: false, constrained: false)
        let cell = NetworkPath(connected: true, expensive: true, constrained: false)
        let lowData = NetworkPath(connected: true, expensive: false, constrained: true)
        let off = NetworkPath(connected: false, expensive: false, constrained: false)
        XCTAssertTrue(iCloudDownloadAllowed(.indexBackground, wifi)); XCTAssertTrue(iCloudDownloadAllowed(.indexForeground, wifi))
        XCTAssertFalse(iCloudDownloadAllowed(.indexBackground, cell)); XCTAssertFalse(iCloudDownloadAllowed(.indexForeground, cell))
        XCTAssertTrue(iCloudDownloadAllowed(.judge, cell))          // a search the person waits on
        for p in [lowData, off] { for u in [FetchPurpose.judge, .indexForeground, .indexBackground] { XCTAssertFalse(iCloudDownloadAllowed(u, p)) } }
        XCTAssertFalse(iCloudDownloadAllowed(.localOnly, wifi))
    }

    func testFullResolution() {
        XCTAssertTrue(isFullResolution(gotW: 1280, gotH: 960, requestedSide: 1280, originalW: 4032, originalH: 3024))
        XCTAssertFalse(isFullResolution(gotW: 512, gotH: 384, requestedSide: 1280, originalW: 4032, originalH: 3024))   // stand-in
        XCTAssertTrue(isFullResolution(gotW: 640, gotH: 480, requestedSide: 1280, originalW: 640, originalH: 480))     // small original
        XCTAssertTrue(isFullResolution(gotW: 100, gotH: 100, requestedSide: 1280, originalW: 0, originalH: 0))         // unknown
    }

    func testIndexWork() {
        let lib = ["new", "low", "wait", "fail", "bad", "done"]
        let idx = ["low": true, "done": false]
        let nr: [String: ReadOutcome] = ["wait": .waitingForICloud, "fail": .downloadFailed, "bad": .unreadable]
        let a = indexWork(library: lib, indexedLowRes: idx, notRead: nr, downloads: false, retryFailed: false)
        XCTAssertEqual(a.local, ["new", "bad"]); XCTAssertEqual(a.download, [])
        let b = indexWork(library: lib, indexedLowRes: idx, notRead: nr, downloads: true, retryFailed: false)
        XCTAssertEqual(b.download, ["low", "wait"])
        let c = indexWork(library: lib, indexedLowRes: idx, notRead: nr, downloads: true, retryFailed: true)
        XCTAssertEqual(c.download, ["low", "wait", "fail"])
        let lowFailed = indexWork(library: ["low"], indexedLowRes: ["low": true], notRead: ["low": .downloadFailed], downloads: true, retryFailed: false)
        XCTAssertEqual(lowFailed.download, [])
        XCTAssertEqual(removedFromLibrary(indexed: ["a", "b", "c"], library: ["b"]), ["a", "c"])
    }

    func testSummary() {
        XCTAssertNil(notReadSummary([:], lowRes: 0))
        let s = notReadSummary(["a": .waitingForICloud, "b": .waitingForICloud, "c": .unreadable], lowRes: 3)!
        XCTAssertTrue(s.contains("2 stored only in iCloud")); XCTAssertTrue(s.contains("1 could not be read")); XCTAssertTrue(s.contains("3 are indexed"))
    }
}

final class PeoplePickTests: XCTestCase {
    func v(_ x: Float, _ y: Float) -> [Float] { [x, y] }

    func testOwnerIsTheSelfieGroupElseLargest() {
        // group 0 largest (no selfies), group 1 in 4 front-camera photos
        let items = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11]]
        var cam = [String](repeating: "back", count: 12); for i in [6, 7, 8, 9] { cam[i] = "front" }
        XCTAssertEqual(suggestGroup(groupItems: items, itemCamera: cam, forOwner: true, assigned: []), 1)
        XCTAssertEqual(suggestGroup(groupItems: items, itemCamera: cam, forOwner: false, assigned: []), 0)
        XCTAssertEqual(suggestGroup(groupItems: items, itemCamera: cam, forOwner: false, assigned: [0]), 1)   // Mom: not the named one
        XCTAssertEqual(suggestGroup(groupItems: items, itemCamera: cam, forOwner: true, assigned: [1]), 0)
        let noTags = [String](repeating: "", count: 12)
        XCTAssertEqual(suggestGroup(groupItems: items, itemCamera: noTags, forOwner: true, assigned: []), 0)  // no EXIF: largest
        var two = cam; two[8] = "back"; two[9] = "back"                                                      // 2 selfies < 3
        XCTAssertEqual(suggestGroup(groupItems: items, itemCamera: two, forOwner: true, assigned: []), 0)
        XCTAssertNil(suggestGroup(groupItems: items, itemCamera: cam, forOwner: false, assigned: [0, 1, 2]))
        XCTAssertEqual(otherGroupsOrder(count: 4, suggested: 1, assigned: [0]), [2, 3, 0])
    }

    func testAssignedGroupsAndQuestions() {
        let groups = [[v(1, 0), v(0.9, 0.1)], [v(0, 1), v(0.1, 0.9)]]
        XCTAssertEqual(assignedGroups(groups: groups, named: [[v(0, 1)]]), [1])
        XCTAssertEqual(whoQuestion("me"), "Is this you?"); XCTAssertEqual(whoQuestion("Mom"), "Is this Mom?")
    }

    func testPickRefFacesSharedPerson() {
        // photo 1: a big stranger + the person; photo 2: only the person -> the shared face, not the biggest
        let p1: [(px: Double, emb: [Float])] = [(200, v(0, 1)), (80, v(1, 0))]
        let p2: [(px: Double, emb: [Float])] = [(90, v(0.98, 0.05))]
        let r = pickRefFaces(photos: [p1, p2, []])
        XCTAssertEqual(r.count, 2); XCTAssertEqual(r[0], v(1, 0)); XCTAssertEqual(r[1], v(0.98, 0.05))
        XCTAssertEqual(pickRefFaces(photos: [p1]), [v(0, 1)])          // one photo: the largest face
    }
}
