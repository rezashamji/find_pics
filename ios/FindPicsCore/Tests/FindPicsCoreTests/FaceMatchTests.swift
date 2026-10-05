import Foundation
import XCTest
@testable import FindPicsCore

struct GroupJ: Decodable { let faces: [Int]; let items: [Int]; let rep: Int }
struct FaceFixture: Decodable {
    let faces: [[Float]]; let faceItem: [Int]; let nItems: Int; let facePx: [Float]; let det: [Float]
    let groups: [GroupJ]; let refs: [[Float]]; let expanded_n: Int; let others: [[Float]]
    let scores: [Float]; let best: [Int]; let scores_plain: [Float]
}

final class FaceMatchTests: XCTestCase {
    /// Same answers as people.py on 600 synthetic faces of 40 made-up identities.
    func testMatchesPython() throws {
        let url = Bundle.module.url(forResource: "faces", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(FaceFixture.self, from: Data(contentsOf: url))
        let g = faceGroups(faces: f.faces, faceItem: f.faceItem, facePx: f.facePx, det: f.det)
        XCTAssertEqual(g.count, f.groups.count)
        for (a, b) in zip(g, f.groups) { XCTAssertEqual(a.faces, b.faces); XCTAssertEqual(a.items, b.items); XCTAssertEqual(a.rep, b.rep) }
        XCTAssertEqual(expandRefs(f.faces, refs: f.refs, accept: 0.55, rounds: 3).count, f.expanded_n)
        let (s, b) = itemPersonScores(faces: f.faces, faceItem: f.faceItem, nItems: f.nItems, refs: f.refs, others: f.others)
        for i in 0..<f.nItems { XCTAssertEqual(s[i], f.scores[i], accuracy: 1e-4, "item \(i)") }
        XCTAssertEqual(b, f.best)
        let (s0, _) = itemPersonScores(faces: f.faces, faceItem: f.faceItem, nItems: f.nItems, refs: f.refs)
        for i in 0..<f.nItems { XCTAssertEqual(s0[i], f.scores_plain[i], accuracy: 1e-4) }
    }
}
