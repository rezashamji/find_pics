import Foundation
import XCTest
@testable import FindPicsCore

/// RESULTS 62: person searches' first answer. Golden answers from findpics.person_first on public data (IMDB-WIKI eras,
/// DISBench 'tired' person sets) and synthetic sets (eval/person_first/fixtures.py regenerates the file).
private struct PFFixture: Decodable {
    struct Walk: Decodable { let ids: [String]; let eventOf: [String: String]?; let seed: Int; let cap: Int?; let bySize: Bool; let order: [Int] }
    struct Single: Decodable {
        let name: String; let ids: [String]; let eventOf: [String: String]?; let p: [String: Double]
        let propagate: Bool; let expected: [String: Bool]
    }
    struct Pair: Decodable {
        let name: String; let ids: [String]; let eventOf: [String: String]?; let pA: [String: Double]; let pB: [String: Double]
        let propagate: Bool; let expected: [String: Int?]
    }
    struct Fact: Decodable { let name: String; let cos: [Float]; let head: [Float]?; let order: [Int] }
    struct Stream: Decodable {
        let name: String; let ids: [String]; let eventOf: [String: String]?; let pA: [String: Double]; let pB: [String: Double]
        let questions: Int; let seed: Int; let bySize: Bool; let updates: [[UpdateCell]]; let final: [String: Int?]
    }
    enum UpdateCell: Decodable, Equatable {
        case int(Int), bool(Bool)
        init(from d: Decoder) throws {
            let c = try d.singleValueContainer()
            if let b = try? c.decode(Bool.self) { self = .bool(b) } else { self = .int(try c.decode(Int.self)) }
        }
    }
    let walk: [Walk], single: [Single], pair: [Pair], fact: [Fact], stream: [Stream]
}

final class PersonFirstTests: XCTestCase {
    private func fixture() throws -> PFFixture {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "person_first", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(PFFixture.self, from: Data(contentsOf: url))
    }

    func testFNVKnownVectors() {
        XCTAssertEqual(fnv1a64(""), 0xcbf29ce484222325)
        XCTAssertEqual(fnv1a64("a"), 0xaf63dc4c8601ec8c)
        XCTAssertEqual(fnv1a64("foobar"), 0x85944171f73967e8)
        XCTAssertEqual(fmix64(0), 0)
    }

    func testWalkOrderMatchesPython() throws {
        let f = try fixture()
        XCTAssertGreaterThanOrEqual(f.walk.count, 5)
        XCTAssertTrue(f.walk.contains { $0.bySize })
        for c in f.walk { XCTAssertEqual(lookWalkOrder(ids: c.ids, eventOf: c.eventOf, seed: c.seed, cap: c.cap, bySize: c.bySize), c.order) }
    }

    func testSingleLookMatchesPython() throws {
        let f = try fixture()
        var mismatches = 0, photos = 0
        for c in f.single {
            let got = lookSinglePartial(ids: c.ids, eventOf: c.eventOf, pYes: c.p, propagate: c.propagate)
            XCTAssertEqual(got.count, c.expected.count, c.name)
            for (id, want) in c.expected { photos += 1; if got[id] != want { mismatches += 1 } }
        }
        print("PERSONFIRST single mismatches \(mismatches) of \(photos)")
        XCTAssertEqual(mismatches, 0)
    }

    func testPairLookMatchesPython() throws {
        let f = try fixture()
        var mismatches = 0, photos = 0
        for c in f.pair {
            let got = lookPairAlbums(ids: c.ids, eventOf: c.eventOf, pA: c.pA, pB: c.pB, propagate: c.propagate)
            XCTAssertEqual(got.count, c.expected.count, c.name)
            for (id, want) in c.expected {
                photos += 1
                guard let g = got[id] else { mismatches += 1; continue }
                if g != want { mismatches += 1 }
            }
        }
        print("PERSONFIRST pair mismatches \(mismatches) of \(photos)")
        XCTAssertEqual(mismatches, 0)
    }

    /// The blend is Float32 here and float64 in Python: same keys within 1e-4, and the order agrees except among
    /// photos whose keys are that close.
    func testFactOrderMatchesPython() throws {
        let f = try fixture()
        for c in f.fact {
            let got = personFactOrder(cosine: c.cos, headLogits: c.head)
            XCTAssertEqual(got.count, c.order.count, c.name)
            XCTAssertEqual(Set(got), Set(0..<c.cos.count), c.name)
            let key: [Float] = c.head.map { blendScores(cosine: c.cos, headLogits: $0) } ?? c.cos
            var rank = [Int: Int](); for (r, k) in c.order.enumerated() { rank[k] = r }
            for k in 1..<got.count where rank[got[k]]! < rank[got[k - 1]]! {
                XCTAssertLessThan(abs(key[got[k]] - key[got[k - 1]]), 1e-4, "\(c.name) at \(k)")
            }
        }
    }

    func testStreamPersonLookMatchesPython() async throws {
        let f = try fixture()
        for c in f.stream {
            var ups: [[PFFixture.UpdateCell]] = [], last: LookUpdate? = nil
            let P = [c.pA, c.pB]
            try await streamPersonLook(ids: c.ids, eventOf: c.eventOf, questions: c.questions, seed: c.seed, bySize: c.bySize,
                                       judge: { id, q in P[q][id]! }, onUpdate: { u in
                ups.append([.int(u.calls), .int(u.sorted), .bool(u.firstAnswer)]); last = u; return true })
            XCTAssertEqual(ups, c.updates, c.name)
            let fin = try XCTUnwrap(last)
            XCTAssertTrue(fin.last)
            XCTAssertEqual(fin.albums.count, c.final.count, c.name)
            for (id, want) in c.final { XCTAssertEqual(fin.albums[id] ?? .some(-1), want, "\(c.name) \(id)") }
        }
    }

    /// With every photo judged, the partial rules ARE the full rules: splitPair / rankMarginPair and personAlbumFound.
    func testEverythingJudgedIsTheFullRule() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "pairing", withExtension: "json", subdirectory: "Fixtures"))
        let cases = try JSONDecoder().decode([PairCase].self, from: Data(contentsOf: url))
        for c in cases {
            let ids = c.pA.keys.sorted()
            let full = splitPair(pA: c.pA, pB: c.pB, eventOf: c.eventOf) ?? rankMarginPair(pA: c.pA, pB: c.pB)
            let got = lookPairAlbums(ids: ids, eventOf: c.eventOf, pA: c.pA, pB: c.pB)
            XCTAssertEqual(got.count, full.count, c.name)
            for id in ids { XCTAssertEqual(got[id] ?? .some(-1), full[id] ?? .some(-1), "\(c.name) \(id)") }
            let single = lookSinglePartial(ids: ids, eventOf: c.eventOf, pYes: c.pA)
            let found = Set(personAlbumFound(pYes: c.pA, question: "Is the person looking heavier?", accept: 0.7))
            XCTAssertEqual(Set(single.filter { $0.value }.keys), found, c.name)
        }
    }

    /// Weights of 1 are the original split, bit for bit (the pairing fixtures depend on it).
    func testUnitWeightsChangeNothing() {
        var g = SeededRNG(seed: 62)
        for _ in 0..<10 {
            let n = 40 + Int(g.next() % 200)
            let x = (0..<n).map { i in (i % 3 == 0 ? 2.0 : -1.5) + Double(g.next() % 1000) / 400 }
            let groups = (0..<n).map { "e\($0 / 5)" }
            let a = twoGroups(x, groups: groups), b = twoGroups(x, groups: groups, weights: [Double](repeating: 1, count: n))
            XCTAssertEqual(a.post, b.post); XCTAssertEqual(a.clear, b.clear); XCTAssertEqual(a.mid, b.mid)
        }
    }

    func testUpdateCadence() {
        XCTAssertTrue(lookUpdateDue(calls: 20, total: 5000)); XCTAssertFalse(lookUpdateDue(calls: 30, total: 5000))
        XCTAssertTrue(lookUpdateDue(calls: 480, total: 5000)); XCTAssertFalse(lookUpdateDue(calls: 500, total: 5000))
        XCTAssertTrue(lookUpdateDue(calls: 720, total: 5000)); XCTAssertTrue(lookUpdateDue(calls: 5000, total: 5000))
    }
}
