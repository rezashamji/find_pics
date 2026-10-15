import Foundation
import XCTest
@testable import FindPicsCore

/// RESULTS 59d: QuestionTree = findpics.compose on 660 benchmark judge questions with their libraries' place texts
/// (Fixtures/compose.json from eval/facts_scorer/r59d/compose_fixture.py).
final class QuestionTreeTests: XCTestCase {
    struct Case: Decodable { let lib: String; let q: String; let tree: Node }
    struct MaskCase: Decodable { let name: String; let rows: [Int] }
    struct FoldCase: Decodable { let s: String; let fold: String }
    struct SplitCase: Decodable {
        let q: String; let split: [Part]?
        enum Part: Decodable { case op(String), parts([String])
            init(from d: Decoder) throws { let c = try d.singleValueContainer()
                if let s = try? c.decode(String.self) { self = .op(s) } else { self = .parts(try c.decode([String].self)) } } }
    }
    struct Fixture: Decodable {
        let libs: [String: [String]]; let cases: [Case]; let hand_places: [String]; let masks: [MaskCase]; let folds: [FoldCase]
        let splits: [SplitCase]
    }
    final class Node: Decodable, Equatable, CustomStringConvertible {
        var leaf: String?; var meta: [Int]?; var rest: Node?; var or: [Node]?; var and: [Node]?
        var isMeta = false
        enum K: String, CodingKey { case leaf, meta, rest, or, and }
        init() {}
        required init(from d: Decoder) throws {
            let c = try d.container(keyedBy: K.self)
            leaf = try c.decodeIfPresent(String.self, forKey: .leaf)
            isMeta = c.contains(.meta)
            meta = try c.decodeIfPresent([Int].self, forKey: .meta)
            rest = try c.decodeIfPresent(Node.self, forKey: .rest)
            or = try c.decodeIfPresent([Node].self, forKey: .or)
            and = try c.decodeIfPresent([Node].self, forKey: .and)
        }
        static func == (a: Node, b: Node) -> Bool {
            a.leaf == b.leaf && a.isMeta == b.isMeta && a.meta == b.meta && a.rest == b.rest && a.or == b.or && a.and == b.and
        }
        var description: String {
            if let l = leaf { return "leaf(\(l))" }
            if isMeta { return "meta(\(meta ?? []), \(rest.map { $0.description } ?? "nil"))" }
            if let o = or { return "or\(o)" }
            return "and\(and ?? [])"
        }
    }

    static func node(_ t: QuestionTree) -> Node {
        let n = Node()
        switch t {
        case .leaf(let q): n.leaf = q
        case .meta(let mk, let rest):
            n.isMeta = true; n.meta = mk.indices.filter { mk[$0] }; n.rest = rest.map(node)
        case .parts(let op, let ch):
            if op == .or { n.or = ch.map(node) } else { n.and = ch.map(node) }
        }
        return n
    }

    func testMatchesPython() throws {
        let url = Bundle.module.url(forResource: "compose", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        for c in f.folds { XCTAssertEqual(foldText(c.s), c.fold, c.s) }
        for c in f.splits {
            let got = splitParts(c.q)
            if let e = c.split, case .op(let op) = e[0], case .parts(let ps) = e[1] {
                XCTAssertEqual(got?.0.rawValue, op, c.q); XCTAssertEqual(got?.1, ps, c.q)
            } else { XCTAssertNil(got, c.q) }
        }
        let hand = PlaceIndex(f.hand_places)
        for m in f.masks { XCTAssertEqual(hand.mask(m.name).indices.filter { hand.mask(m.name)[$0] }, m.rows, m.name) }
        var idx = [String: PlaceIndex](), bad = 0
        for c in f.cases {
            let PI = idx[c.lib] ?? PlaceIndex(f.libs[c.lib]!)
            idx[c.lib] = PI
            let got = Self.node(QuestionTree.decompose(c.q, places: PI))
            if got != c.tree { bad += 1; if bad <= 10 { print("COMPOSE mismatch: \(c.q)\n  swift \(got)\n  python \(c.tree)") } }
        }
        print("COMPOSE mismatches \(bad)/\(f.cases.count)")
        XCTAssertEqual(bad, 0)
    }

    func testEvaluateUnionMetaAnd() async {
        let t = QuestionTree.meta([true, false, true], .parts(.or, [.leaf("A?"), .leaf("B?")]))
        let P: [String: [Int: Float]] = ["A?": [0: 0.9, 2: 0.1], "B?": [0: 0.2, 2: 0.8]]
        let out = await t.evaluate([0, 1, 2]) { items, q in items.map { P[q]![$0]! } }
        XCTAssertEqual(out, [0.9, 0.01, 0.8])
        var one = [Float]()
        for i in 0..<3 { one.append(await t.evaluateOne(i) { q in P[q]![i] ?? 0 }) }
        XCTAssertEqual(one, out)
        let plain = await QuestionTree.judge("Is there a dog?", items: [0, 1], places: nil) { items, _ in items.map { _ in 0.5 } }
        XCTAssertEqual(plain, [0.5, 0.5])
    }
}
