import Foundation
import XCTest
@testable import FindPicsCore

struct AskJSON: Decodable, Equatable {
    let albums: [Int]; let name: String?; let kind: String; let words: String; let key: String; let display: String
    let prompt: String; let judgeName: String; let fromSaved: Bool
}
struct SubjectAskCase: Decodable { let message: String; let history: [String]; let saved: [SavedSubject]; let plan: Plan; let asks: [AskJSON]; let planned: Plan }
struct ResolveCase: Decodable { let message: String; let saved: [SavedSubject]; let index: Int? }
struct SubjectAskFixture: Decodable { let cases: [SubjectAskCase]; let resolve: [ResolveCase] }

final class SubjectAskTests: XCTestCase {
    /// Same asks and rewritten plan as findpics.subjects on 2,300 grounded planner outputs + hand-written cases.
    func testMatchesPython() throws {
        let url = Bundle.module.url(forResource: "subjects", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(SubjectAskFixture.self, from: Data(contentsOf: url))
        var bad = 0, withAsk = 0
        for c in f.cases {
            let asks = namedSubjects(c.plan, message: c.message, history: c.history, saved: c.saved)
            let got = asks.map { AskJSON(albums: $0.albums, name: $0.name, kind: $0.kind, words: $0.words, key: $0.key, display: $0.display,
                                         prompt: $0.prompt, judgeName: $0.judgeName, fromSaved: $0.fromSaved) }
            if !asks.isEmpty { withAsk += 1 }
            let planned = subjectPlan(c.plan, asks: asks)
            if got != c.asks || planJSON(planned) != planJSON(c.planned) {
                bad += 1
                if bad <= 6 { print("MISMATCH \(c.message.prefix(80))\n  swift:  \(got) \(planJSON(planned).prefix(400))\n  python: \(c.asks) \(planJSON(c.planned).prefix(400))") }
            }
        }
        print("SUBJECT ASK mismatches \(bad) of \(f.cases.count) (\(withAsk) with an ask)")
        XCTAssertEqual(bad, 0)
        XCTAssertGreaterThan(withAsk, 50)
        for r in f.resolve {
            let p = Plan(albums: [Album(name: "x")])
            for a in namedSubjects(p, message: r.message) { XCTAssertEqual(resolveSubject(a, saved: r.saved), r.index, r.message) }
        }
    }

    func testNamedPetAsksAndKeepsTheCondition() {
        var a = Album(name: "Max at the beach"); a.person = "Max"; a.judgeQuestion = "Is Max on a beach?"
        let p = Plan(albums: [a])
        let asks = namedSubjects(p, message: "my dog Max at the beach")
        XCTAssertEqual(asks.map { $0.name }, ["Max"]); XCTAssertEqual(asks.first?.kind, "dog")
        XCTAssertEqual(asks.first?.prompt, "Show me Max: pick 1-3 clear photos of Max.")
        let q = subjectPlan(p, asks: asks).albums[0]
        XCTAssertNil(q.person); XCTAssertEqual(q.judgeQuestion, "Is the dog on a beach?")
    }

    func testNoAskForCategoriesNegationsCompounds() {
        let p = Plan(albums: [Album(name: "x")])
        for m in ["a red car", "dogs at the park", "my dogs", "photos without my dog", "my car keys", "my dog's bowl", "selfies in my car"] {
            XCTAssertTrue(namedSubjects(p, message: m).isEmpty, m)
        }
    }
}
