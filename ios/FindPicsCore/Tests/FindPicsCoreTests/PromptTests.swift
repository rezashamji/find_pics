import Foundation
import XCTest
@testable import FindPicsCore

struct PromptCase: Decodable {
    let message: String; let history: [String]; let current: Plan?; let owner: String; let people: [String]
    let today: String; let prompt: String; let current_json: String?
}

final class PromptTests: XCTestCase {
    /// Byte-identical to converse.build_prompt (58 cases, 29 follow-ups with a current plan).
    func testPromptMatchesPython() throws {
        let url = Bundle.module.url(forResource: "prompts", withExtension: "json", subdirectory: "Fixtures")!
        for c in try JSONDecoder().decode([PromptCase].self, from: Data(contentsOf: url)) {
            if let cur = c.current, let want = c.current_json { XCTAssertEqual(planJSON(cur), want) }
            let got = PlannerPrompt.build(message: c.message, history: c.history, current: c.current, owner: c.owner,
                                          people: c.people, today: Day(iso: c.today)!)
            XCTAssertEqual(got, c.prompt, c.message)
        }
    }
}
