import Foundation
import XCTest
@testable import FindPicsCore

struct GroundCase: Decodable {
    let message: String; let history: [String]; let owner: String; let people: [String]; let today: String
    let raw: Plan; let grounded: Plan
}

final class GroundingTests: XCTestCase {
    /// Same grounded plan (all fields but the notes text) as converse.ground on 2,300 9B planner outputs.
    func testMatchesPython() throws {
        let url = Bundle.module.url(forResource: "ground", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([GroundCase].self, from: Data(contentsOf: url))
        var bad = 0
        for c in cases {
            let said = (c.history + [c.message]).joined(separator: " \n ")
            var p = c.raw
            stripIdentityConditions(&p, owner: c.owner)   // as Planner.swift does, before the check (M42)
            if unanswerable(p, names: c.people + [c.owner], said: said) != nil { p = dropUnanswerable(p, said: said, names: c.people + [c.owner]) }
            let g = ground(p, message: c.message, history: c.history, today: Day(iso: c.today)!, owner: c.owner, people: c.people)
            if planJSON(g) != planJSON(c.grounded) {
                bad += 1
                if bad <= 6 { print("MISMATCH \(c.message.prefix(80))\n  swift:  \(planJSON(g).prefix(600))\n  python: \(planJSON(c.grounded).prefix(600))") }
            }
        }
        print("GROUND mismatches \(bad) of \(cases.count)")
        XCTAssertEqual(bad, 0)
    }

    /// RESULTS 47: "beach photos" asks "Is a beach clearly visible in this photo?"; other subjects keep the subject form.
    func testBeachSubjectWording() {
        func run(_ q: String, _ msg: String, looks: [String] = ["a beach"]) -> String? {
            var a = Album(name: "a"); a.looks = looks; a.judgeQuestion = q
            return ground(Plan(albums: [a]), message: msg, history: [], today: Day(iso: "2026-10-12")!, owner: "Reza",
                          people: ["Reza"]).albums[0].judgeQuestion
        }
        XCTAssertEqual(beachSubjectQuestion, "Is a beach clearly visible in this photo?")
        XCTAssertEqual(run("Is there a beach in this photo?", "beach photos"), beachSubjectQuestion)
        for q in ["Is this a photo of a beach?", "Is this a photo of beaches?", "is this a picture of the beach"] {
            XCTAssertEqual(run(q, "photos of beaches"), beachSubjectQuestion, q)
        }
        for q in ["Is this a photo of a beach trip?", "Is this a photo of a sunset on a beach?", "Is this a photo of food?",
                  "Is this a photo of a sunset?"] {
            XCTAssertEqual(run(q, "photos", looks: ["x"]), q, q)
        }
        XCTAssertEqual(run("Is there food in this photo?", "food photos", looks: ["food"]), "Is this a photo of food?")
    }

    /// RESULTS 60: the real-object rewrite is skipped when the thing asked for IS a picture, writing, figure or character
    /// (head noun of each listed thing); real things keep it, also when a depiction is only their modifier.
    func testDepictionsKeepThePlainQuestion() {
        func run(_ q: String, _ msg: String) -> String? {
            var a = Album(name: "a"); a.judgeQuestion = q
            return ground(Plan(albums: [a]), message: msg, history: [], today: Day(iso: "2026-10-14")!, owner: "me",
                          people: []).albums[0].judgeQuestion
        }
        let tail = " anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
        XCTAssertEqual(run("Is there a trail map in this photo?", "the photos of the trail maps"), "Is there a trail map in this photo?")
        XCTAssertEqual(run("Is there Mickey Mouse in this photo?", "the photos with Mickey Mouse"), "Is there Mickey Mouse in this photo?")
        XCTAssertEqual(run("Is there a dinosaur in this photo?", "photos with dinosaurs"), "Is there a dinosaur in this photo?")
        XCTAssertEqual(run("Is there a dog or a picture of a dog in this photo?", "dogs"), "Is there a dog or a picture of a dog in this photo?")
        XCTAssertEqual(run("Is there a toy car in this photo?", "my son's little cars"), "Is there a toy car in this photo?")
        XCTAssertEqual(run("Is there a dog in this photo?", "photos with a dog"), "Is there a real dog" + tail)
        XCTAssertEqual(run("Is there a tattoo artist in this photo?", "the tattoo artist"), "Is there a real tattoo artist" + tail)
        XCTAssertEqual(run("Is there a person wearing an Elmo costume in this photo?", "the guy in the Elmo suit"),
                       "Is there a real person wearing an Elmo costume" + tail)
        XCTAssertEqual(run("Is there the Santa Monica pier in this photo?", "the pier"), "Is there real the Santa Monica pier" + tail)
        XCTAssertFalse(asksForDepiction("reflection visible in the water"))
        XCTAssertFalse(asksForDepiction("whiteboard with notes"))
        XCTAssertTrue(asksForDepiction("Santa Claus"))
        XCTAssertFalse(asksForDepiction("van with the Mystery Machine logo"))
        XCTAssertFalse(asksForDepiction("Eiffel Tower"))
        XCTAssertFalse(asksForDepiction("inflatable bouncy castle in a garden"))
    }

    /// M42 (10-11): an owner album ("me") had no name words, so "Is the owner visible in this photo?" was unanswerable
    /// and the phone searched by looks. Identity is the face filter's job; only the rest of the request is asked.
    func testOwnerQuestionsGoToTheFaceFilter() {
        func run(_ q: String, looks: [String] = [], person: String = "me", owner: String = "Reza Shamji") -> (String?, [String]) {
            var a = Album(name: "a"); a.person = person; a.looks = looks; a.judgeQuestion = q
            var p = Plan(albums: [a])
            stripIdentityConditions(&p, owner: owner)
            return (p.albums[0].judgeQuestion, p.albums[0].looks)
        }
        XCTAssertNil(run("Is the owner visible in this photo?", looks: ["a close-up of a face"]).0)
        XCTAssertNil(run("Does this photo show me?").0)
        XCTAssertNil(run("Is Reza in this photo?").0)
        XCTAssertEqual(run("Is the owner visible in this photo?", looks: ["a beach", "a person"]).0, "Does this photo show a beach?")
        XCTAssertEqual(run("Does the owner look tired?").0, "Does the person in the red box look tired?")
        XCTAssertEqual(run("Am I at the beach?").0, "Is the person in the red box at the beach?")
        XCTAssertEqual(run("Does the user look heavier?").0, "Does the person in the red box look heavier?")
        XCTAssertEqual(run("Does this photo show me looking fit?").0, "Does this photo show the person in the red box looking fit?")
        XCTAssertEqual(run("Is Dad laughing?", person: "Dad").0, "Is the person in the red box laughing?")
        XCTAssertNil(run("Is Dad visible in this photo?", person: "Dad").0)
        // a non-owner album keeps "me" (the owner is not the boxed person)
        XCTAssertEqual(run("Is Dad with me?", person: "Dad").0, "Is the person in the red box with me?")
        XCTAssertEqual(run("Is this a photo of Dad?", looks: ["a beach"], person: "Dad").0, "Does this photo show a beach?")
        var r = Album(name: "a"); r.person = "me"; r.excludeQuestion = "Is the person in the photo the owner's mother?"
        var rp = Plan(albums: [r]); stripIdentityConditions(&rp, owner: "Reza")
        XCTAssertEqual(rp.albums[0].excludeQuestion, "Is the person in the photo the owner's mother?")   // stays unanswerable
        // none of these is left for the unanswerable fallback
        for q in ["Is the owner visible in this photo?", "Does the owner look tired?", "Am I at the beach?"] {
            var a = Album(name: "a"); a.person = "me"; a.looks = ["a beach"]; a.judgeQuestion = q
            var p = Plan(albums: [a])
            stripIdentityConditions(&p, owner: "Reza")
            XCTAssertNil(unanswerable(p, names: ["Reza"], said: "photos of me"), q)
        }
    }
}
