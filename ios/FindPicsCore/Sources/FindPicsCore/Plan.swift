// The search plan: same JSON schema as the Python planner (src/findpics/converse.py: Album, Plan, Step).
import Foundation

public struct Step: Codable, Equatable, Sendable {
    public var looks: [String] = []
    public var judgeQuestion: String?
    enum CodingKeys: String, CodingKey { case looks, judgeQuestion = "judge_question" }
    public init(looks: [String] = [], judgeQuestion: String? = nil) { self.looks = looks; self.judgeQuestion = judgeQuestion }
    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        looks = try c.decodeIfPresent([String].self, forKey: .looks) ?? []
        judgeQuestion = try c.decodeIfPresent(String.self, forKey: .judgeQuestion)
    }
}

public struct Album: Codable, Equatable, Sendable {
    public var name: String
    public var person: String?
    public var looks: [String] = []
    public var avoid: [String] = []
    public var judgeQuestion: String?
    public var dateFrom: String?          // ISO date, inclusive
    public var dateTo: String?            // ISO date, exclusive
    public var timePhrase: String?
    public var place: String?
    public var timeOfDay: String?         // "HH:MM-HH:MM", local clock, may wrap midnight
    public var media: String = "any"      // photo | video | any
    public var camera: String?            // "front" = selfie camera only (set by code from the word "selfie")
    public var want: String = "all"       // all | best
    public var maxItems: Int?
    public var anchor: Step?
    public var window: String?
    public var excludeQuestion: String?
    public var filterQuestion: String?
    public var withPeople: [String] = []
    public var until: Step?

    enum CodingKeys: String, CodingKey {
        case name, person, looks, avoid, place, media, camera, want, anchor, window, until
        case judgeQuestion = "judge_question", dateFrom = "date_from", dateTo = "date_to", timePhrase = "time_phrase"
        case timeOfDay = "time_of_day", maxItems = "max_items", excludeQuestion = "exclude_question"
        case filterQuestion = "filter_question", withPeople = "with_people"
    }

    public init(name: String) { self.name = name }

    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? ""
        person = try c.decodeIfPresent(String.self, forKey: .person)
        looks = try c.decodeIfPresent([String].self, forKey: .looks) ?? []
        avoid = try c.decodeIfPresent([String].self, forKey: .avoid) ?? []
        judgeQuestion = try c.decodeIfPresent(String.self, forKey: .judgeQuestion)
        dateFrom = try c.decodeIfPresent(String.self, forKey: .dateFrom)
        dateTo = try c.decodeIfPresent(String.self, forKey: .dateTo)
        timePhrase = try c.decodeIfPresent(String.self, forKey: .timePhrase)
        place = try c.decodeIfPresent(String.self, forKey: .place)
        timeOfDay = try c.decodeIfPresent(String.self, forKey: .timeOfDay)
        media = try c.decodeIfPresent(String.self, forKey: .media) ?? "any"
        camera = try c.decodeIfPresent(String.self, forKey: .camera)
        want = try c.decodeIfPresent(String.self, forKey: .want) ?? "all"
        maxItems = try c.decodeIfPresent(Int.self, forKey: .maxItems)
        anchor = try c.decodeIfPresent(Step.self, forKey: .anchor)
        window = try c.decodeIfPresent(String.self, forKey: .window)
        excludeQuestion = try c.decodeIfPresent(String.self, forKey: .excludeQuestion)
        filterQuestion = try c.decodeIfPresent(String.self, forKey: .filterQuestion)
        withPeople = try c.decodeIfPresent([String].self, forKey: .withPeople) ?? []
        until = try c.decodeIfPresent(Step.self, forKey: .until)
    }
}

public struct Plan: Codable, Equatable, Sendable {
    public var albums: [Album]
    public var notes: String = ""
    public var unknownPeople: [String] = []
    enum CodingKeys: String, CodingKey { case albums, notes, unknownPeople = "unknown_people" }
    public init(albums: [Album], notes: String = "") { self.albums = albums; self.notes = notes }
    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        albums = try c.decodeIfPresent([Album].self, forKey: .albums) ?? []
        notes = try c.decodeIfPresent(String.self, forKey: .notes) ?? ""
        unknownPeople = try c.decodeIfPresent([String].self, forKey: .unknownPeople) ?? []
    }
}
