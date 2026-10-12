// The photo judge's P(yes) cutoff per question form (RESULTS 35, 10-07; phone judge Qwen3-VL-4B 4-bit).
// "Is this a photo of X?" (the planner's form for "X photos" searches) at 0.99: on six such queries in other libraries,
// wrong photos kept 48/117 -> 19/117 for right kept 91/126 -> 82/126; on 4 fully judged libraries (1,950 blind labels)
// beach / sunset / food precision 0.49 / 0.46 / 0.53 -> 0.59 / 0.56 / 0.66 at <= 1 real photo lost per query. The 11
// real photos it drops (viewed) have the subject small or beside the point. Object questions ("Is there a real X
// anywhere ...?"), person looks and fact questions keep 0.7. Measured on the vision judge only: other judges keep 0.7.
// 10-11 RAISED 0.99 -> 0.999 (eval/judge_cut_sweep, RESULTS 46): on the whole judged set (RESULTS 34 estimator + R45
// census) eye precision beach .62->.67, sunset .74->.83, food .60->.68 for recall .995->.995 / .921->.884 / .931->.931;
// photos between 0.99 and 0.999 were 0/10 right for food and beach by eye. Fast mode replayed at 0.999 (phoneFastParams,
// audit4, 20 seeds): stops at the same place, first-answer precision .672->.734, recall .958->.950 pooled.
// 10-12 PER-WORD EXCEPTIONS (eval/scene_questions, RESULTS 47; public DISBench, V3 eye labels). Keyed on the exact
// question templates, measured for these two words only (the table must not grow by guessing):
// - beach: Grounding writes "Is a beach clearly visible in this photo?" instead of "Is this a photo of a beach?" (that
//   one says yes to open sea, palms, cliffs, lake boats). It keeps the strict subject cut 0.999: precision .69 -> .85,
//   recall .968 -> .949.
// - food: "Is this a photo of food?" at 0.9999 (logit gap 9.2): precision .70 -> .80, recall .957 -> .947.
// Sunset and every other subject stay 0.999; object questions 0.7.
import Foundation

public let defaultAccept = 0.7
public let subjectPhotoAccept = 0.999
public let foodSubjectAccept = 0.9999
/// The beach search's subject question (Grounding.swift; Python converse.BEACH_QUESTION).
public let beachSubjectQuestion = "Is a beach clearly visible in this photo?"

/// True for the "Is this a photo of X?" form (any article / photo word), not for "Is there ... in this photo?".
/// Also true for the beach wording that replaces "Is this a photo of a beach?" (same search, same rules).
public func isSubjectPhotoQuestion(_ q: String) -> Bool {
    q.range(of: #"^\s*is\s+this\s+(a\s+|an\s+)?(photo|picture|image|pic)\s+of\b"#,
            options: [.regularExpression, .caseInsensitive]) != nil || isBeachSubjectQuestion(q)
}

/// "Is a beach clearly visible in this photo?" (case and surrounding spaces ignored).
public func isBeachSubjectQuestion(_ q: String) -> Bool {
    q.range(of: #"^\s*is\s+a\s+beach\s+clearly\s+visible\s+in\s+this\s+photo\s*\??\s*$"#,
            options: [.regularExpression, .caseInsensitive]) != nil
}

/// "Is this a photo of food?" exactly (any photo word / article; not "food and drinks", not "a food truck").
public func isFoodSubjectQuestion(_ q: String) -> Bool {
    q.range(of: #"^\s*is\s+this\s+(a\s+|an\s+)?(photo|picture|image|pic)\s+of\s+food\s*\??\s*$"#,
            options: [.regularExpression, .caseInsensitive]) != nil
}

/// Cutoff for the album's main judge question. `strictSubjects`: the judge is the vision judge the rule was measured on.
public func judgeCutoff(question: String, strictSubjects: Bool) -> Double {
    guard strictSubjects else { return defaultAccept }
    if isFoodSubjectQuestion(question) { return foodSubjectAccept }
    return isSubjectPhotoQuestion(question) ? subjectPhotoAccept : defaultAccept
}
