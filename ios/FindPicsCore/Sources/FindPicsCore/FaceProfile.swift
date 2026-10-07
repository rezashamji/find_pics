// THE place that says which face model the app uses and which cuts go with it (twin of src/findpics/face_profiles.py;
// FaceProfileTests checks both tables are identical). A face cut only means something together with the model that
// set it: AuraFace + flip scores different people higher than buffalo_l does, so buffalo_l's 0.40 would let strangers in.
// To switch models (e.g. buffalo_l again under a paid licence): calibrate it (eval/face_calibrate.py), add its row in
// both tables, set `shipped`, bundle its Core ML file (FaceEngine.resource). The index notices the change and
// re-embeds the faces (PhotoIndex.reembedStaleFaces); saved people are re-derived from the same faces.
import Foundation

public struct FaceProfile: Equatable, Sendable, Codable {
    public let id: String
    /// faceGroups ("the faces I see most often") and assignedGroups (a group already belongs to a named person)
    public let group: Float
    /// expandRefs: faces at least this close to a reference become references (query-time clustering)
    public let expand: Float
    /// a photo shows the person (item score from itemPersonScores)
    public let accept: Float
    /// otherIdentities: a face group matching the person below this on average is someone else; a face closer to such
    /// a group than to the person does not count ("closer to someone else" rule)
    public let other: Float
    /// server only: [maybe, accept) is a "possible" list for the user to confirm
    public let maybe: Float
    /// pickRefFaces: below this best mean cosine the picked photos share no face (take the largest face)
    public let pickFloor: Float
    /// server only (refs_from_items): references whose consensus with the other tagged photos is below this are dropped
    public let consensus: Float

    enum CodingKeys: String, CodingKey { case id, group, expand, accept, other, maybe, pickFloor = "pick_floor", consensus }

    /// InsightFace buffalo_l (w600k_r50). NON-COMMERCIAL weights: dev only, not in the shipped app.
    public static let buffaloL = FaceProfile(id: "buffalo_l", group: 0.55, expand: 0.55, accept: 0.40, other: 0.40, maybe: 0.30,
                                             pickFloor: 0.30, consensus: 0.20)
    /// fal AuraFace-v1 (glintr100, Apache-2.0), crop + mirror averaged (inside the Core ML graph). Cuts: RESULTS 36.
    public static let aurafaceFlip = FaceProfile(id: "auraface_flip", group: 0.62, expand: 0.60, accept: 0.53, other: 0.53,
                                                 maybe: 0.42, pickFloor: 0.42, consensus: 0.30)
    public static let all: [FaceProfile] = [buffaloL, aurafaceFlip]
    /// The face model this build of the app uses.
    public static let shipped = aurafaceFlip
    public static func named(_ id: String) -> FaceProfile? { all.first { $0.id == id } }
}

/// Index entries written before entries recorded their face model were made by buffalo_l (every phone build until 10-07).
public let legacyFaceModel = "buffalo_l"

/// May this entry's face vectors be compared with vectors of `current`? An entry without faces has nothing to mix.
public func faceVectorsCurrent(model: String?, hasFaces: Bool, current: String = FaceProfile.shipped.id) -> Bool {
    !hasFaces || (model ?? legacyFaceModel) == current
}
