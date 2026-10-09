// What the index knows about one photo / video. Two shapes:
// - FullIndexEntry: with its vectors. What the app hands to the store after reading a photo, and exactly the JSON
//   shape of the old index.json (builds before 10-07 kept every entry like this in RAM and in one JSON file).
// - IndexRecord: metadata + where its vectors are in the store's files (IndexStore). This is what stays in RAM:
//   ~200 bytes + its faces' boxes per entry instead of ~4 KB of Float32 vectors.
import Foundation

/// One face in a photo / video frame with its fingerprint (the face model's L2-normalized vector).
public struct DetectedFace: Codable, Equatable, Sendable {
    public let box: [Double]          // x1, y1, x2, y2 in image pixels (top-left origin) of the analysed image
    public let imageW: Double, imageH: Double
    public let confidence: Float
    public let embedding: [Float]
    public var frameT: Double? = nil  // videos: the second this face was seen
    public var px: Double { min(box[2] - box[0], box[3] - box[1]) }
    public init(box: [Double], imageW: Double, imageH: Double, confidence: Float, embedding: [Float], frameT: Double? = nil) {
        self.box = box; self.imageW = imageW; self.imageH = imageH; self.confidence = confidence; self.embedding = embedding
        self.frameT = frameT
    }
    public func withFrame(_ t: Double) -> DetectedFace { var f = self; f.frameT = t; return f }
}

/// One sampled moment of a video: its image vector and faces.
public struct FrameUnit: Codable, Equatable, Sendable {
    public let t: Double
    public let vector: [Float]
    public var faces: [DetectedFace]
    public init(t: Double, vector: [Float], faces: [DetectedFace]) { self.t = t; self.vector = vector; self.faces = faces }
}

/// An index entry with its vectors (field names = the old index.json keys; see IndexRecord for their meaning).
public struct FullIndexEntry: Codable, Equatable, Sendable {
    public var id: String
    public var isVideo: Bool
    public var taken: Double?
    public var localMinutes: Int?
    public var lat: Double?, lon: Double?
    public var vector: [Float]
    public var faces: [DetectedFace]? = nil
    public var place: String? = nil
    public var frames: [FrameUnit]? = nil
    public var camera: String? = nil
    public var isScreenshot: Bool? = nil
    public var lowRes: Bool? = nil
    public var faceModel: String? = nil
    public var imageVersion: Int? = nil
    public var faceSide: Double? = nil
    /// Videos: `frames` are the frames pass's sweep 1 (cover + middle + end: LazyVideo.swift), not yet every 4 s.
    public var framesPartial: Bool? = nil
    public init(id: String, isVideo: Bool, taken: Double?, localMinutes: Int?, lat: Double?, lon: Double?, vector: [Float],
                faces: [DetectedFace]? = nil, place: String? = nil, frames: [FrameUnit]? = nil, camera: String? = nil,
                isScreenshot: Bool? = nil, lowRes: Bool? = nil, faceModel: String? = nil, imageVersion: Int? = nil,
                faceSide: Double? = nil, framesPartial: Bool? = nil) {
        self.id = id; self.isVideo = isVideo; self.taken = taken; self.localMinutes = localMinutes; self.lat = lat
        self.lon = lon; self.vector = vector; self.faces = faces; self.place = place; self.frames = frames
        self.camera = camera; self.isScreenshot = isScreenshot; self.lowRes = lowRes; self.faceModel = faceModel
        self.imageVersion = imageVersion; self.faceSide = faceSide; self.framesPartial = framesPartial
    }
}

/// A face as the store keeps it in RAM: box, size, detector score, and the row of its fingerprint in the face file.
public struct StoredFace: Equatable, Sendable {
    public var box: [Double]
    public var imageW: Double, imageH: Double
    public var confidence: Float
    /// The face's own frame time field (DetectedFace.frameT as it was handed in; normally nil).
    public var frameT: Double?
    /// Videos: which sampled frame (index into IndexRecord.frameTs); nil: a photo's face.
    public var frame: Int?
    /// Row of the fingerprint in the store's face-vector file.
    public var row: Int
    public var px: Double { min(box[2] - box[0], box[3] - box[1]) }
    public init(box: [Double], imageW: Double, imageH: Double, confidence: Float, frameT: Double?, frame: Int?, row: Int) {
        self.box = box; self.imageW = imageW; self.imageH = imageH; self.confidence = confidence; self.frameT = frameT
        self.frame = frame; self.row = row
    }
}

/// One indexed photo / video without its vectors.
public struct IndexRecord: Equatable, Sendable {
    public var id: String
    public var isVideo: Bool
    public var taken: Double?          // seconds since 1970
    public var localMinutes: Int?      // wall clock where taken (nil if unknown)
    public var lat: Double?, lon: Double?
    public var place: String?          // offline place name from GPS
    public var camera: String?         // "front" | "back" | nil/"" unknown
    public var isScreenshot: Bool?
    public var lowRes: Bool?           // indexed from a smaller local copy; re-read once the original downloads
    /// The face model that made the faces (FaceProfile id); nil = buffalo_l (FindPicsCore.legacyFaceModel).
    public var faceModel: String?
    /// How the image vectors were made (the app's Embedder.imageVersion); nil = before 10-07.
    public var imageVersion: Int?
    /// Long side (px) of the read the faces came from; nil = not recorded (effectiveFaceSide infers it).
    public var faceSide: Double?
    /// Row of the entry's image vector (a photo's only unit; a video's `vector`).
    public var vectorRow: Int
    /// Videos with sampled frames: frame k's vector is row frameRow + k (frameRow == vectorRow when `vector` is frame 0).
    public var frameRow: Int
    /// Videos: the sampled frame times (one image-vector unit each); nil: a photo (its vector is the one unit).
    public var frameTs: [Double]?
    /// Photo-level faces were looked for (the old entry's `faces != nil`); false: indexed before faces existed.
    public var facesKnown: Bool
    /// The photo's faces (frame nil) and the video frames' faces (frame k).
    public var faces: [StoredFace]
    /// Videos: frameTs are the frames pass's sweep 1 (cover + middle + end), not yet every 4 s (videoFramesStage 1).
    /// Stored as a flag bit, so older stores (no bit) read as sweep 2 / full sampling.
    public var framesPartial: Bool

    public init(id: String, isVideo: Bool, taken: Double?, localMinutes: Int?, lat: Double?, lon: Double?, place: String?,
                camera: String?, isScreenshot: Bool?, lowRes: Bool?, faceModel: String?, imageVersion: Int?, faceSide: Double?,
                vectorRow: Int, frameRow: Int, frameTs: [Double]?, facesKnown: Bool, faces: [StoredFace],
                framesPartial: Bool = false) {
        self.id = id; self.isVideo = isVideo; self.taken = taken; self.localMinutes = localMinutes; self.lat = lat
        self.lon = lon; self.place = place; self.camera = camera; self.isScreenshot = isScreenshot; self.lowRes = lowRes
        self.faceModel = faceModel; self.imageVersion = imageVersion; self.faceSide = faceSide; self.vectorRow = vectorRow
        self.frameRow = frameRow; self.frameTs = frameTs; self.facesKnown = facesKnown; self.faces = faces
        self.framesPartial = framesPartial
    }

    /// Image-vector rows this entry owns in the image file (for compaction accounting).
    public var ownedImageRows: Int {
        guard let ts = frameTs else { return 1 }
        return frameRow == vectorRow ? max(ts.count, 1) : 1 + ts.count
    }
    /// Any face, in the photo or in a frame (old IndexEntry.hasFaces).
    public var hasFaces: Bool { !faces.isEmpty }
    /// The photo itself has faces (old `!(faces ?? []).isEmpty`).
    public var hasPhotoFaces: Bool { faces.contains { $0.frame == nil } }
    public var facesCurrent: Bool { faceVectorsCurrent(model: faceModel, hasFaces: hasFaces) }
    /// A video indexed from its cover frame only (one image vector, the still PhotoKit holds; no frame times yet): the
    /// frames pass samples it later (FindPicsCore/LazyVideo.swift). No store format change: such an entry is simply a
    /// video without frameTs.
    public var videoFramesPending: Bool { isVideo && frameTs == nil }
    /// Videos: how far the frames pass got (LazyVideo.swift). 0 = cover frame only, 1 = sweep 1 (cover + middle and
    /// end frames), 2 = every 4 s (sweep 2, or sampled in full when first indexed). Photos: nil.
    public var videoFramesStage: Int? {
        guard isVideo else { return nil }
        return frameTs == nil ? 0 : framesPartial ? 1 : 2
    }
    /// A cover frame is read like a photo (448 px), so its faces count by their recorded read size, not as sampled frames.
    public var faceSideEffective: Double {
        effectiveFaceSide(stored: faceSide, isVideo: isVideo && !videoFramesPending, imageVersion: imageVersion, lowRes: lowRes)
    }
    public var facesFullSize: Bool { faceSideEffective >= faceReadSide }
    /// A photo whose faces were found on a small read: the face upgrade re-reads it at faceReadSide.
    public var needsFaceUpgrade: Bool { !isVideo && hasPhotoFaces && !facesFullSize }
    /// The faces people searches use, as the old allFaces listed them: a video's frame faces (frameT = the frame's
    /// time), otherwise the photo's faces.
    public var searchFaces: [StoredFace] {
        if let ts = frameTs {
            return faces.compactMap { f in f.frame.map { k in var g = f; g.frameT = ts[k]; return g } }
        }
        return faces.filter { $0.frame == nil }
    }
}
