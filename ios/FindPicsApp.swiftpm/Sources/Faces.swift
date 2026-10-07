// Faces on the phone: Apple Vision finds faces + landmarks (built into iOS, free, on-device); the 5 landmarks align
// the face to the ArcFace template (FindPicsCore.similarityTransform, identical to insightface); the Core ML face
// model gives the 512-d fingerprint. Which model: FindPicsCore.FaceProfile.shipped (the ONE place that selects it; its
// cuts go with it). Shipped: fal AuraFace-v1 (Apache-2.0), Models/face_auraface.mlpackage (~130 MB fp16), whose graph
// embeds the crop and its mirror and averages them (scripts/convert_face_coreml.py). buffalo_l (non-commercial) is not
// bundled. Matching (who is this) = FindPicsCore.itemPersonScores, identical to people.py.
import CoreGraphics
import CoreImage
import CoreML
@preconcurrency import FindPicsCore
import Vision

struct DetectedFace: Codable {
    let box: [Double]          // x1, y1, x2, y2 in image pixels (top-left origin) of the analysed image
    let imageW: Double, imageH: Double
    let confidence: Float
    let embedding: [Float]
    var frameT: Double? = nil  // videos: the second this face was seen
    var px: Double { min(box[2] - box[0], box[3] - box[1]) }
    func withFrame(_ t: Double) -> DetectedFace { var f = self; f.frameT = t; return f }
}

enum FaceEngineError: Error { case modelNotBundled(String) }

final class FaceEngine: @unchecked Sendable {   // immutable after init (MLModel is not marked Sendable)
    /// Bundled Core ML file per face model (Sources/Models/<name>.mlpackage; docs/BUILD_ON_MAC.md).
    static let resource: [String: String] = ["auraface_flip": "face_auraface", "buffalo_l": "face_buffalo_l"]
    let model: MLModel
    let profile: FaceProfile
    init(profile: FaceProfile = .shipped) throws {
        self.profile = profile
        guard let name = FaceEngine.resource[profile.id],
              let url = Bundle.module.url(forResource: "Models/" + name, withExtension: "mlmodelc")
                ?? Bundle.module.url(forResource: "Models/" + name, withExtension: "mlpackage")
        else { throw FaceEngineError.modelNotBundled(profile.id) }
        let cfg = MLModelConfiguration(); cfg.computeUnits = .all
        model = try MLModel(contentsOf: url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url), configuration: cfg)
    }

    /// All faces in a photo with their fingerprints.
    func faces(in cg: CGImage) throws -> [DetectedFace] {
        let W = Double(cg.width), H = Double(cg.height)
        let req = VNDetectFaceLandmarksRequest()
        try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
        guard let obs = req.results, !obs.isEmpty else { return [] }
        let rgba = FaceEngine.rgba(cg)
        var out = [DetectedFace]()
        for o in obs {
            // Vision: normalized, bottom-left origin -> pixels, top-left origin
            let b = o.boundingBox
            let box: [Double] = [b.minX * W, (1 - b.maxY) * H, b.maxX * W, (1 - b.minY) * H]
            guard let e = try embedding(o, rgba: rgba, w: cg.width, h: cg.height) else { continue }
            out.append(DetectedFace(box: box, imageW: W, imageH: H, confidence: o.confidence, embedding: e))
        }
        return out
    }

    /// The same faces again with THIS model's fingerprints (the app switched face models): landmarks are found inside
    /// each stored box (Vision's face-rectangle step is skipped), so the face list, boxes, sizes and detector scores stay
    /// as they were and only `embedding` changes. A face whose landmarks are not found again is dropped. `cg` is the
    /// same photo / frame, at any size (boxes are image-relative).
    func reembed(_ old: [DetectedFace], in cg: CGImage) throws -> [DetectedFace] {
        let rgba = FaceEngine.rgba(cg)
        var out = [DetectedFace]()
        for f in old {
            // stored box: pixels of the analysed image, top-left origin -> Vision: normalized, bottom-left origin
            let r = CGRect(x: f.box[0] / f.imageW, y: 1 - f.box[3] / f.imageH,
                           width: (f.box[2] - f.box[0]) / f.imageW, height: (f.box[3] - f.box[1]) / f.imageH)
            let req = VNDetectFaceLandmarksRequest()
            req.inputFaceObservations = [VNFaceObservation(boundingBox: r)]
            try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
            guard let o = req.results?.first, let e = try embedding(o, rgba: rgba, w: cg.width, h: cg.height) else { continue }
            out.append(DetectedFace(box: f.box, imageW: f.imageW, imageH: f.imageH, confidence: f.confidence, embedding: e,
                                    frameT: f.frameT))
        }
        return out
    }

    /// Landmarks -> aligned 112x112 crop -> fingerprint (nil when the 5 points cannot be read).
    private func embedding(_ o: VNFaceObservation, rgba: [UInt8], w: Int, h: Int) throws -> [Float]? {
        guard let lm = o.landmarks,
              let five = FaceEngine.fivePoints(lm, faceBox: o.boundingBox, W: Double(w), H: Double(h)) else { return nil }
        let M = similarityTransform(from: five)
        let crop = FaceEngine.warp(rgba, w: w, h: h, M: M)
        let arr = try MLMultiArray(shape: [1, 3, 112, 112], dataType: .float32)
        let p = arr.dataPointer.bindMemory(to: Float.self, capacity: 3 * 112 * 112)
        for i in 0..<(112 * 112) { for c in 0..<3 { p[c * 112 * 112 + i] = (crop[i * 3 + c] - 127.5) / 127.5 } }
        let name = model.modelDescription.inputDescriptionsByName.keys.first!
        let res = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [name: arr]))
        // the converted models are L2-normalized inside (flip-averaged too for AuraFace); "embedding" names the output
        guard let m = (res.featureValue(for: "embedding") ?? res.featureValue(for: res.featureNames.first!))?.multiArrayValue else { return nil }
        return (0..<m.count).map { Float(truncating: m[$0]) }
    }

    /// Eye centres, nose tip, mouth corners (image-left first), in pixels, top-left origin. Vision's points are
    /// normalized to the face box. NOTE: these differ slightly from SCRFD's keypoints the model was trained with;
    /// the on-device self-check compares fingerprints with the server's for the same photo.
    static func fivePoints(_ lm: VNFaceLandmarks2D, faceBox b: CGRect, W: Double, H: Double) -> [(Double, Double)]? {
        func pts(_ r: VNFaceLandmarkRegion2D?) -> [(Double, Double)] {
            (r?.normalizedPoints ?? []).map { p in (Double(b.minX + p.x * b.width) * W, (1 - Double(b.minY + p.y * b.height)) * H) }
        }
        func mean(_ a: [(Double, Double)]) -> (Double, Double)? {
            a.isEmpty ? nil : (a.map { $0.0 }.reduce(0, +) / Double(a.count), a.map { $0.1 }.reduce(0, +) / Double(a.count))
        }
        guard var e1 = mean(pts(lm.leftPupil).isEmpty ? pts(lm.leftEye) : pts(lm.leftPupil)),
              var e2 = mean(pts(lm.rightPupil).isEmpty ? pts(lm.rightEye) : pts(lm.rightPupil)) else { return nil }
        if e1.0 > e2.0 { swap(&e1, &e2) }
        let nose = pts(lm.nose), lips = pts(lm.outerLips)
        guard let tip = nose.max(by: { $0.1 < $1.1 }), let ml = lips.min(by: { $0.0 < $1.0 }), let mr = lips.max(by: { $0.0 < $1.0 }) else { return nil }
        return [e1, e2, tip, ml, mr]
    }

    static func rgba(_ cg: CGImage) -> [UInt8] {
        var px = [UInt8](repeating: 0, count: cg.width * cg.height * 4)
        px.withUnsafeMutableBytes { buf in     // `&px` would give CGContext a pointer that dies after the init call
            let ctx = CGContext(data: buf.baseAddress, width: cg.width, height: cg.height, bitsPerComponent: 8, bytesPerRow: cg.width * 4,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))   // row 0 = top of the image
        }
        return px
    }

    /// cv2.warpAffine(img, M, (112, 112), borderValue 0) with bilinear sampling: crop(u,v) = img(M^-1 (u,v)). RGB floats.
    static func warp(_ img: [UInt8], w: Int, h: Int, M: [[Double]]) -> [Float] {
        let a = M[0][0], b = M[0][1], c = M[0][2], d = M[1][0], e = M[1][1], f = M[1][2]
        let det = a * e - b * d
        var out = [Float](repeating: 0, count: 112 * 112 * 3)
        for v in 0..<112 { for u in 0..<112 {
            let x = ( e * (Double(u) - c) - b * (Double(v) - f)) / det
            let y = (-d * (Double(u) - c) + a * (Double(v) - f)) / det
            let x0 = Int(floor(x)), y0 = Int(floor(y)), fx = Float(x - floor(x)), fy = Float(y - floor(y))
            for ch in 0..<3 {
                func at(_ xx: Int, _ yy: Int) -> Float { (xx < 0 || yy < 0 || xx >= w || yy >= h) ? 0 : Float(img[(yy * w + xx) * 4 + ch]) }
                let top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx, bot = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
                out[(v * 112 + u) * 3 + ch] = top * (1 - fy) + bot * fy
            }
        } }
        return out
    }
}
