// Image and text vectors with the Core ML PE-Core-B-16 towers (scripts/convert_coreml.py). Image tower fp16 (int8
// weights cost ~0.995 cosine and top-600 overlap 0.96 vs the server, 10-07); text tower int8 (0.9999 on the phone).
// Image preprocessing = open_clip's for PE-Core-B-16, byte for byte: the photo as 8-bit sRGB, Pillow's
// resize((224, 224), BILINEAR) (squash, no crop; FindPicsCore.PILResize, bit-exact with Pillow), (x - 0.5) / 0.5.
// Until 10-07 the resize was Core Image's affine transform: one 2x2 bilinear tap per output pixel, no antialiasing,
// which put Self-check at 0.9345 / 0.9765 (eval/coreml_preproc_parity.py).
// Text: CLIP BPE ids, context 32 (FindPicsCore.ClipTokenizer, identical to open_clip). Outputs are L2-normalized.
import CoreImage
import CoreML
@preconcurrency import FindPicsCore
import Foundation

final class Embedder: @unchecked Sendable {   // immutable after init; shared by the index actor and searches
    let image: MLModel, text: MLModel
    let tokenizer: ClipTokenizer
    /// Stamped on index entries (IndexEntry.imageVersion). 2 = Pillow-exact resize + fp16 image weights (10-07).
    /// Bump it whenever the image vectors change (preparation or model): older entries are then re-indexed.
    static let imageVersion = 2
    let ctx = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])

    init() throws {
        let cfg = MLModelConfiguration(); cfg.computeUnits = .all          // Neural Engine when it can
        func load(_ name: String) throws -> MLModel {
            let url = Bundle.module.url(forResource: name, withExtension: "mlmodelc")
                ?? Bundle.module.url(forResource: name, withExtension: "mlpackage")!
            let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
            return try MLModel(contentsOf: compiled, configuration: cfg)
        }
        image = try load("Models/pe_core_image"); text = try load("Models/pe_core_text")
        tokenizer = try ClipTokenizer()
    }

    func vector(of ci: CIImage) throws -> [Float] {
        let side = 224
        var src = ci
        // The server decodes at most 1600 px (findpics.media.load_image: antialiased thumbnail); the index reads 1280.
        let longest = max(ci.extent.width, ci.extent.height)
        if longest > 1600 {
            src = ci.applyingFilter("CILanczosScaleTransform",
                                    parameters: [kCIInputScaleKey: 1600 / longest, kCIInputAspectRatioKey: 1.0])
        }
        // whole pixels only (a Lanczos edge can be a fraction of a pixel: half transparent, it would darken the border)
        let w = max(Int(src.extent.width), 1), h = max(Int(src.extent.height), 1)
        var full = [UInt8](repeating: 0, count: w * h * 4)
        ctx.render(src, toBitmap: &full, rowBytes: w * 4,
                   bounds: CGRect(x: src.extent.minX, y: src.extent.minY, width: CGFloat(w), height: CGFloat(h)),
                   format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let px = PILResize.bilinear(full, width: w, height: h, channels: 4, toWidth: side, toHeight: side)
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: side), NSNumber(value: side)], dataType: .float32)
        let p = arr.dataPointer.bindMemory(to: Float.self, capacity: 3 * side * side)
        for y in 0..<side { for x in 0..<side { for c in 0..<3 {
            p[c * side * side + y * side + x] = (Float(px[(y * side + x) * 4 + c]) / 255 - 0.5) / 0.5
        } } }
        let name = image.modelDescription.inputDescriptionsByName.keys.first!
        let out = try image.prediction(from: MLDictionaryFeatureProvider(dictionary: [name: arr]))
        return Embedder.floats(out)
    }

    func vector(of words: String) throws -> [Float] {
        let ids = tokenizer(words)
        let arr = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        for (i, v) in ids.enumerated() { arr[i] = NSNumber(value: v) }
        let name = text.modelDescription.inputDescriptionsByName.keys.first!
        let out = try text.prediction(from: MLDictionaryFeatureProvider(dictionary: [name: arr]))
        return Embedder.floats(out)
    }

    static func floats(_ out: MLFeatureProvider) -> [Float] {
        let m = out.featureValue(for: out.featureNames.first!)!.multiArrayValue!
        var v = (0..<m.count).map { Float(truncating: m[$0]) }
        let n = sqrt(v.map { $0 * $0 }.reduce(0, +)); if n > 0 { v = v.map { $0 / n } }
        return v
    }
}
