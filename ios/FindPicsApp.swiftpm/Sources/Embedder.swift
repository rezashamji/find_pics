// Image and text vectors with the Core ML PE-Core-B-16 towers (converted by scripts/convert_coreml.py; int8 weights).
// Image preprocessing = open_clip's for PE-Core-B-16: resize to 224x224 (no crop), RGB, (x - 0.5) / 0.5.
// Text: CLIP BPE ids, context 32 (FindPicsCore.ClipTokenizer, identical to open_clip). Outputs are L2-normalized.
import CoreImage
import CoreML
import FindPicsCore
import Foundation

final class Embedder {
    let image: MLModel, text: MLModel
    let tokenizer: ClipTokenizer
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
        let sx = CGFloat(side) / ci.extent.width, sy = CGFloat(side) / ci.extent.height
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        var px = [UInt8](repeating: 0, count: side * side * 4)
        ctx.render(scaled, toBitmap: &px, rowBytes: side * 4, bounds: CGRect(x: 0, y: 0, width: side, height: side),
                   format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
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
