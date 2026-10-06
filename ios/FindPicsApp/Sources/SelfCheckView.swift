// Self-check: recompute, on this phone, vectors the server computed for bundled test images (SelfCheck/refs.json)
// and show the cosine. > 0.99 = the Core ML models + image preparation match the server; a low number means a
// conversion or orientation/normalization problem to fix before trusting any search.
import CoreImage
import os
import SwiftUI
import UIKit

struct SelfCheckView: View {
    let embedder: Embedder?
    let faces: FaceEngine?
    @State var lines: [String] = []

    var body: some View {
        List {
            // memory iOS still lets this app use right now (the budget question: can the 9B ever fit?)
            Text("App memory available: \(String(format: "%.2f", Double(os_proc_available_memory()) / 1_073_741_824)) GB "
                 + "(device RAM \(ProcessInfo.processInfo.physicalMemory / 1_073_741_824) GB)").font(.footnote.monospaced())
            Button("Run self-check") { lines = run() }
            ForEach(lines, id: \.self) { Text($0).font(.footnote.monospaced()) }
        }
    }

    func cos(_ a: [Float], _ b: [Double]) -> Double {
        var d = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) { d += Double(x) * y; na += Double(x * x); nb += y * y }
        return d / (sqrt(na * nb) + 1e-12)
    }

    func run() -> [String] {
        guard let url = Bundle.module.url(forResource: "SelfCheck/refs", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let refs = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return ["refs.json missing"] }
        var out = [String]()
        if let e = embedder {
            for (name, v) in (refs["images"] as? [String: [Double]]) ?? [:] {
                let u = Bundle.module.url(forResource: "SelfCheck/" + name, withExtension: nil)!
                if let ci = CIImage(contentsOf: u), let got = try? e.vector(of: ci) { out.append(String(format: "image %@: cosine %.4f", name, cos(got, v))) }
            }
            for (t, v) in (refs["texts"] as? [String: [Double]]) ?? [:] {
                if let got = try? e.vector(of: t) { out.append(String(format: "text '%@': cosine %.4f", t, cos(got, v))) }
            }
        } else { out.append("photo scanner models missing") }
        if let fe = faces, let f = refs["face"] as? [String: Any], let v = f["embedding"] as? [Double],
           let u = Bundle.module.url(forResource: "SelfCheck/face", withExtension: "png"), let cg = UIImage(contentsOfFile: u.path)?.cgImage {
            let found = (try? fe.faces(in: cg)) ?? []
            if let best = found.max(by: { $0.confidence < $1.confidence }) {
                out.append(String(format: "face: cosine %.4f (Vision landmarks vs server keypoints; > 0.9 is fine)", cos(best.embedding, v)))
            } else { out.append("face: Vision found no face") }
        } else { out.append("face model missing") }
        return out
    }
}
