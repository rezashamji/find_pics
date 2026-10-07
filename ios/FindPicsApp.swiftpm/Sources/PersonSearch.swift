// Person albums ("me looking heavier"): identity ONLY from faces (the judge cannot recognise people); the judge rates
// the look on a crop around THAT person with a red box; photos are ranked within the person's own photos; two
// opposite looks of one person are split by FindPicsCore.splitPair (the rule measured on Reza's labeled photos).
import CoreGraphics
import CoreImage
@preconcurrency import FindPicsCore
import Foundation
import UIKit

struct PersonScored { var ids: [String]; var pYes: [String: Double] }

extension SearchEngine {
    /// The face cuts belong to the face model (FindPicsCore.FaceProfile.shipped; buffalo_l was 0.40 / 0.55 / 0.40).
    static let faceProfile = FaceProfile.shipped
    static var personAccept: Float { faceProfile.accept }

    /// Identity-matched photos of the album's person (in scope) and, if the album has a look question, the judge's
    /// P(yes) for each on the red-box crop.
    func runPerson(_ album: Album, refs r0: [[Float]], others: [[[Float]]], update: @escaping @Sendable (AlbumResult) -> Void) async throws -> PersonScored {
        var res = AlbumResult(name: album.name)
        let f = await index.allFaces()
        let pr = SearchEngine.faceProfile
        let refs = expandRefs(f.emb, refs: r0, accept: pr.expand, rounds: 3)
        let other = otherIdentities(groups: others, refs: refs, accept: pr.other)
        let ids = Array(Set(f.item)).sorted()
        let num = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
        let (score, best) = itemPersonScores(faces: f.emb, faceItem: f.item.map { num[$0]! }, nItems: ids.count, refs: refs, others: other)
        let items = await index.libraryItems(order: ids)
        let mask = scopeMask(items, album)
        let ident = ids.indices.filter { mask[$0] && score[$0] >= SearchEngine.personAccept }
        res.inScope = ident.count
        var p = [String: Double]()
        if let q0 = album.judgeQuestion {
            let q = q0.replacingOccurrences(of: #"(?i)\bthe person in (?:this|the) (?:photo|image|picture|video)\b|\bthe person\b(?! in the red box)"#,
                                            with: "the person in the red box", options: .regularExpression)
            // vote: a person's look uses its first model alone (engine.py; the mean broke heavier/fit, demo 10-06)
            let pj: any PhotoJudge = (judge as? EnsembleJudge)?.first ?? judge
            for k in ident {
                let id = ids[k], face = f.box[best[k]]
                let key = id + "|box\(best[k])|" + q
                if let c = await pj.cached(key) { p[id] = c }
                else if let img = await SearchEngine.redBoxCrop(id: id, face: face) {
                    let v = try await pj.pYes(img, question: q); await pj.remember(key, v); p[id] = v
                }
                res.judged += 1
                if res.judged % 20 == 0 { res.found = []; update(res) }
            }
        }
        return PersonScored(ids: ident.map { ids[$0] }, pYes: p)
    }

    /// The analysed image (same size the faces were found on), cropped around the person, red box on them.
    static func redBoxCrop(id: String, face: DetectedFace) async -> CIImage? {
        let cg: CGImage
        if let t = face.frameT {                 // a video: the frame the face was seen in
            guard let ci = await VideoFrames.frame(id, at: t), let c = CIContext().createCGImage(ci, from: ci.extent) else { return nil }
            cg = c
        } else {
            // full resolution only (an iCloud original is downloaded when allowed); never a smaller stand-in
            guard case .full(let ui) = await PhotoLibrary.read(id, side: 1280, purpose: .judge), let c = ui.cgImage else { return nil }
            cg = c
        }
        let sx = Double(cg.width) / face.imageW, sy = Double(cg.height) / face.imageH
        let fr = Rect(face.box[0] * sx, face.box[1] * sy, face.box[2] * sx, face.box[3] * sy)
        let (crop, box) = personCrop(face: fr, imageW: Double(cg.width), imageH: Double(cg.height))
        let cw = Int(crop.x2 - crop.x1), ch = Int(crop.y2 - crop.y1)
        guard cw > 0, ch > 0, let sub = cg.cropping(to: CGRect(x: crop.x1, y: crop.y1, width: Double(cw), height: Double(ch))),
              let ctx = CGContext(data: nil, width: cw, height: ch, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(sub, in: CGRect(x: 0, y: 0, width: cw, height: ch))
        ctx.setStrokeColor(red: 1, green: 0, blue: 0, alpha: 1)
        let lw = CGFloat(boxLineWidth(cropW: Double(cw), cropH: Double(ch))); ctx.setLineWidth(lw)
        // CGContext origin is bottom-left; the box is in top-left coordinates
        ctx.stroke(CGRect(x: box.x1 + Double(lw) / 2, y: Double(ch) - box.y2 + Double(lw) / 2,
                          width: box.x2 - box.x1 - Double(lw), height: box.y2 - box.y1 - Double(lw)))
        return ctx.makeImage().map { CIImage(cgImage: $0) }
    }
}

/// Within-person rank (fraction of the person's photos scoring at or below), like the server's `rel`.
// withinPersonRank: FindPicsCore (Pairing.swift), tested against the engine.
