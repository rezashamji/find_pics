// Person albums ("me looking heavier"): identity ONLY from faces (the judge cannot recognise people); the judge rates
// the look on a crop around THAT person with a red box; photos are ranked within the person's own photos; two
// opposite looks of one person are split by FindPicsCore.splitPair (the rule measured on Reza's labeled photos).
import CoreGraphics
import CoreImage
@preconcurrency import FindPicsCore
import Foundation
import UIKit

/// `unchecked`: photos in scope whose faces came from a small read (not decided yet; FindPicsCore.matchPerson).
struct PersonScored { var ids: [String]; var pYes: [String: Double]; var unchecked = 0 }

extension SearchEngine {
    /// The face cuts belong to the face model (FindPicsCore.FaceProfile.shipped; buffalo_l was 0.40 / 0.55 / 0.40).
    static let faceProfile = FaceProfile.shipped
    static var personAccept: Float { faceProfile.accept }

    /// Identity-matched photos of the album's person (in scope) and, if the album has a look question, the judge's
    /// P(yes) for each on the red-box crop.
    func runPerson(_ album: Album, refs r0: [[Float]], others: [[[Float]]], negatives: [[Float]] = [],
                   update: @escaping @Sendable (AlbumResult) -> Void) async throws -> PersonScored {
        var res = AlbumResult(name: album.name)
        let f = await index.allFaces()
        let small = await index.uncheckedFaceItems()
        let ids = Array(Set(f.item)).sorted()
        let num = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
        let items = await index.libraryItems(order: ids)
        // A BARE "selfies" IS A PERSON SEARCH, which is why M50's detector never ran: the planner reads "selfies"
        // as being about the owner, so the search comes here and not through SearchEngine.run's plain-scope path
        // where I first wired it (MEASURED 10-12: "selfies" returned 269 - the camera-tagged photos alone - with
        // no SELFIEANSWER line anywhere in the log).
        // Here the camera filter is relaxed and replaced by the detector: of the owner's photos, keep the ones the
        // camera tag already proves are front-facing, plus the ones the detector scores above THIS library's
        // calibrated cut. ~170k photos whose EXIF we will never download become reachable, still with no judge
        // call. Scoping to the owner's own face set is what makes a 95%-precision cut worth having (M50 step 3).
        let bareSelfie = album.camera == "front" && album.judgeQuestion == nil
        var scopeAlbum = album
        if bareSelfie { scopeAlbum.camera = nil }
        var mask = scopeMask(items, scopeAlbum)
        if bareSelfie, let det = SearchEngine.selfieDetector, let t = await index.selfieThreshold(det) {
            let likely = Set(await index.selfieLikely(det, ids: ids, threshold: t))
            var tagged = 0, byScore = 0
            for i in items.indices where mask[i] {
                if items[i].isScreenshot || items[i].media == "video" { mask[i] = false; continue }
                if items[i].camera == "front" { tagged += 1 }
                else if likely.contains(ids[i]) { byScore += 1 }
                else { mask[i] = false }
            }
            Judge.logMem("SELFIEANSWER tagged=\(tagged) likely=\(byScore) t=\(String(format: "%.3f", t))")
            res.note = "\(tagged.formatted()) your phone recorded with its front camera, plus "
                     + "\(byScore.formatted()) more of you that look like selfies. Nothing was opened by the AI."
        } else if bareSelfie {
            mask = scopeMask(items, album)      // no calibration possible: fall back to the camera tag alone
        }
        // expandRefs + otherIdentities + itemPersonScores on CHECKED faces only: a face found on the ~480 px index read
        // decides nothing (in or out) until the face upgrade re-reads it; those photos are counted in the album note
        let m = matchPerson(faces: f.emb, faceItem: f.item.map { num[$0]! }, faceChecked: f.item.map { !small.contains($0) },
                            nItems: ids.count, inScope: mask, refs: r0, others: others, profile: SearchEngine.faceProfile,
                            negatives: negatives)
        let ident = m.members, best = m.best
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
        return PersonScored(ids: ident.map { ids[$0] }, pYes: p, unchecked: m.unchecked.count)
    }

    /// WHERE A PERSON SEARCH'S TIME ACTUALLY GOES (M64/M66). The flagship has now failed to finish twice on the
    /// device - 4 judge calls in 10 minutes on 10-12, and nothing at all overnight - and the suspicion is that the
    /// cost is not the judge but THIS: every judged photo is read at full resolution, which for an iCloud-only
    /// original means a download per photo. The rule is that the judge is never shown a smaller copy, so the fix
    /// cannot be to lower resolution; it has to be prefetching, ordering or skipping. That decision needs the
    /// number, so each read is timed and reported every 10 photos: read seconds vs the rest.
    /// A tiny actor rather than a lock: the counters are touched from an async context, where NSLock is unavailable.
    private actor ReadClock {
        private var secs = 0.0, n = 0
        func add(_ s: Double) -> (Int, Double) { secs += s; n += 1; return (n, secs) }
    }
    private static let readClock = ReadClock()

    /// The analysed image (same size the faces were found on), cropped around the person, red box on them.
    static func redBoxCrop(id: String, face: StoredFace) async -> CIImage? {
        let t0 = Date()
        let out = await redBoxCropInner(id: id, face: face)
        let (n, s) = await readClock.add(Date().timeIntervalSince(t0))
        if n % 10 == 0 {
            Judge.logMem(String(format: "PERSONREAD %d photos, %.1f s total, %.2f s each", n, s, s / Double(n)))
        }
        return out
    }

    private static func redBoxCropInner(id: String, face: StoredFace) async -> CIImage? {
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
