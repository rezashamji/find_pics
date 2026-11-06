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
            let pids = ident.map { ids[$0] }
            let faceOf = Dictionary(uniqueKeysWithValues: ident.map { (ids[$0], f.box[best[$0]]) })
            let boxOf = Dictionary(uniqueKeysWithValues: ident.map { (ids[$0], best[$0]) })
            // A PAIR MUST WALK IN LOCKSTEP. "heavier" and "fit" are two albums of the same person, and the split
            // between them is computed from the photos BOTH have scores for. Seeding the walk from the album name
            // gave each its own order, so after 480 calls each they had judged largely DIFFERENT photos and the
            // intersection - the only thing the split can use - was a fraction of either. Seeding from the PERSON
            // makes both walk the same photos in the same order, so every judged photo has both scores.
            // (The cluster's M72(3) goes further - one walk asking both questions on the same crop, halving the
            // reads. That is the speed version; this is the correctness one, and it comes first.)
            let seedSource = isLookQuestion(q0) ? (album.person ?? album.name) : album.name
            let seed = Int(truncatingIfNeeded: fnv1a64(seedSource))
            // ONE JUDGE CALL, SHARED BY BOTH PATHS. `for k in ident` judged every one of his 41,307 photos - 34
            // hours, no first answer, which is why the flagship never once reported (M67). Now both paths stop.
            func ask(_ id: String) async throws -> Double {
                let key = id + "|box\(boxOf[id] ?? 0)|" + q
                if let c = await pj.cached(key) { return c }
                guard let face = faceOf[id], let img = await SearchEngine.redBoxCrop(id: id, face: face) else { return 0 }
                let v = try await SearchEngine.timedPersonJudge { try await pj.pYes(img, question: q) }
                await pj.remember(key, v)
                return v
            }
            if isLookQuestion(q0) {
                // M72(2): a LOOK walks EVENTS, not photos - at most `lookCap` per occasion - so 480 calls spread
                // across his 5,764 events instead of sinking into one wedding. Stops at the first answer.
                let taken = await index.entries
                let ev = events(pids.map { taken[$0]?.taken })
                let eventOf = Dictionary(uniqueKeysWithValues: zip(pids, ev.map(String.init)))
                try await streamPersonLook(ids: pids, eventOf: eventOf, questions: 1, seed: seed, judge: { id, _ in
                    let v = try await ask(id); res.judged += 1; return v
                }, onUpdate: { u in
                    p = u.p[0]
                    res.found = []; update(res)
                    return !u.firstAnswer        // stop at the first answer; keep-looking is a later pass
                })
            } else {
                // M72(1): a FACT is an ordinary fast-mode search over his photos, ranked by the look score blended
                // with the learned head, and stopped by the same budget.
                let (units, unitItem, _) = await index.units(ids: pids)
                let looks = try (album.looks.isEmpty ? [q0] : album.looks).map { try embedder.vector(of: $0) }
                let avoid = try album.avoid.map { try embedder.vector(of: $0) }
                let cos = lookScores(units: units, unitItem: unitItem, nItems: pids.count, looks: looks, avoid: avoid)
                let hl = SearchEngine.learnedHead?.fold(question: try embedder.vector(of: q0))
                    .itemLogits(units: units, unitItem: unitItem, nItems: pids.count)
                let order = personFactOrder(cosine: cos, headLogits: hl).map { pids[$0] }
                for id in order.prefix(lookFirstCalls) {
                    if Task.isCancelled { break }
                    p[id] = try await ask(id)
                    res.judged += 1
                    if res.judged % 20 == 0 { res.found = []; update(res) }
                }
            }
            Judge.logMem("PERSONFIRST \(album.name): judged \(res.judged) of \(pids.count), "
                         + (isLookQuestion(q0) ? "look walk" : "fact order")
                         + " | PERSONSPLIT " + (await SearchEngine.personSplitLine()))
        }
        return PersonScored(ids: ident.map { ids[$0] }, pYes: p, unchecked: m.unchecked.count)
    }

    /// WHERE A PERSON SEARCH'S TIME ACTUALLY GOES (M64/M66). The flagship has now failed to finish twice on the
    /// device - 4 judge calls in 10 minutes on 10-12, and nothing at all overnight - and the suspicion is that the
    /// cost is not the judge but THIS: every judged photo is read at full resolution, which for an iCloud-only
    /// original means a download per photo. The rule is that the judge is never shown a smaller copy, so the fix
    /// cannot be to lower resolution; it has to be prefetching, ordering or skipping. That decision needs the
    /// number, so each read is timed and reported every 10 photos: read seconds vs the rest.
    /// THE SPLIT THE CLUSTER ASKED FOR (M100(4), repeated as M105(h)): read vs crop+box vs judge, per photo. It
    /// decides what the flagship fix has to be, and the three have different fixes - a read is prefetching or
    /// ordering, a crop is cheaper pixels, a judge call is nothing but fewer calls.
    /// AND IT CORRECTS A NUMBER I GAVE THE CLUSTER. I reported 7.95 s/photo for "person searches" from PERSONREAD
    /// (90 photos, 715.4 s), and PERSONREAD wrapped redBoxCrop only - the full-resolution read AND the crop AND the
    /// box, with NO judge call in it. So 7.95 s was image preparation, and the judge's ~3.17 s sits on top of it.
    /// A person leaf is therefore at least ~11.1 s, not 7.95, and the price in SearchEngine was too low.
    /// A tiny actor rather than a lock: the counters are touched from an async context, where NSLock is unavailable.
    private actor ReadClock {
        private var read = 0.0, crop = 0.0, judge = 0.0, n = 0, judged = 0
        func addPrep(read r: Double, crop c: Double) -> (Int, Double, Double) {
            read += r; crop += c; n += 1; return (n, read, crop)
        }
        func addJudge(_ s: Double) -> (Int, Double) { judge += s; judged += 1; return (judged, judge) }
        func line() -> String {
            String(format: "read %.2f s + crop/box %.2f s + judge %.2f s = %.2f s per photo "
                   + "(%d prepared, %d judged)",
                   n > 0 ? read / Double(n) : 0, n > 0 ? crop / Double(n) : 0,
                   judged > 0 ? judge / Double(judged) : 0,
                   (n > 0 ? (read + crop) / Double(n) : 0) + (judged > 0 ? judge / Double(judged) : 0),
                   n, judged)
        }
    }
    private static let readClock = ReadClock()

    /// The split as one line, for the end of a person album.
    static func personSplitLine() async -> String { await readClock.line() }

    /// Time one judge call of a person search into the same clock, so the three parts are reported together.
    static func timedPersonJudge<T>(_ work: () async throws -> T) async throws -> T {
        let t0 = Date()
        let out = try await work()
        _ = await readClock.addJudge(Date().timeIntervalSince(t0))
        return out
    }

    /// The analysed image (same size the faces were found on), cropped around the person, red box on them.
    static func redBoxCrop(id: String, face: StoredFace) async -> CIImage? {
        var readS = 0.0
        let t0 = Date()
        let out = await redBoxCropInner(id: id, face: face, readSecs: &readS)
        let prep = Date().timeIntervalSince(t0)
        let (n, r, c) = await readClock.addPrep(read: readS, crop: max(0, prep - readS))
        if n % 10 == 0 {
            Judge.logMem(String(format: "PERSONSPLIT %d photos: read %.2f s + crop/box %.2f s each | %@",
                                n, r / Double(n), c / Double(n), await readClock.line()))
        }
        return out
    }

    private static func redBoxCropInner(id: String, face: StoredFace, readSecs: inout Double) async -> CIImage? {
        let cg: CGImage
        let r0 = Date()
        if let t = face.frameT {                 // a video: the frame the face was seen in
            guard let ci = await VideoFrames.frame(id, at: t), let c = CIContext().createCGImage(ci, from: ci.extent) else { return nil }
            cg = c
        } else {
            // full resolution only (an iCloud original is downloaded when allowed); never a smaller stand-in
            guard case .full(let ui) = await PhotoLibrary.read(id, side: 1280, purpose: .judge), let c = ui.cgImage else { return nil }
            cg = c
        }
        readSecs = Date().timeIntervalSince(r0)
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
