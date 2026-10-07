// FACE UPGRADE: indexing reads every photo at indexReadSide (448, PhotoKit's local ~480 px rendition; 8ded039), which is
// fine for image vectors but not for faces (docs/PHONE_PARITY.md, "indexing from the ~480 px local renditions"):
// 40-65% of faces fall below the 40 px gate and same-face AuraFace cosine for small faces is unreliable (p5 0.19-0.33);
// detection itself holds (761 vs 777 faces). So photos whose faces were found on a small read are re-read at
// faceReadSide (downloads allowed: Wi-Fi in the foreground, the charger task), and until then a person search does not
// decide on their faces: they stay out of person albums and the album says how many are not checked yet.
// Platform-free decisions; the app (Index.swift, PersonSearch.swift, Moments.swift) does the reading.
import Foundation

/// Size faces must be read at to count as checked (the long side PhotoKit is asked for).
public let faceReadSide: Double = 1280

/// What every build before 8ded039 asked PhotoKit for when indexing (image vector and faces from one read).
public let legacyFaceReadSide: Double = 1280

/// Long side (px) of the read a photo's faces came from. `stored`: what the entry recorded (nil = written before entries
/// recorded it). Unrecorded: a smaller local stand-in (`lowRes`) was below the ask; imageVersion >= 2 was only ever
/// written by builds that read at indexReadSide (49b8906 and 8ded039 were installed together, MAC M11); older entries
/// were read at legacyFaceReadSide. Video frames are not affected by the index read size (VideoFrames reads 1280).
public func effectiveFaceSide(stored: Double?, isVideo: Bool, imageVersion: Int?, lowRes: Bool?) -> Double {
    if isVideo { return faceReadSide }
    if let s = stored { return s }
    if lowRes == true { return minStandInSide }
    if (imageVersion ?? 1) >= 2 { return indexReadSide }
    return legacyFaceReadSide
}

/// The faces of this entry were found on a read of at least faceReadSide ("checked").
public func facesFromFullRead(stored: Double?, isVideo: Bool, imageVersion: Int?, lowRes: Bool?) -> Bool {
    effectiveFaceSide(stored: stored, isVideo: isVideo, imageVersion: imageVersion, lowRes: lowRes) >= faceReadSide
}

/// The side to record for a read: the ask when PhotoKit gave the full size for it (or the whole, smaller original),
/// otherwise the long side actually received (a stand-in).
public func recordedFaceSide(full: Bool, requested: Double, gotLongSide: Double) -> Double { full ? requested : gotLongSide }

/// Photos the face upgrade re-reads, newest first (undated last; ties by id so the order is stable). Only photos that
/// HAVE faces from a small read: detection survives the small read, so a photo with no face there is not re-read.
public func faceUpgradeWork(_ items: [(id: String, taken: Double?, needsUpgrade: Bool)], skip: Set<String> = []) -> [String] {
    items.filter { $0.needsUpgrade && !skip.contains($0.id) }
        .sorted { a, b in
            switch (a.taken, b.taken) {
            case let (x?, y?): return x != y ? x > y : a.id < b.id
            case (_?, nil): return true
            case (nil, _?): return false
            case (nil, nil): return a.id < b.id
            }
        }
        .map(\.id)
}

/// The progress line while faces are re-read: k = photos with faces whose faces are checked, of all photos with faces.
public func faceUpgradeLine(checked: Int, total: Int) -> String { "Improving faces: \(checked) of \(total)" }

// MARK: - people searches

/// Who is in which photo, deciding only on checked faces. `members`: items in scope whose best checked face reaches
/// `profile.accept`. `unchecked`: items in scope whose faces all came from a small read: NOT decided either way (a small
/// face can look like a stranger or like the person), so they are neither in the album nor silently dropped: the album
/// note counts them. `best`: per item, the index into the ORIGINAL `faces` list of its best face (-1: none checked).
public struct PersonMatch: Equatable, Sendable {
    public let members: [Int]
    public let unchecked: [Int]
    public let scores: [Float]
    public let best: [Int]
}

/// expandRefs + otherIdentities + itemPersonScores (the server's people.py chain) on the checked faces only.
/// `expandRounds` 0 and no `others`: plain "does this person's face match" (the "with Jay" filter).
public func matchPerson<V: EmbeddingRows>(faces: V, faceItem: [Int], faceChecked: [Bool], nItems: Int, inScope: [Bool],
                        refs r0: [[Float]], others: [[[Float]]] = [], profile: FaceProfile = .shipped,
                        expandRounds: Int = 3) -> PersonMatch {
    let keep = (0..<faces.count).filter { faceChecked[$0] }
    let fe = faces.subset(keep), fi = keep.map { faceItem[$0] }
    let refs = expandRounds > 0 ? expandRefs(fe, refs: r0, accept: profile.expand, rounds: expandRounds) : r0
    let other = others.isEmpty ? [] : otherIdentities(groups: others, refs: refs, accept: profile.other)
    let (s, b) = itemPersonScores(faces: fe, faceItem: fi, nItems: nItems, refs: refs, others: other)
    let checkedItems = Set(fi), anyFace = Set(faceItem)
    let members = (0..<nItems).filter { inScope[$0] && checkedItems.contains($0) && s[$0] >= profile.accept }
    let unchecked = (0..<nItems).filter { inScope[$0] && anyFace.contains($0) && !checkedItems.contains($0) }
    return PersonMatch(members: members, unchecked: unchecked, scores: s, best: b.map { $0 < 0 ? -1 : keep[$0] })
}

/// The person album's note about photos whose faces are not checked yet (nil: none).
public func uncheckedFacesNote(_ n: Int) -> String? {
    guard n > 0 else { return nil }
    return "\(n) photo\(n == 1 ? "" : "s") not checked for faces yet (improving overnight)."
}

// MARK: - reads

/// How a read asks PhotoKit. Index reads at indexReadSide use resizeMode .fast (lets PhotoKit serve the local ~480 px
/// rendition) and take a big-enough local copy as final. A bigger index ask (the face upgrade at faceReadSide) is there
/// to get MORE than that copy: resizeMode .exact (the shape MAC M10 measured: ~0.8 s / photo at 1280 over Wi-Fi) and
/// the local copy is not final, so it downloads when allowed.
public struct ReadPolicy: Equatable, Sendable {
    public let fastResize: Bool
    public let localCopyIsFinal: Bool
}

public func readPolicy(_ purpose: FetchPurpose, side: Double) -> ReadPolicy {
    // .localOnly too: it is the index's FIRST pass (Index.update), where the ~170k iCloud-only photos must be taken
    // from their local ~480 px copies (MAC 10-07 14:57: leaving it out sent them all to a ~120 s/photo download pass)
    let index = (purpose == .indexForeground || purpose == .indexBackground || purpose == .localOnly) && side <= indexReadSide
    return ReadPolicy(fastResize: index, localCopyIsFinal: index)
}

/// Requests in flight at once during the face upgrade (MAC M10: 5 at a time ~3x faster than one by one).
public let faceUpgradeParallel = 5

/// Foreground face upgrade: photos per job, so new photos are not held up behind a pass of tens of thousands.
public let faceUpgradeForegroundChunk = 600

// MARK: - saved people after the upgrade

/// A face of a photo as the index holds it now (box in the analysed image's pixels).
public struct FaceCandidate: Equatable, Sendable {
    public let box: [Double]
    public let imageW: Double, imageH: Double
    public let embedding: [Float]
    public init(box: [Double], imageW: Double, imageH: Double, embedding: [Float]) {
        self.box = box; self.imageW = imageW; self.imageH = imageH; self.embedding = embedding
    }
}

/// A source photo's faces now: `side` = the read they came from (FaceSource / IndexEntry semantics).
public struct SourcePhotoFaces: Equatable, Sendable {
    public let side: Double
    public let faces: [FaceCandidate]
    public init(side: Double, faces: [FaceCandidate]) { self.side = side; self.faces = faces }
}

/// A saved person's references after the face upgrade re-read some of their source photos. `keep` false: too few were
/// found again (FindPicsCore.keepAfterRederive): ask "Is this you?" again (refsUpgradeReaskNote).
public struct RefsAfterUpgrade: Equatable, Sendable {
    public let refs: [[Float]]
    public let sources: [FaceSource]
    public let found: Int, total: Int, rederived: Int, lost: Int
    public let keep: Bool
}

/// Re-derive a saved person's references whose vectors came from a small read once their photo was re-read at
/// faceReadSide: the face with the same box (image-relative IoU >= 0.5, matchSourceFace) gives the new vector.
/// `now[k]`: the CURRENT faces of sources[k]'s photo (nil: not in the index, or a video frame, which the upgrade does
/// not touch). A reference stays as it is when its photo is not re-read yet, when it already came from a full-size
/// read (source side >= faceReadSide, e.g. a photo the person picked, read at 1280), or when its vector is still one
/// of the photo's faces exactly (nothing re-read it). One whose face is not found again is dropped.
/// Same rule as the face-model migration, over ALL references: kept + re-derived must be at least half
/// (keepAfterRederive); references whose photos were not re-read yet count as found (they are not lost, only not
/// improved yet, and the upgrade works in chunks). nil: nothing to change (or refs / sources do not pair up).
public func refsAfterUpgrade(refs: [[Float]], sources: [FaceSource], now: [SourcePhotoFaces?]) -> RefsAfterUpgrade? {
    guard refs.count == sources.count, now.count == sources.count, !refs.isEmpty else { return nil }
    var outR = [[Float]](), outS = [FaceSource](), rederived = 0, lost = 0, stamped = 0
    for k in refs.indices {
        let s = sources[k]
        guard s.t == nil, let n = now[k], n.side >= faceReadSide, (s.side ?? 0) < faceReadSide else {
            outR.append(refs[k]); outS.append(s); continue
        }
        if n.faces.contains(where: { $0.embedding == refs[k] }) {      // still this photo's face: already full-size
            outR.append(refs[k]); outS.append(s.withSide(n.side)); stamped += 1; continue
        }
        if let i = matchSourceFace(s, candidates: n.faces.map { (box: $0.box, imageW: $0.imageW, imageH: $0.imageH) }) {
            let f = n.faces[i]
            outR.append(f.embedding)
            outS.append(FaceSource(id: s.id, t: nil, box: f.box, imageW: f.imageW, imageH: f.imageH, side: n.side))
            rederived += 1
        } else { lost += 1 }
    }
    if rederived == 0 && lost == 0 && stamped == 0 { return nil }
    return RefsAfterUpgrade(refs: outR, sources: outS, found: outR.count, total: refs.count, rederived: rederived, lost: lost,
                            keep: keepAfterRederive(found: outR.count, total: refs.count))
}

/// The picker's note when a saved person must be confirmed again after the face upgrade.
public let refsUpgradeReaskNote = "find pics re-read your photos at full size to check faces and could not find your earlier "
    + "pick again. Please confirm once more."

/// "Who is X?" groups from reliable vectors: faceGroups on the checked faces (read at faceReadSide) only; all faces only
/// when the checked ones form no group at all (e.g. a fresh install before the upgrade ran). Face and rep indices are
/// into the full `faces` list either way.
public func faceGroupsPreferChecked<V: EmbeddingRows>(faces: V, faceItem: [Int], facePx: [Float], det: [Float], checked: [Bool],
                                    top: Int = 12, accept: Float = FaceProfile.shipped.group, minPx: Float = 40,
                                    minDet: Float = 0.7) -> [FaceGroup] {
    let sub = (0..<faces.count).filter { checked[$0] }
    if !sub.isEmpty {
        let g = faceGroups(faces: faces.subset(sub), faceItem: sub.map { faceItem[$0] }, facePx: sub.map { facePx[$0] },
                           det: sub.map { det[$0] }, top: top, accept: accept, minPx: minPx, minDet: minDet)
        if !g.isEmpty { return g.map { FaceGroup(faces: $0.faces.map { sub[$0] }, items: $0.items, rep: sub[$0.rep]) } }
    }
    return faceGroups(faces: faces, faceItem: faceItem, facePx: facePx, det: det, top: top, accept: accept, minPx: minPx, minDet: minDet)
}
