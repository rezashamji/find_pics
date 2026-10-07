// The on-phone index: one image vector per photo (videos: one per sampled frame), stored in the app's own container
// (Application Support), never anywhere else. Kept current: AppModel runs `update` at launch, when the library changes
// while the app is open (PHPhotoLibraryChangeObserver), and in the charger/idle background task. What to read and
// what may download from iCloud is decided by FindPicsCore (indexWork, iCloudDownloadAllowed).
import CoreImage
import CoreLocation
@preconcurrency import FindPicsCore
import Foundation
import UIKit

struct IndexEntry: Codable {
    let id: String
    let isVideo: Bool
    let taken: Double?          // seconds since 1970
    let localMinutes: Int?      // wall clock where taken (from the phone's time zone at capture; nil if unknown)
    let lat: Double?, lon: Double?
    let vector: [Float]
    var faces: [DetectedFace]? = nil      // nil: indexed before faces existed
    var place: String? = nil               // offline place name from GPS ("Paris, ..., FR, France")
    var frames: [FrameUnit]? = nil         // videos: one unit per sampled frame (photos: nil, `vector` is the unit)
    var camera: String? = nil              // "front" | "back" from the photo's EXIF lens model; nil/"" unknown
    var isScreenshot: Bool? = nil
    var lowRes: Bool? = nil                // indexed from a smaller local copy; re-read once the original downloads
    /// The face model that made `faces` / frame faces (FaceProfile id). nil: indexed before entries recorded it, i.e.
    /// by buffalo_l (FindPicsCore.legacyFaceModel). Vectors of another model than FaceProfile.shipped are never used.
    var faceModel: String? = nil
    /// How `vector` / frame vectors were made (Embedder.imageVersion). nil = before 10-07: Core Image resize (no
    /// antialiasing) + int8 image weights, ~0.95-0.97 cosine from the server's; such entries are re-indexed (update).
    var imageVersion: Int? = nil

    var hasFaces: Bool { !(faces ?? []).isEmpty || (frames ?? []).contains { !$0.faces.isEmpty } }
    var facesCurrent: Bool { faceVectorsCurrent(model: faceModel, hasFaces: hasFaces) }
}

struct FrameUnit: Codable { let t: Double; let vector: [Float]; var faces: [DetectedFace] }

/// Indexing progress for the screen: `downloading` = the iCloud pass.
struct IndexProgress: Sendable { var done: Int; var total: Int; var downloading: Bool }

/// Re-embedding faces after a face-model change: `total` photos / videos with old-model faces when the pass began,
/// `done` re-embedded, `waiting` not readable now (e.g. in iCloud without Wi-Fi: retried on the charger).
/// `pass` numbers the passes (late progress reports of an older pass are ignored); `finished` = it ran to the end.
struct FaceReindexProgress: Sendable, Equatable {
    var done: Int; var total: Int; var waiting: Int; var running: Bool; var pass: Int; var finished = false
}

actor PhotoIndex {
    private(set) var entries: [String: IndexEntry] = [:]
    /// Assets not (fully) read yet and why (FindPicsCore.ReadOutcome); kept across launches so failed downloads wait
    /// for the charger task instead of stalling every launch.
    private(set) var notRead: [String: ReadOutcome] = [:]
    private var loaded = false
    private var facePass = 0
    private lazy var geocoder: Geocoder? = try? Geocoder()
    private let dir: URL = {
        let d = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }()
    private var file: URL { dir.appendingPathComponent("index.json") }
    private var notReadFile: URL { dir.appendingPathComponent("not_read.json") }

    /// Idempotent: the launch path and the background task may both ask. False when an index file exists but cannot
    /// be read (e.g. the phone has not been unlocked since it restarted): then nothing may be indexed or saved, or a
    /// partial index would overwrite the full one. Files use "until first unlock" protection so the charger task
    /// (which runs while the phone is locked) can read them.
    @discardableResult func load() -> Bool {
        if loaded { return true }
        let fm = FileManager.default
        var e: [IndexEntry] = [], n: [String: ReadOutcome] = [:]
        if fm.fileExists(atPath: file.path) {
            guard let d = try? Data(contentsOf: file), let x = try? JSONDecoder().decode([IndexEntry].self, from: d) else { return false }
            e = x
        }
        if fm.fileExists(atPath: notReadFile.path), let d = try? Data(contentsOf: notReadFile),
           let x = try? JSONDecoder().decode([String: ReadOutcome].self, from: d) { n = x }
        entries = Dictionary(e.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a }); notRead = n
        loaded = true
        return true
    }

    func save() {
        guard loaded else { return }
        if let d = try? JSONEncoder().encode(Array(entries.values)) { try? d.write(to: file, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication]) }
        if let d = try? JSONEncoder().encode(notRead) { try? d.write(to: notReadFile, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication]) }
    }

    func add(_ e: IndexEntry) { entries[e.id] = e }
    var count: Int { entries.count }
    var lowResCount: Int { entries.values.filter { $0.lowRes == true }.count }
    /// The "not searchable yet" line (nil when everything was read at full resolution).
    func summary() -> String? { notReadSummary(notRead, lowRes: lowResCount) }

    /// Photos / videos deleted from the library: out of the index (and of the not-read list).
    func remove(_ ids: [String]) {
        guard !ids.isEmpty, load() else { return }
        for id in ids { entries[id] = nil; notRead[id] = nil }
        save()
    }

    /// Index what `indexWork` says for these assets: new ones from what is on the phone first (fast, the library is
    /// searchable soon), then the iCloud downloads if `purpose` allows them on the current network (re-checked per
    /// asset: leaving Wi-Fi stops the pass). `retryFailed`: also retry failed downloads (the charger task).
    /// Saves every 200 assets so a stop (or the background task's expiry) loses little.
    /// Then (face-model change) the faces of photos indexed with another face model are re-embedded, from what is on the
    /// phone after the local pass, and with downloads after the iCloud pass (reembedStaleFaces).
    /// Returns the last face re-embedding pass's final state (nil: none ran).
    @discardableResult
    func update(assets: [LibraryAsset], embedder: Embedder, faceEngine: FaceEngine?, purpose: FetchPurpose, retryFailed: Bool,
                progress: @Sendable (IndexProgress) -> Void,
                faceProgress: @Sendable (FaceReindexProgress) -> Void = { _ in }) async -> FaceReindexProgress? {
        guard load() else { return nil }
        var facePassResult: FaceReindexProgress? = nil
        let byId = Dictionary(assets.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        let work = indexWork(library: assets.map(\.id), indexedLowRes: entries.mapValues { $0.lowRes ?? false }, notRead: notRead,
                             downloads: false, retryFailed: retryFailed)
        // entries whose image vectors an older image preparation made: re-read and re-index them after the new ones
        // (they stay searchable with their old vectors meanwhile; one that cannot be read now is retried next time)
        let local = work.local + assets.map(\.id).filter { id in
            entries[id].map { ($0.imageVersion ?? 1) != Embedder.imageVersion } ?? false
        }
        var done = 0
        for id in local {
            if Task.isCancelled { break }
            if let a = byId[id] { await index(a, embedder: embedder, faceEngine: faceEngine, purpose: .localOnly) }
            done += 1
            if done % 200 == 0 { save() }
            progress(IndexProgress(done: done, total: local.count, downloading: false))
        }
        save()
        if let fe = faceEngine, !Task.isCancelled {
            facePassResult = await reembedStaleFaces(faceEngine: fe, purpose: .localOnly, progress: faceProgress)
        }
        done = 0
        // again after the local pass: it just found which new assets are only in iCloud
        let download = iCloudDownloadAllowed(purpose, NetworkState.shared.path) ? indexWork(library: assets.map(\.id), indexedLowRes: entries.mapValues { $0.lowRes ?? false },
                                             notRead: notRead, downloads: true, retryFailed: retryFailed).download : []
        if !download.isEmpty { progress(IndexProgress(done: 0, total: download.count, downloading: true)) }
        for id in download {
            if Task.isCancelled || !iCloudDownloadAllowed(purpose, NetworkState.shared.path) { break }
            if let a = byId[id] { await index(a, embedder: embedder, faceEngine: faceEngine, purpose: purpose) }
            done += 1
            if done % 200 == 0 { save() }
            progress(IndexProgress(done: done, total: download.count, downloading: true))
        }
        save()
        if let fe = faceEngine, !Task.isCancelled, iCloudDownloadAllowed(purpose, NetworkState.shared.path), staleFaceCount() > 0 {
            facePassResult = await reembedStaleFaces(faceEngine: fe, purpose: purpose, progress: faceProgress) ?? facePassResult
        }
        return facePassResult
    }

    /// Photos / videos whose faces came from another face model than the shipped one.
    func staleFaceCount() -> Int { entries.values.filter { !$0.facesCurrent }.count }

    /// Face-model change: the faces each stale entry already has are re-embedded with `faceEngine` (same boxes; only
    /// photos / videos that had faces are re-read). Image vectors, places, dates are kept. All-or-nothing per entry:
    /// an entry is stamped with the new model only when every frame was re-read, so one entry never holds two models'
    /// vectors; one that cannot be read now stays stale (left out of face matching) and is retried on the next pass.
    @discardableResult
    func reembedStaleFaces(faceEngine fe: FaceEngine, purpose: FetchPurpose,
                           progress: @Sendable (FaceReindexProgress) -> Void) async -> FaceReindexProgress? {
        guard load() else { return nil }
        let stale = entries.values.filter { !$0.facesCurrent }.sorted { ($0.taken ?? 0) > ($1.taken ?? 0) }.map(\.id)  // newest first
        guard !stale.isEmpty else { return nil }
        facePass += 1
        var p = FaceReindexProgress(done: 0, total: stale.count, waiting: 0, running: true, pass: facePass)
        progress(p)
        for id in stale {
            if Task.isCancelled { break }
            guard let e = entries[id] else { continue }
            // (entries[id] checked again after the await: never bring back a photo deleted meanwhile)
            if let new = await reembedded(e, faceEngine: fe, purpose: purpose), entries[id] != nil { entries[id] = new; p.done += 1 }
            else { p.waiting += 1 }
            if (p.done + p.waiting) % 200 == 0 { save() }
            if (p.done + p.waiting) % 20 == 0 { progress(p) }
        }
        save()
        p.running = false; p.finished = !Task.isCancelled
        progress(p)
        return p
    }

    /// The entry with its faces re-embedded by `fe`, or nil when the photo / a frame cannot be read now.
    private func reembedded(_ e: IndexEntry, faceEngine fe: FaceEngine, purpose: FetchPurpose) async -> IndexEntry? {
        var out = e
        if let units = e.frames {
            let ts = units.filter { !$0.faces.isEmpty }.map(\.t)
            guard let frames = await VideoFrames.frames(e.id, at: ts, purpose: purpose) else { return nil }
            var newUnits = units
            for k in newUnits.indices where !newUnits[k].faces.isEmpty {
                guard let ci = frames[newUnits[k].t], let cg = CIContext().createCGImage(ci, from: ci.extent),
                      let fs = try? fe.reembed(newUnits[k].faces, in: cg) else { return nil }
                newUnits[k].faces = fs
            }
            out.frames = newUnits
        }
        if let fs = e.faces, !fs.isEmpty {
            let cg: CGImage
            switch await PhotoLibrary.read(e.id, side: 1280, purpose: purpose) {
            case .full(let ui): guard let c = ui.cgImage else { return nil }; cg = c
            case .notFull(let standIn, _):
                // indexed from a stand-in: its faces were found on a smaller local copy, which is still fine to re-read
                guard e.lowRes == true, let c = standIn?.cgImage else { return nil }
                cg = c
            }
            guard let new = try? fe.reembed(fs, in: cg) else { return nil }
            out.faces = new
        }
        out.faceModel = fe.profile.id
        return out
    }

    /// Read one asset and index it (full resolution, or a stand-in marked lowRes), or record why not.
    private func index(_ a: LibraryAsset, embedder: Embedder, faceEngine: FaceEngine?, purpose: FetchPurpose) async {
        let net = purpose != .localOnly
        var frameUnits: [FrameUnit]? = nil
        var vector: [Float]? = nil, faces: [DetectedFace]? = nil, lowRes = false
        if a.isVideo {                          // several moments of the video, each with its vector and faces
            let (frames, why) = await VideoFrames.sample(a.id, purpose: purpose)
            var units = [FrameUnit]()
            for (t, im) in frames {
                guard let v = try? embedder.vector(of: im) else { continue }
                var fs = [DetectedFace]()
                if let fe = faceEngine, let cg = CIContext().createCGImage(im, from: im.extent) { fs = (try? fe.faces(in: cg)) ?? [] }
                units.append(FrameUnit(t: t, vector: v, faces: fs))
            }
            if units.isEmpty { record(a.id, why ?? .unreadable); return }
            frameUnits = units; vector = units[0].vector
        } else {
            let img: UIImageBox
            switch await PhotoLibrary.read(a.id, side: 1280, purpose: purpose) {
            case .full(let ui): img = UIImageBox(ui)
            case .notFull(let standIn, let why):
                // a smaller local copy keeps the photo searchable until its original downloads (never shown to the judge)
                guard let s = standIn, max(PhotoLibrary.pixelSize(s).0, PhotoLibrary.pixelSize(s).1) >= minStandInSide,
                      entries[a.id] == nil else { record(a.id, why); return }
                img = UIImageBox(s); lowRes = true; notRead[a.id] = why
            }
            guard let cg = img.image.cgImage else { record(a.id, .unreadable); return }
            let ci = CIImage(cgImage: cg)
            guard let v = try? embedder.vector(of: ci) else { record(a.id, .unreadable); return }
            vector = v
            if let fe = faceEngine { faces = try? fe.faces(in: cg) }
        }
        guard let v = vector else { return }
        let cal = Calendar.current
        let lm = a.created.map { cal.component(.hour, from: $0) * 60 + cal.component(.minute, from: $0) }
        let cam = a.isVideo ? nil : await PhotoLibrary.camera(a.id, network: net && !lowRes)
        add(IndexEntry(id: a.id, isVideo: a.isVideo, taken: a.created?.timeIntervalSince1970, localMinutes: lm,
                       lat: a.location?.coordinate.latitude, lon: a.location?.coordinate.longitude, vector: v, faces: faces,
                       place: a.location.flatMap { geocoder?.placeText(lat: $0.coordinate.latitude, lon: $0.coordinate.longitude) },
                       frames: frameUnits, camera: cam, isScreenshot: a.isScreenshot, lowRes: lowRes ? true : nil,
                       faceModel: faceEngine?.profile.id, imageVersion: Embedder.imageVersion))
        if !lowRes { notRead[a.id] = nil }
    }

    private func record(_ id: String, _ why: ReadOutcome) {
        guard PhotoLibrary.asset(id) != nil else { notRead[id] = nil; return }   // deleted meanwhile
        // a stand-in already indexed stays; only its not-read reason changes
        notRead[id] = why
    }

    // FindPicsCore.LibraryItem spelled out: DeveloperToolsSupport (in scope via the app's SwiftUI/UIKit imports)
    // also exports a LibraryItem, so the bare name is ambiguous here.
    func libraryItems(order: [String]) -> [FindPicsCore.LibraryItem] {
        order.compactMap { entries[$0] }.map {
            FindPicsCore.LibraryItem(id: $0.id, media: $0.isVideo ? "video" : "photo", taken: $0.taken, localMinutes: $0.localMinutes, place: $0.place,
                                     camera: $0.camera ?? "", isScreenshot: $0.isScreenshot ?? false)
        }
    }
}

/// UIImage handed from PhotoKit to the index actor once (immutable).
struct UIImageBox: @unchecked Sendable { let image: UIImage; init(_ i: UIImage) { image = i } }

extension PhotoIndex {
    /// Every face in the library made by face model `model` (default: the shipped one; vectors of two models are never
    /// returned together): fingerprints, the photo each is in, size and detector confidence.
    func allFaces(model: String = FaceProfile.shipped.id) -> (emb: [[Float]], item: [String], px: [Float], det: [Float], box: [DetectedFace]) {
        var e = [[Float]](), it = [String](), px = [Float](), det = [Float](), bx = [DetectedFace]()
        for (id, en) in entries where en.hasFaces && (en.faceModel ?? legacyFaceModel) == model {
            let fs = en.frames.map { $0.flatMap { u in u.faces.map { f in f.withFrame(u.t) } } } ?? en.faces ?? []
            for f in fs { e.append(f.embedding); it.append(id); px.append(Float(f.px)); det.append(f.confidence); bx.append(f) }
        }
        return (e, it, px, det, bx)
    }
}

extension PhotoIndex {
    /// Image-vector units: one per photo, one per sampled video frame (unitItem = position in `ids`, unitT = frame time).
    func units(ids: [String]) -> (vectors: [[Float]], unitItem: [Int], unitT: [Double?]) {
        var v = [[Float]](), ui = [Int](), ut = [Double?]()
        for (k, id) in ids.enumerated() {
            guard let e = entries[id] else { continue }
            if let fr = e.frames { for u in fr { v.append(u.vector); ui.append(k); ut.append(u.t) } }
            else { v.append(e.vector); ui.append(k); ut.append(nil) }
        }
        return (v, ui, ut)
    }
}
