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
}

struct FrameUnit: Codable { let t: Double; let vector: [Float]; let faces: [DetectedFace] }

/// Indexing progress for the screen: `downloading` = the iCloud pass.
struct IndexProgress: Sendable { var done: Int; var total: Int; var downloading: Bool }

actor PhotoIndex {
    private(set) var entries: [String: IndexEntry] = [:]
    /// Assets not (fully) read yet and why (FindPicsCore.ReadOutcome); kept across launches so failed downloads wait
    /// for the charger task instead of stalling every launch.
    private(set) var notRead: [String: ReadOutcome] = [:]
    private var loaded = false
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
    func update(assets: [LibraryAsset], embedder: Embedder, faceEngine: FaceEngine?, purpose: FetchPurpose, retryFailed: Bool,
                progress: @Sendable (IndexProgress) -> Void) async {
        guard load() else { return }
        let byId = Dictionary(assets.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        let work = indexWork(library: assets.map(\.id), indexedLowRes: entries.mapValues { $0.lowRes ?? false }, notRead: notRead,
                             downloads: false, retryFailed: retryFailed)
        var done = 0
        for id in work.local {
            if Task.isCancelled { break }
            if let a = byId[id] { await index(a, embedder: embedder, faceEngine: faceEngine, purpose: .localOnly) }
            done += 1
            if done % 200 == 0 { save() }
            progress(IndexProgress(done: done, total: work.local.count, downloading: false))
        }
        save()
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
                       frames: frameUnits, camera: cam, isScreenshot: a.isScreenshot, lowRes: lowRes ? true : nil))
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
    /// Every face in the library: fingerprints, the photo each is in, size and detector confidence.
    func allFaces() -> (emb: [[Float]], item: [String], px: [Float], det: [Float], box: [DetectedFace]) {
        var e = [[Float]](), it = [String](), px = [Float](), det = [Float](), bx = [DetectedFace]()
        for (id, en) in entries {
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
