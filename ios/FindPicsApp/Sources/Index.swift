// The on-phone index: one image vector per photo (videos: their poster frame for now), stored in the app's own
// container (Application Support), never anywhere else. Built once in the background, then kept up to date.
import CoreImage
import CoreLocation
import FindPicsCore
import Foundation

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
}

struct FrameUnit: Codable { let t: Double; let vector: [Float]; let faces: [DetectedFace] }

actor PhotoIndex {
    private(set) var entries: [String: IndexEntry] = [:]
    private lazy var geocoder: Geocoder? = try? Geocoder()
    private let file: URL = {
        let d = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d.appendingPathComponent("index.json")
    }()

    func load() {
        if let d = try? Data(contentsOf: file), let e = try? JSONDecoder().decode([IndexEntry].self, from: d) {
            entries = Dictionary(uniqueKeysWithValues: e.map { ($0.id, $0) })
        }
    }

    func save() {
        if let d = try? JSONEncoder().encode(Array(entries.values)) { try? d.write(to: file, options: .completeFileProtection) }
    }

    func add(_ e: IndexEntry) { entries[e.id] = e }
    /// Photos the phone could not give us an image for while indexing (e.g. originals only in iCloud with "Optimize
    /// iPhone Storage"): counted so the first device test shows how many a search would miss.
    private(set) var unreadable = 0
    var count: Int { entries.count }

    /// Index every asset not yet indexed. `progress(done, total)`. Saves every 200 photos so a stop loses little.
    func build(assets: [LibraryAsset], embedder: Embedder, faceEngine: FaceEngine?, progress: @Sendable (Int, Int) -> Void) async {
        let todo = assets.filter { entries[$0.id] == nil }
        var done = 0
        for a in todo {
            if Task.isCancelled { break }
            var frameUnits: [FrameUnit]? = nil
            if a.isVideo {                          // several moments of the video, each with its vector and faces
                var units = [FrameUnit]()
                for (t, im) in await VideoFrames.sample(a.id) {
                    guard let v = try? embedder.vector(of: im) else { continue }
                    var fs = [DetectedFace]()
                    if let fe = faceEngine, let cg = CIContext().createCGImage(im, from: im.extent) { fs = (try? fe.faces(in: cg)) ?? [] }
                    units.append(FrameUnit(t: t, vector: v, faces: fs))
                }
                if !units.isEmpty { frameUnits = units }
            }
            if let ci = await PhotoLibrary.ciImage(a.id, side: 1280), let v = frameUnits?.first?.vector ?? (try? embedder.vector(of: ci)) {
                var faces: [DetectedFace]? = nil
                if frameUnits == nil, let fe = faceEngine, let cg = CIContext().createCGImage(ci, from: ci.extent) { faces = try? fe.faces(in: cg) }
                let cal = Calendar.current
                let lm = a.created.map { cal.component(.hour, from: $0) * 60 + cal.component(.minute, from: $0) }
                add(IndexEntry(id: a.id, isVideo: a.isVideo, taken: a.created?.timeIntervalSince1970, localMinutes: lm,
                               lat: a.location?.coordinate.latitude, lon: a.location?.coordinate.longitude, vector: v, faces: faces,
                               place: a.location.flatMap { geocoder?.placeText(lat: $0.coordinate.latitude, lon: $0.coordinate.longitude) },
                               frames: frameUnits, camera: a.isVideo ? nil : await PhotoLibrary.camera(a.id)))
            }
            else { unreadable += 1 }
            done += 1
            if done % 200 == 0 { save() }
            progress(done, todo.count)
        }
        save()
    }

    func libraryItems(order: [String]) -> [LibraryItem] {
        order.compactMap { entries[$0] }.map {
            LibraryItem(id: $0.id, media: $0.isVideo ? "video" : "photo", taken: $0.taken, localMinutes: $0.localMinutes, place: $0.place,
                        camera: $0.camera ?? "")
        }
    }
}

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
