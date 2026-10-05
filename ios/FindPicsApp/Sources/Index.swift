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
}

actor PhotoIndex {
    private(set) var entries: [String: IndexEntry] = [:]
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
    var count: Int { entries.count }

    /// Index every asset not yet indexed. `progress(done, total)`. Saves every 200 photos so a stop loses little.
    func build(assets: [LibraryAsset], embedder: Embedder, faceEngine: FaceEngine?, progress: @Sendable (Int, Int) -> Void) async {
        let todo = assets.filter { entries[$0.id] == nil }
        var done = 0
        for a in todo {
            if Task.isCancelled { break }
            if let ci = await PhotoLibrary.ciImage(a.id, side: 1280), let v = try? embedder.vector(of: ci) {
                var faces: [DetectedFace]? = nil
                if let fe = faceEngine, let cg = CIContext().createCGImage(ci, from: ci.extent) { faces = try? fe.faces(in: cg) }
                let cal = Calendar.current
                let lm = a.created.map { cal.component(.hour, from: $0) * 60 + cal.component(.minute, from: $0) }
                add(IndexEntry(id: a.id, isVideo: a.isVideo, taken: a.created?.timeIntervalSince1970, localMinutes: lm,
                               lat: a.location?.coordinate.latitude, lon: a.location?.coordinate.longitude, vector: v, faces: faces))
            }
            done += 1
            if done % 200 == 0 { save() }
            progress(done, todo.count)
        }
        save()
    }

    func libraryItems(order: [String]) -> [LibraryItem] {
        order.compactMap { entries[$0] }.map {
            LibraryItem(id: $0.id, media: $0.isVideo ? "video" : "photo", taken: $0.taken, localMinutes: $0.localMinutes, place: nil)
        }
    }
}

extension PhotoIndex {
    /// Every face in the library: fingerprints, the photo each is in, size and detector confidence.
    func allFaces() -> (emb: [[Float]], item: [String], px: [Float], det: [Float], box: [DetectedFace]) {
        var e = [[Float]](), it = [String](), px = [Float](), det = [Float](), bx = [DetectedFace]()
        for (id, en) in entries { for f in en.faces ?? [] { e.append(f.embedding); it.append(id); px.append(Float(f.px)); det.append(f.confidence); bx.append(f) } }
        return (e, it, px, det, bx)
    }
}
