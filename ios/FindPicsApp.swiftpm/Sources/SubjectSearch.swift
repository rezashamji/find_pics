// "This specific dog / thing / place" from 1-3 example photos (port of the server's subject path, converse.py):
// 1. rank: image vectors with neighbour smoothing (FindPicsCore.subjectScores is the tested reference; here the same
//    math with Accelerate so a 30k-photo library takes seconds, not minutes);
// 2. veto: the judge sees [example | candidate] side by side and is asked whether it is the SAME individual; a photo
//    stays unless the judge is fairly sure it is not (P >= 0.2 keeps it; server measurement: kept 140/147 true photos).
// Identity of a pet / object is NOT certified: no completeness bound is shown, best matches first.
// UNTESTED on a device (written on Linux); expect compile fixes at the first Xcode build.
// Subject store: the example photos picked for a named pet / thing ("Show me Max"), so the next request about it
// does not ask again (FindPicsCore.namedSubjects / resolveSubject decide when to ask).
import Accelerate
import CoreImage
@preconcurrency import FindPicsCore
import Foundation

extension SearchEngine {
    static let subjectQuestion = "The left panel shows %@, one specific %@. Compare individual features: colour pattern and "
        + "markings, shape, scars, distinctive details. Is the %@ in the right panel the SAME individual %@, not just a "
        + "similar-looking one? If you are not sure, answer no."
    static let subjectKeep = 0.2
    static let subjectCandidates = 300

    /// `album`: the rest of the plan (dates, place, media, selfie camera) scopes the candidates; its own condition
    /// (judgeQuestion, "at the beach"), filterQuestion and excludeQuestion are then asked about the WHOLE photo of each
    /// identity match (converse._album_stream's cond_q / filter / exclude on subject albums). `restrictTo`: a moment's window.
    func runSubject(examples: [CIImage], name: String, kind: String, album: Album, restrictTo: Set<String>? = nil,
                    update: @escaping @Sendable (AlbumResult) -> Void) async throws -> AlbumResult {
        var res = AlbumResult(name: album.name)
        let entries = await index.entries
        let ids = Array(entries.keys).filter { restrictTo?.contains($0) ?? true }.sorted()
        let items = await index.libraryItems(order: ids)
        let mask = scopeMask(items, album)
        let u = await index.units(ids: ids)
        let refs = try examples.map { try embedder.vector(of: $0) }
        let score = SearchEngine.smoothedSubjectScores(units: u.vectors, unitItem: u.unitItem, nItems: ids.count, refs: refs, inScope: mask)
        let order = ids.indices.filter { mask[$0] }.sorted { score[$0] > score[$1] }.prefix(SearchEngine.subjectCandidates)
        res.inScope = order.count
        let q = String(format: SearchEngine.subjectQuestion, name, kind, kind, kind)
        guard let ref = examples.first else { res.done = true; return res }
        var kept: [String] = [], missing = 0
        for k in order {
            if Task.isCancelled { break }
            let id = ids[k]
            let key = id + "|subject|" + q
            var p = await judge.cached(key)
            if p == nil, let img = await PhotoLibrary.ciImage(id, side: 1280) {
                let v = try await judge.pYes(SearchEngine.sideBySide(ref, img), question: q)
                await judge.remember(key, v); p = v
            }
            res.judged += 1
            guard let pv = p else { missing += 1; continue }        // full-resolution photo not available (iCloud)
            var keep = pv >= SearchEngine.subjectKeep
            if keep, let cq = album.judgeQuestion { keep = (try await wholePhotoP(id, cq) ?? 0) >= SearchEngine.accept }
            if keep, let fq = album.filterQuestion { keep = (try await wholePhotoP(id, fq) ?? 0) >= SearchEngine.accept }
            if keep, let ex = album.excludeQuestion { keep = (try await wholePhotoP(id, ex) ?? 0) < SearchEngine.accept }
            if keep { kept.append(id) }
            if res.judged % 20 == 0 { res.found = kept; update(res) }
        }
        res.found = kept
        res.note = "Best matches first. A pet's or object's identity is checked side by side but not guaranteed: "
            + "look through the list."
        if album.judgeQuestion != nil { res.note += " Only photos that also pass \"\(album.judgeQuestion!)\"." }
        if missing > 0 { res.note += " \(missing) candidate photo(s) could not be checked: their originals are in iCloud and could not be downloaded now." }
        res.done = true
        return res
    }

    /// P(yes) on the whole photo at the judge's size (896 px, like run()); nil if the full-resolution image is not
    /// available (an iCloud original that could not be downloaded): such a photo is never judged on a smaller copy.
    func wholePhotoP(_ id: String, _ question: String) async throws -> Double? {
        if let c = await judge.cached(id + "|" + question) { return c }
        guard let img = await PhotoLibrary.ciImage(id, side: 896) else { return nil }
        let v = try await judge.pYes(img, question: question); await judge.remember(id + "|" + question, v); return v
    }

    /// [reference | candidate] on white, reference <= 640 px tall, candidate <= 1024 x 640 (engine.side_by_side).
    static func sideBySide(_ ref: CIImage, _ im: CIImage, H: CGFloat = 640) -> CIImage {
        func fit(_ x: CIImage, _ w: CGFloat, _ h: CGFloat) -> CIImage {
            let s = min(1, min(w / x.extent.width, h / x.extent.height))
            let y = x.transformed(by: CGAffineTransform(scaleX: s, y: s))
            return y.transformed(by: CGAffineTransform(translationX: -y.extent.minX, y: -y.extent.minY))
        }
        let r = fit(ref, H, H), i = fit(im, H * 1.6, H)
        let W = r.extent.width + 24 + i.extent.width, Ht = max(r.extent.height, i.extent.height)
        let bg = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: W, height: Ht))
        // top-aligned like PIL's paste at (x, 0): CoreImage's origin is bottom-left
        let rr = r.transformed(by: CGAffineTransform(translationX: 0, y: Ht - r.extent.height))
        let ii = i.transformed(by: CGAffineTransform(translationX: r.extent.width + 24, y: Ht - i.extent.height))
        return ii.composited(over: rr.composited(over: bg))
    }

    /// FindPicsCore.subjectScores' neighbour smoothing (k = 2, mean over references, best unit per item) restricted to
    /// the subjectCandidateUnits units closest to the smoothed references among the in-scope photos
    /// (FindPicsCore.subjectScoresCandidates: O(n + K^2) instead of library x library; quality: JOURNAL 10-07), over the
    /// index store's mapped rows with Accelerate's sgemm per block.
    static func smoothedSubjectScores<V: EmbeddingRows>(units: V, unitItem: [Int], nItems: Int, refs: [[Float]], k: Int = 2,
                                                        inScope: [Bool]? = nil) -> [Float] {
        subjectScoresCandidates(units: units, unitItem: unitItem, nItems: nItems, refs: refs, k: k, inScope: inScope)
    }
}

/// Example photos picked for a pet / thing, by FindPicsCore.SavedSubject key; in the app's own container only.
actor SubjectStore {
    private(set) var saved: [SavedSubject] = []
    private let file: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("subjects.json")
    private var loaded = false
    /// False when subjects.json exists but cannot be read yet (locked since restart): then nothing is saved over it.
    @discardableResult func load() -> Bool {
        if loaded { return true }
        if FileManager.default.fileExists(atPath: file.path) {
            guard let d = try? Data(contentsOf: file), let s = try? JSONDecoder().decode([SavedSubject].self, from: d) else { return false }
            saved = s
        }
        loaded = true
        return true
    }
    func add(_ s: SavedSubject) {
        guard load() else { return }
        saved.removeAll { $0.key == s.key }; saved.append(s)
        if let d = try? JSONEncoder().encode(saved) { try? d.write(to: file, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication]) }
    }
}
