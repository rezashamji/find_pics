// The on-phone index: one image vector per photo (videos: one per sampled frame), stored in the app's own container
// (Application Support), never anywhere else. Kept current: AppModel runs `update` at launch, when the library changes
// while the app is open (PHPhotoLibraryChangeObserver), and in the charger/idle background task. What to read and
// what may download from iCloud is decided by FindPicsCore (indexWork, iCloudDownloadAllowed).
import CoreImage
import CoreLocation
@preconcurrency import FindPicsCore
import Foundation
import os
import UIKit

/// One indexed photo / video: metadata only (dates, place, camera, flags, versions, face boxes). Its image vectors and
/// face fingerprints stay in the store's memory-mapped files (FindPicsCore.IndexStore, FindPicsCore.IndexRecord).
typealias IndexEntry = IndexRecord

/// Indexing progress for the screen: `downloading` = the iCloud pass; `faces` = the face upgrade (done = photos with
/// checked faces, total = photos with faces).
struct IndexProgress: Sendable { var done: Int; var total: Int; var downloading: Bool; var faces = false }

/// Re-embedding faces after a face-model change: `total` photos / videos with old-model faces when the pass began,
/// `done` re-embedded, `waiting` not readable now (e.g. in iCloud without Wi-Fi: retried on the charger).
/// `pass` numbers the passes (late progress reports of an older pass are ignored); `finished` = it ran to the end.
struct FaceReindexProgress: Sendable, Equatable {
    var done: Int; var total: Int; var waiting: Int; var running: Bool; var pass: Int; var finished = false
}

actor PhotoIndex {
    /// The binary store (FindPicsCore.IndexStore): metadata in RAM, vectors mapped from disk, saves append-only.
    private var store: IndexStore?
    /// Every indexed photo / video (metadata; no vectors). Empty until load().
    var entries: [String: IndexEntry] { store?.records ?? [:] }
    /// Assets not (fully) read yet and why (FindPicsCore.ReadOutcome); kept across launches so failed downloads wait
    /// for the charger task instead of stalling every launch.
    var notRead: [String: ReadOutcome] { store?.notRead ?? [:] }
    private var facePass = 0
    /// Face upgrade: photos that could not be read at faceReadSide this launch (retried by the charger task).
    private var upgradeSkipped = Set<String>()
    private lazy var geocoder: Geocoder? = try? Geocoder()
    /// Developer line (-localSizes, MAC_INBOX M15): how the store opened, the one-time conversion, the last save error.
    private var storeNote = ""
    private let dir: URL = {
        let d = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }()
    /// The old JSON index of builds before 10-07: converted once into the store, then deleted (derived data).
    private var file: URL { dir.appendingPathComponent("index.json") }
    private var notReadFile: URL { dir.appendingPathComponent("not_read.json") }
    private var storeDir: URL { dir.appendingPathComponent("index_store") }
    /// Image vectors: PE-Core-L-14-336 (1024); face fingerprints: AuraFace / buffalo_l (512). Both stored as Float16:
    /// cosine changes <= 3.2e-5 (image, 60k query x photo pairs) and <= 1.1e-4 (faces, 73M pairs), JOURNAL 10-07.
    static let storeConfig = IndexStore.Config(imageDim: 1024, faceDim: 512)

    /// Idempotent: the launch path and the background task may both ask. False when the index cannot be read now (e.g.
    /// the phone has not been unlocked since it restarted): then nothing may be indexed or saved. The first launch of
    /// this build converts the old index.json once (`progress` 0...1). A damaged store is derived data: it is removed
    /// and the library indexed again.
    @discardableResult func load(progress: @Sendable (Double) -> Void = { _ in }) -> Bool {
        if store != nil { return true }
        let t0 = Date()
        func open() throws {
            let (s, rep) = try IndexStore.openMigrating(dir: storeDir, legacyIndex: file, legacyNotRead: notReadFile,
                                                        config: PhotoIndex.storeConfig, progress: progress)
            store = s
            storeNote = String(format: "index store: %d entries opened in %.2f s", s.records.count, s.openStats.seconds)
            if let r = rep {
                storeNote += String(format: "; converted from index.json (%.0f MB) in %.1f s: %d entries, %d duplicates, %d unreadable%@",
                                    Double(r.jsonBytes) / 1e6, r.seconds, r.entries, r.duplicates, r.skipped, r.truncated ? ", file was cut off" : "")
            }
            if s.openStats.logBytesCut > 0 || s.openStats.imageRowsCut > 0 {
                storeNote += "; recovered from an interrupted save (\(s.openStats.imageRowsCut) unsaved photos dropped)"
            }
        }
        do { try open() }
        catch IndexStoreError.corrupt(let why) { storeNote = "index store damaged (\(why)): rebuilt"; return reset(after: t0, open) }
        catch IndexStoreError.incompatible(let why) { storeNote = "index store format changed (\(why)): rebuilt"; return reset(after: t0, open) }
        catch { storeNote = "index store not readable now: \(error)"; return false }
        storeNote += String(format: " (load %.2f s total)", Date().timeIntervalSince(t0))
        return true
    }

    private func reset(after t0: Date, _ open: () throws -> Void) -> Bool {
        let note = storeNote
        try? FileManager.default.removeItem(at: storeDir)
        do { try open() } catch { storeNote = note + "; reopen failed: \(error)"; return false }
        storeNote = note + "; " + storeNote
        return true
    }

    /// Makes the changes so far durable (append-only: new vectors + a journal group; FindPicsCore.IndexStore.commit).
    func save() {
        guard let s = store else { return }
        do { try s.commit() } catch { storeNote = "last save failed: \(error)" }
    }

    /// The developer line for -localSizes.
    func storeLine() -> String {
        guard let s = store else { return storeNote }
        return storeNote + String(format: "; on disk %.0f MB (%d image rows, %d face rows, %d dead)", Double(s.diskBytes) / 1e6,
                                  s.imageRows, s.faceRowCount, s.deadImageRows + s.deadFaceRows)
    }

    func add(_ e: FullIndexEntry) {
        do { try store?.put(e) } catch { storeNote = "could not add \(e.id): \(error)"; record(e.id, .unreadable) }
    }
    private func setNotRead(_ id: String, _ why: ReadOutcome?) { store?.setNotRead(id, why) }
    var count: Int { entries.count }
    var lowResCount: Int { entries.values.filter { $0.lowRes == true }.count }
    /// The "not searchable yet" line (nil when everything was read at full resolution).
    func summary() -> String? { notReadSummary(notRead, lowRes: lowResCount) }

    /// Photos / videos deleted from the library: out of the index (and of the not-read list).
    func remove(_ ids: [String]) {
        guard !ids.isEmpty, load() else { return }
        for id in ids { store?.remove(id); setNotRead(id, nil) }
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
            if let a = byId[id] {
                let item = await PhotoIndex.prepare(a, embedder: embedder, faceEngine: faceEngine, purpose: .localOnly,
                                                    alreadyIndexed: entries[id] != nil, kept: keptFaces(a, faceEngine))
                commit(a, item, faceEngine: faceEngine)
            }
            done += 1
            if done % 200 == 0 { save() }
            progress(IndexProgress(done: done, total: local.count, downloading: false))
        }
        save()
        if let fe = faceEngine, !Task.isCancelled {
            facePassResult = await reembedStaleFaces(faceEngine: fe, purpose: .localOnly, progress: faceProgress)
        }
        done = 0
        // again after the local pass: it just found which new assets are only in iCloud. Photos first, videos last
        // (FindPicsCore.downloadOrder); downloadParallel assets in flight, each read and embedded OFF the actor
        // (prepare), its entry added on the actor (commit). A save (every 200) no longer holds up the downloads: the
        // reads in flight keep going while the actor appends.
        let download = iCloudDownloadAllowed(purpose, NetworkState.shared.path)
            ? downloadOrder(downloadQueue(assets: assets, retryFailed: retryFailed),
                            videos: Set(assets.lazy.filter(\.isVideo).map(\.id)))
            : []
        if !download.isEmpty { progress(IndexProgress(done: 0, total: download.count, downloading: true)) }
        await withTaskGroup(of: PreparedResult.self) { g in
            var next = 0, inFlight = 0
            while true {
                while inFlight < downloadParallel, next < download.count, !Task.isCancelled,
                      iCloudDownloadAllowed(purpose, NetworkState.shared.path) {
                    let id = download[next]; next += 1
                    guard let a = byId[id] else { done += 1; continue }
                    let already = entries[id] != nil, kept = keptFaces(a, faceEngine)
                    g.addTask {
                        let t0 = Date()
                        let item = await PhotoIndex.prepare(a, embedder: embedder, faceEngine: faceEngine, purpose: purpose,
                                                            alreadyIndexed: already, kept: kept)
                        IndexTiming.record(a.isVideo ? "0 download-pass video (wall)" : "0 download-pass photo (wall)",
                                           Date().timeIntervalSince(t0))
                        return PreparedResult(asset: a, item: item)
                    }
                    inFlight += 1
                }
                guard let r = await g.next() else { break }
                inFlight -= 1
                commit(r.asset, r.item, faceEngine: faceEngine)
                done += 1
                if done % 200 == 0 { save() }
                progress(IndexProgress(done: done, total: download.count, downloading: true))
            }
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
            if let new = await reembedded(e, faceEngine: fe, purpose: purpose), entries[id] != nil,
               (try? store?.replaceFaces(id, photo: new.photo, frames: new.frames, faceModel: fe.profile.id, faceSide: new.faceSide)) != nil {
                p.done += 1
            } else { p.waiting += 1 }
            if (p.done + p.waiting) % 200 == 0 { save() }
            if (p.done + p.waiting) % 20 == 0 { progress(p) }
        }
        save()
        p.running = false; p.finished = !Task.isCancelled
        progress(p)
        return p
    }

    /// The entry's faces re-embedded by `fe` (same boxes), or nil when the photo / a frame cannot be read now.
    private func reembedded(_ e: IndexEntry, faceEngine fe: FaceEngine, purpose: FetchPurpose) async -> NewFaces? {
        guard let old = try? store?.full(e.id) else { return nil }
        var out = NewFaces(photo: old.faces, frames: old.frames?.map(\.faces), faceSide: e.faceSide)
        if let units = old.frames {
            let ts = units.filter { !$0.faces.isEmpty }.map(\.t)
            guard let frames = await VideoFrames.frames(e.id, at: ts, purpose: purpose) else { return nil }
            var newFaces = units.map(\.faces)
            for k in units.indices where !units[k].faces.isEmpty {
                guard let ci = frames[units[k].t], let cg = CIContext().createCGImage(ci, from: ci.extent),
                      let fs = try? fe.reembed(units[k].faces, in: cg) else { return nil }
                newFaces[k] = fs
            }
            out.frames = newFaces
        }
        if let fs = old.faces, !fs.isEmpty {
            let cg: CGImage, side: Double
            switch await PhotoLibrary.read(e.id, side: CGFloat(indexReadSide), purpose: purpose) {
            case .full(let ui): guard let c = ui.cgImage else { return nil }; cg = c; side = indexReadSide
            case .notFull(let standIn, _):
                // indexed from a stand-in: its faces were found on a smaller local copy, which is still fine to re-read
                guard e.lowRes == true, let c = standIn?.cgImage else { return nil }
                cg = c; side = Double(max(c.width, c.height))
            }
            guard let new = try? fe.reembed(fs, in: cg) else { return nil }
            out.photo = new
            out.faceSide = min(e.faceSideEffective, side)   // vectors from a small read: the face upgrade re-reads it
        }
        return out
    }

    // MARK: face upgrade

    /// Photos with faces (not videos) and how many of them have checked faces (read at faceReadSide).
    func faceUpgradeCounts() -> (checked: Int, total: Int) {
        var c = 0, t = 0
        for e in entries.values where !e.isVideo && e.hasPhotoFaces { t += 1; if e.facesFullSize { c += 1 } }
        return (c, t)
    }

    /// Photos whose faces all came from a small read: people searches do not decide on them yet (FindPicsCore.matchPerson).
    func uncheckedFaceItems() -> Set<String> { Set(entries.values.lazy.filter { $0.needsFaceUpgrade }.map(\.id)) }

    /// FACE UPGRADE (FindPicsCore/FaceUpgrade.swift): photos whose faces were found on a read smaller than faceReadSide
    /// are read again at faceReadSide (downloading when `purpose` allows it on the current network, re-checked before
    /// each new read), their faces detected and embedded again, and the entry's faces replaced in one step (image
    /// vector, place, dates kept). Newest first, faceUpgradeParallel reads in flight, saved every 200. `limit`: at most
    /// that many photos (the foreground works in chunks). A photo that cannot be read now is not tried again this launch
    /// unless `retryFailed` (the charger task). Returns how many photos still need it (not counting skipped ones) and
    /// which were upgraded (saved people whose reference faces are there are re-derived: PeopleStore.refreshAfterUpgrade).
    @discardableResult
    func upgradeFaces(faceEngine fe: FaceEngine, purpose: FetchPurpose, limit: Int? = nil, retryFailed: Bool,
                      progress: @Sendable (IndexProgress) -> Void) async -> FaceUpgradeRun {
        guard load() else { return FaceUpgradeRun(left: 0, upgraded: []) }
        if retryFailed { upgradeSkipped = [] }
        let all = faceUpgradeWork(entries.values.map { (id: $0.id, taken: $0.taken, needsUpgrade: $0.needsFaceUpgrade) },
                                  skip: upgradeSkipped)
        let todo = limit.map { Array(all.prefix($0)) } ?? all
        guard !todo.isEmpty, iCloudDownloadAllowed(purpose, NetworkState.shared.path) else {
            return FaceUpgradeRun(left: all.count, upgraded: [])
        }
        let (checked0, total) = faceUpgradeCounts()
        var upgraded = 0, tried = 0, next = 0, upgradedIds = Set<String>()
        progress(IndexProgress(done: checked0, total: total, downloading: false, faces: true))
        await withTaskGroup(of: UpgradedFaces.self) { g in
            while next < todo.count && next < faceUpgradeParallel {
                let id = todo[next]; next += 1
                g.addTask { await PhotoIndex.readFaces(id, faceEngine: fe, purpose: purpose) }
            }
            while let r = await g.next() {
                tried += 1
                // checked again after the await: deleted meanwhile -> not brought back; already upgraded -> left alone
                if let fs = r.faces, let e = entries[r.id], e.needsFaceUpgrade,
                   (try? store?.replaceFaces(r.id, photo: fs, frames: nil, faceModel: fe.profile.id, faceSide: faceReadSide)) != nil {
                    upgraded += 1; upgradedIds.insert(r.id)
                } else if r.faces == nil { upgradeSkipped.insert(r.id) }
                if tried % 200 == 0 { save() }
                if tried % 5 == 0 || tried == todo.count {
                    progress(IndexProgress(done: checked0 + upgraded, total: total, downloading: false, faces: true))
                }
                if next < todo.count, !Task.isCancelled, iCloudDownloadAllowed(purpose, NetworkState.shared.path) {
                    let id = todo[next]; next += 1
                    g.addTask { await PhotoIndex.readFaces(id, faceEngine: fe, purpose: purpose) }
                }
            }
        }
        save()
        return FaceUpgradeRun(left: entries.values.filter { $0.needsFaceUpgrade && !upgradeSkipped.contains($0.id) }.count,
                              upgraded: upgradedIds)
    }

    /// Per photo / video with faces: the long side of the read its faces came from (IndexEntry.faceSideEffective).
    func faceSidesByItem() -> [String: Double] {
        var out = [String: Double]()
        for (id, e) in entries where e.hasFaces { out[id] = e.faceSideEffective }
        return out
    }

    /// The current faces of these photos (shipped face model only; videos and stale-model entries left out), for
    /// re-deriving saved people after the face upgrade (FindPicsCore.refsAfterUpgrade).
    func photoFaces(_ ids: Set<String>) -> [String: SourcePhotoFaces] {
        var out = [String: SourcePhotoFaces]()
        for id in ids {
            guard let e = entries[id], !e.isVideo, e.facesCurrent,
                  let fs = try? store?.detectedFaces(e.faces.filter { $0.frame == nil }) else { continue }
            out[id] = SourcePhotoFaces(side: e.faceSideEffective, faces: fs.map {
                FaceCandidate(box: $0.box, imageW: $0.imageW, imageH: $0.imageH, embedding: $0.embedding) })
        }
        return out
    }

    /// One face-upgrade read (off the actor: faceUpgradeParallel of these run at once): the photo at faceReadSide and
    /// its faces, or nil faces when it cannot be read at that size now (never a smaller stand-in).
    private static func readFaces(_ id: String, faceEngine fe: FaceEngine, purpose: FetchPurpose) async -> UpgradedFaces {
        guard case .full(let ui) = await PhotoLibrary.read(id, side: CGFloat(faceReadSide), purpose: purpose),
              let cg = ui.cgImage, let fs = try? fe.faces(in: cg) else { return UpgradedFaces(id: id, faces: nil) }
        return UpgradedFaces(id: id, faces: fs)
    }

    /// The download queue (FindPicsCore.indexWork, downloads allowed), library order.
    func downloadQueue(assets: [LibraryAsset], retryFailed: Bool) -> [String] {
        guard load() else { return [] }
        return indexWork(library: assets.map(\.id), indexedLowRes: entries.mapValues { $0.lowRes ?? false }, notRead: notRead,
                         downloads: true, retryFailed: retryFailed).download
    }

    /// Re-read for a new image preparation only (imageVersion): faces this face model already found on a full-size
    /// read are kept; the 448 px read would only make them worse (FindPicsCore.faceReadSide). nil: detect again.
    private func keptFaces(_ a: LibraryAsset, _ faceEngine: FaceEngine?) -> [DetectedFace]? {
        guard let fe = faceEngine, !a.isVideo, let old = entries[a.id], !old.isVideo, old.facesKnown, old.facesFullSize,
              (old.faceModel ?? legacyFaceModel) == fe.profile.id, (old.imageVersion ?? 1) != Embedder.imageVersion
        else { return nil }
        return try? store?.detectedFaces(old.faces.filter { $0.frame == nil })
    }

    /// Core ML / Vision work of the index passes, one asset at a time (MLModel's synchronous prediction is not
    /// documented as safe to call concurrently on one instance; it also bounds memory to one asset's frames). Reads
    /// and downloads do not take it, so downloadParallel of them still overlap.
    static let modelGate = AsyncGate()

    /// Read one asset and compute its vectors and faces, OFF the index actor (the download pass runs
    /// downloadParallel of these at once). `alreadyIndexed`: a stand-in never replaces an entry. `kept`: keptFaces.
    /// Network requests per asset: photo = one 448 px resizeMode .fast image request (PhotoLibrary.read; local first,
    /// then with network); video = one AVAsset request (VideoFrames.avAsset; local first, then the medium-quality
    /// derivative for indexing). The EXIF camera read never downloads.
    static func prepare(_ a: LibraryAsset, embedder: Embedder, faceEngine: FaceEngine?, purpose: FetchPurpose,
                        alreadyIndexed: Bool, kept: [DetectedFace]?) async -> PreparedItem {
        if a.isVideo {                          // several moments of the video, each with its vector and faces
            let (av, why) = await VideoFrames.avAsset(a.id, purpose: purpose)
            guard let asset = av else { return .notRead(why ?? .unreadable) }
            await modelGate.acquire()
            let t0 = Date()
            var units = [FrameUnit]()
            let ctx = CIContext()
            _ = await VideoFrames.sampleEach(asset) { t, cg in
                let im = CIImage(cgImage: cg)
                guard let v = try? embedder.vector(of: im) else { return }
                var fs = [DetectedFace]()
                if let fe = faceEngine, let c = ctx.createCGImage(im, from: im.extent) { fs = (try? fe.faces(in: c)) ?? [] }
                units.append(FrameUnit(t: t, vector: v, faces: fs))
            }
            modelGate.release()
            IndexTiming.record("7 video frames + vectors + faces", Date().timeIntervalSince(t0))
            if units.isEmpty { return .notRead(.unreadable) }
            return .entry(PreparedEntry(vector: units[0].vector, faces: nil, frames: units, camera: nil, lowRes: false,
                                        standInWhy: nil, faceSide: nil))
        }
        let img: UIImage, lowRes: Bool, standInWhy: ReadOutcome?
        switch await IndexTiming.measureAsync("1 PhotoLibrary.read", { await PhotoLibrary.read(a.id, side: CGFloat(indexReadSide), purpose: purpose) }) {
        case .full(let ui): img = ui; lowRes = false; standInWhy = nil
        case .notFull(let standIn, let why):
            // a smaller local copy keeps the photo searchable until its original downloads (never shown to the judge)
            guard let s = standIn, max(PhotoLibrary.pixelSize(s).0, PhotoLibrary.pixelSize(s).1) >= minStandInSide,
                  !alreadyIndexed else { return .notRead(why) }
            img = s; lowRes = true; standInWhy = why
        }
        guard let cg = img.cgImage else { return .notRead(.unreadable) }
        // never download for the camera tag: EXIF needs the whole original (170k iCloud originals on Reza's phone);
        // iCloud-only photos get camera "" (selfie scope then misses them; known gap, JOURNAL 10-07)
        let cam = await IndexTiming.measureAsync("4 PhotoLibrary.camera (EXIF)", { await PhotoLibrary.camera(a.id, network: false) })
        await modelGate.acquire()
        let (vector, faces, faceSide) = computePhoto(cg, lowRes: lowRes, embedder: embedder, faceEngine: faceEngine, kept: kept)
        modelGate.release()
        guard let v = vector else { return .notRead(.unreadable) }
        return .entry(PreparedEntry(vector: v, faces: faces, frames: nil, camera: cam, lowRes: lowRes, standInWhy: standInWhy,
                                    faceSide: faceSide))
    }

    /// The image vector and faces of one photo read (synchronous: runs while holding modelGate).
    private static func computePhoto(_ cg: CGImage, lowRes: Bool, embedder: Embedder, faceEngine: FaceEngine?,
                                     kept: [DetectedFace]?) -> (vector: [Float]?, faces: [DetectedFace]?, faceSide: Double?) {
        guard let v = try? IndexTiming.measure("2 embedder.vector", { try embedder.vector(of: CIImage(cgImage: cg)) }) else {
            return (nil, nil, nil)
        }
        guard let fe = faceEngine else { return (v, nil, nil) }
        if let kept { return (v, kept, faceReadSide) }
        let faces = try? IndexTiming.measure("3 faces(in:) Vision+AuraFace", { try fe.faces(in: cg) })
        return (v, faces, recordedFaceSide(full: !lowRes, requested: indexReadSide, gotLongSide: Double(max(cg.width, cg.height))))
    }

    /// Add what `prepare` read (on the actor), or record why not.
    private func commit(_ a: LibraryAsset, _ item: PreparedItem, faceEngine: FaceEngine?) {
        guard case .entry(let e) = item else {
            if case .notRead(let why) = item { record(a.id, why) }
            return
        }
        if e.lowRes {
            // checked again: an entry made while this read was in flight is never replaced by a stand-in
            if entries[a.id] != nil { record(a.id, e.standInWhy ?? .downloadFailed); return }
            setNotRead(a.id, e.standInWhy)
        }
        let cal = Calendar.current
        let lm = a.created.map { cal.component(.hour, from: $0) * 60 + cal.component(.minute, from: $0) }
        let tAdd = Date()
        add(FullIndexEntry(id: a.id, isVideo: a.isVideo, taken: a.created?.timeIntervalSince1970, localMinutes: lm,
                           lat: a.location?.coordinate.latitude, lon: a.location?.coordinate.longitude, vector: e.vector,
                           faces: e.faces,
                           place: a.location.flatMap { geocoder?.placeText(lat: $0.coordinate.latitude, lon: $0.coordinate.longitude) },
                           frames: e.frames, camera: e.camera, isScreenshot: a.isScreenshot, lowRes: e.lowRes ? true : nil,
                           faceModel: faceEngine?.profile.id, imageVersion: Embedder.imageVersion, faceSide: e.faceSide))
        IndexTiming.record("5 add + save store", Date().timeIntervalSince(tAdd))
        if !e.lowRes { setNotRead(a.id, nil) }
    }

    private func record(_ id: String, _ why: ReadOutcome) {
        guard PhotoLibrary.asset(id) != nil else { setNotRead(id, nil); return }   // deleted meanwhile
        // a stand-in already indexed stays; only its not-read reason changes
        setNotRead(id, why)
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

/// One face-upgrade run: photos still waiting (not counting ones skipped this launch) and the ones upgraded.
struct FaceUpgradeRun: Sendable { let left: Int; let upgraded: Set<String> }

/// One asset read and embedded off the index actor (PhotoIndex.prepare), handed to the actor once (commit).
enum PreparedItem: Sendable {
    case entry(PreparedEntry)
    case notRead(ReadOutcome)
}

struct PreparedEntry: Sendable {
    var vector: [Float]
    var faces: [DetectedFace]?
    var frames: [FrameUnit]?
    var camera: String?
    var lowRes: Bool
    var standInWhy: ReadOutcome?        // the stand-in's reason (lowRes only)
    var faceSide: Double?
}

struct PreparedResult: Sendable { let asset: LibraryAsset; let item: PreparedItem }

/// A FIFO async mutex: waiting tasks SUSPEND (a blocking lock would hold one of the few cooperative threads per waiter).
final class AsyncGate: Sendable {
    private struct State: Sendable { var busy = false; var waiters: [CheckedContinuation<Void, Never>] = [] }
    private let state = OSAllocatedUnfairLock(initialState: State())

    func acquire() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            let now = state.withLock { s -> Bool in
                if s.busy { s.waiters.append(c); return false }
                s.busy = true; return true
            }
            if now { c.resume() }
        }
    }

    func release() {
        let next = state.withLock { s -> CheckedContinuation<Void, Never>? in
            if s.waiters.isEmpty { s.busy = false; return nil }
            return s.waiters.removeFirst()      // ownership passes straight to the next waiter (busy stays true)
        }
        next?.resume()
    }
}

/// One face-upgrade result handed from a reading task to the index actor.
struct UpgradedFaces: Sendable { let id: String; let faces: [DetectedFace]? }

/// An entry's faces after re-embedding: the photo's, each video frame's, and the read size they came from.
struct NewFaces { var photo: [DetectedFace]?; var frames: [[DetectedFace]]?; var faceSide: Double? }


extension PhotoIndex {
    /// Every face in the library made by face model `model` (default: the shipped one; vectors of two models are never
    /// returned together): fingerprints (mapped rows, not copied), the photo each is in, size and detector confidence.
    func allFaces(model: String = FaceProfile.shipped.id) -> LibraryFaces {
        (try? store?.allFaces(model: model)) ?? LibraryFaces()
    }

    /// One entry's faces with fingerprints: a video frame's (`frameT`) or the photo's.
    func detectedFaces(_ id: String, frameT t: Double?) -> [DetectedFace] {
        guard let e = entries[id] else { return [] }
        let fs: [StoredFace]
        if let t { fs = e.frameTs?.firstIndex(of: t).map { k in e.faces.filter { $0.frame == k } } ?? [] }
        else { fs = e.faces.filter { $0.frame == nil } }
        return (try? store?.detectedFaces(fs)) ?? []
    }

    /// The image vector of each id (nil: not indexed).
    func vectors(_ ids: [String]) -> [[Float]?] { ids.map { (try? store?.vector($0)) ?? nil } }
}

extension PhotoIndex {
    /// Image-vector units: one per photo, one per sampled video frame (unitItem = position in `ids`, unitT = frame time).
    /// The vectors are rows of the store's mapped file, read in blocks by lookScores etc. (never all in RAM).
    func units(ids: [String]) -> (vectors: StoredRows, unitItem: [Int], unitT: [Double?]) {
        (try? store?.units(ids: ids)) ?? (StoredRows(), [], [])
    }
}

/// Per-stage timing for the index pass (docs/MAC_INBOX.md M14: "where do the ~4 s per photo go?").
/// Deliberately NOT `#if DEBUG`: M14 asks for these numbers from a RELEASE build, so it is gated at RUNTIME by the
/// `-timeIndex` launch argument and costs two Date() reads per stage when off.
enum IndexTiming {
    nonisolated(unsafe) private static var samples: [String: [Double]] = [:]
    private static let lock = NSLock()
    nonisolated(unsafe) static var on = false

    static func measure<T>(_ stage: String, _ body: () throws -> T) rethrows -> T {
        guard on else { return try body() }
        let t0 = Date(); defer { record(stage, Date().timeIntervalSince(t0)) }
        return try body()
    }

    static func measureAsync<T>(_ stage: String, _ body: () async -> T) async -> T {
        guard on else { return await body() }
        let t0 = Date(); let r = await body(); record(stage, Date().timeIntervalSince(t0)); return r
    }

    static func record(_ stage: String, _ seconds: Double) {
        guard on else { return }
        lock.lock(); samples[stage, default: []].append(seconds); lock.unlock()
    }

    /// Median ms per stage, plus how many photos contributed, ordered slowest first.
    static func report() -> String {
        lock.lock(); let s = samples; lock.unlock()
        guard !s.isEmpty else { return "no timing collected" }
        let rows = s.map { (stage, xs) -> (String, Double, Int) in
            let sorted = xs.sorted()
            return (stage, sorted[sorted.count / 2] * 1000, xs.count)
        }.sorted { $0.1 > $1.1 }
        let total = rows.reduce(0.0) { $0 + $1.1 }
        return rows.map { String(format: "%@: %.0f ms (n=%d)", $0.0, $0.1, $0.2) }.joined(separator: "\n")
             + String(format: "\nsum of medians: %.0f ms", total)
    }
}
