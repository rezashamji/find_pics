// The photo library, through PhotoKit. READ-ONLY except one thing: after the person taps "Save as album", a NEW album is
// created and the found photos are ADDED to it. Nothing is ever deleted, moved or edited (no delete API is called).
// Originals kept only in iCloud ("Optimize iPhone Storage") are DOWNLOADED to the phone to be read (never uploaded),
// when FindPicsCore.iCloudDownloadAllowed says so (Wi-Fi for indexing, not in Low Data Mode), with a per-asset stall
// timeout. The judge only ever gets the full-resolution image (FindPicsCore.isFullResolution), never a stand-in.
import CoreImage
import CoreLocation
@preconcurrency import FindPicsCore
import ImageIO
import Network
import os
import Photos
import UIKit

struct LibraryAsset: Identifiable, Hashable, @unchecked Sendable {   // CLLocation is immutable
    let id: String              // PHAsset.localIdentifier
    let isVideo: Bool
    let created: Date?
    let location: CLLocation?
    var isScreenshot = false        // PHAssetMediaSubtype.photoScreenshot
}

/// The current network, from NWPathMonitor (expensive = cellular / hotspot, constrained = Low Data Mode). Until the
/// first report it says "not connected", so nothing downloads by accident.
final class NetworkState: @unchecked Sendable {     // all state behind the lock
    static let shared = NetworkState()
    private let monitor = NWPathMonitor()
    private let state = OSAllocatedUnfairLock(initialState: NetworkPath(connected: false, expensive: true, constrained: true))
    private init() {
        let s = state
        monitor.pathUpdateHandler = { p in
            let np = NetworkPath(connected: p.status == .satisfied, expensive: p.isExpensive, constrained: p.isConstrained)
            s.withLock { $0 = np }
        }
        monitor.start(queue: DispatchQueue(label: "findpics.network"))
    }
    var path: NetworkPath { state.withLock { $0 } }
}

/// What a read gave: the full-resolution image, or why not (with a smaller local copy when PhotoKit had one).
enum ImageRead: @unchecked Sendable {        // UIImage is immutable once made; handed over once
    case full(UIImage)
    case notFull(standIn: UIImage?, reason: ReadOutcome)
}

/// One PhotoKit request -> exactly one answer: the result handler (maybe called twice: degraded, then final), the
/// stall watchdog and the cancel race for it; the first one wins. Progress resets the stall clock.
final class OnceBox<T>: @unchecked Sendable {   // all state behind the lock
    private let lock = NSLock()
    private var cont: CheckedContinuation<T, Never>?
    private var lastProgress = Date()
    private(set) var standIn: UIImage?
    init(_ c: CheckedContinuation<T, Never>) { cont = c }
    @discardableResult func fire(_ v: sending T) -> Bool {
        lock.lock(); let c = cont; cont = nil; lock.unlock()
        guard let c else { return false }
        c.resume(returning: v); return true
    }
    var done: Bool { lock.lock(); defer { lock.unlock() }; return cont == nil }
    func touch() { lock.lock(); lastProgress = Date(); lock.unlock() }
    func stalled(_ seconds: Double) -> Bool { lock.lock(); defer { lock.unlock() }; return Date().timeIntervalSince(lastProgress) > seconds }
    func keep(_ im: UIImage) { lock.lock(); standIn = im; lock.unlock() }
}

/// Watchdog: every 2 s; gives up when no progress came for `stall` seconds (or after 4 x stall in total).
func watchStall<T>(_ box: OnceBox<T>, stall: Double, started: Date = Date(), onGiveUp: @escaping @Sendable () -> Void) {
    DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
        if box.done { return }
        if box.stalled(stall) || Date().timeIntervalSince(started) > 4 * stall { onGiveUp(); return }
        watchStall(box, stall: stall, started: started, onGiveUp: onGiveUp)
    }
}

enum PhotoLibrary {
    static func requestAccess() async -> Bool {
        let s = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        return s == .authorized || s == .limited
    }

    /// Access already given (the background task must not prompt).
    static var hasAccess: Bool {
        let s = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        return s == .authorized || s == .limited
    }

    static func fetchAll() -> PHFetchResult<PHAsset> {
        let opts = PHFetchOptions()
        opts.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        return PHAsset.fetchAssets(with: opts)
    }

    static func libraryAsset(_ a: PHAsset) -> LibraryAsset? {
        guard a.mediaType == .image || a.mediaType == .video else { return nil }
        return LibraryAsset(id: a.localIdentifier, isVideo: a.mediaType == .video, created: a.creationDate,
                            location: a.location, isScreenshot: a.mediaSubtypes.contains(.photoScreenshot))
    }

    /// Every photo and video, newest first.
    static func allAssets() -> [LibraryAsset] {
        let r = fetchAll()
        var out = [LibraryAsset](); out.reserveCapacity(r.count)
        r.enumerateObjects { a, _, _ in if let x = libraryAsset(a) { out.append(x) } }
        return out
    }

    static func asset(_ id: String) -> PHAsset? {
        PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil).firstObject
    }

    private enum Raw: @unchecked Sendable { case image(UIImage), inCloud, failed, timedOut }

    /// One request at `side` px. `network`: PhotoKit may download the original (progress resets the stall timer).
    private static func request(_ a: PHAsset, side: CGFloat, network: Bool, stall: Double) async -> (Raw, UIImage?) {
        let o = PHImageRequestOptions()
        o.deliveryMode = .highQualityFormat; o.isNetworkAccessAllowed = network
        o.resizeMode = .exact; o.isSynchronous = false; o.version = .current
        var boxRef: OnceBox<Raw>?
        let raw = await withCheckedContinuation { (cont: CheckedContinuation<Raw, Never>) in
            let box = OnceBox(cont); boxRef = box
            o.progressHandler = { _, _, _, _ in box.touch() }
            let rid = PHImageManager.default().requestImage(for: a, targetSize: CGSize(width: side, height: side),
                                                            contentMode: .aspectFit, options: o) { img, info in
                if (info?[PHImageResultIsDegradedKey] as? Bool) == true { if let i = img { box.keep(i) }; return }
                if let i = img { box.fire(.image(i)); return }
                if (info?[PHImageResultIsInCloudKey] as? Bool) == true { box.fire(.inCloud); return }
                box.fire(.failed)
            }
            watchStall(box, stall: stall) { if box.fire(.timedOut) { PHImageManager.default().cancelImageRequest(rid) } }
        }
        return (raw, boxRef?.standIn)
    }

    static func pixelSize(_ im: UIImage) -> (Double, Double) {
        if let cg = im.cgImage { return (Double(cg.width), Double(cg.height)) }
        return (Double(im.size.width * im.scale), Double(im.size.height * im.scale))
    }

    /// Read for the models: the full-resolution image at `side`, or why not. Local first; then, if the original is
    /// only in iCloud (or only a smaller copy is on the phone) and `purpose` allows it on this network, download it.
    static func read(_ id: String, side: CGFloat, purpose: FetchPurpose) async -> ImageRead {
        guard let a = asset(id) else { return .notFull(standIn: nil, reason: .unreadable) }
        let ow = Double(a.pixelWidth), oh = Double(a.pixelHeight)
        func isFull(_ im: UIImage) -> Bool {
            let (w, h) = pixelSize(im)
            return isFullResolution(gotW: w, gotH: h, requestedSide: Double(side), originalW: ow, originalH: oh)
        }
        let (local, degraded) = await request(a, side: side, network: false, stall: 20)
        var standIn: UIImage? = degraded
        switch local {
        case .image(let im): if isFull(im) { return .full(im) } else { standIn = im }
        case .failed, .timedOut: if degraded == nil { return .notFull(standIn: nil, reason: .unreadable) }
        case .inCloud: break
        }
        guard iCloudDownloadAllowed(purpose, NetworkState.shared.path) else {
            return .notFull(standIn: standIn, reason: .waitingForICloud)
        }
        let (net, _) = await request(a, side: side, network: true, stall: ICloudTimeout.photo)
        if case .image(let im) = net, isFull(im) { return .full(im) }
        return .notFull(standIn: standIn, reason: .downloadFailed)
    }

    /// An image of the asset at roughly `side` px (a video's poster frame) for the grid / the full-screen view.
    /// `network`: allow an iCloud download (the full-screen view, unless Low Data Mode).
    static func image(_ id: String, side: CGFloat, network: Bool = false) async -> UIImage? {
        guard let a = asset(id) else { return nil }
        let (r, degraded) = await request(a, side: side, network: network && !NetworkState.shared.path.constrained,
                                          stall: ICloudTimeout.photo)
        if case .image(let im) = r { return im }
        return degraded
    }

    /// "front" / "back" from the original's EXIF lens model ("iPhone 13 Pro front camera 2.71mm f/2.2"); "" when the
    /// file has none (screenshots, saved images) or the original is only in iCloud and `network` is false.
    static func camera(_ id: String, network: Bool = false) async -> String {
        guard let a = asset(id) else { return "" }
        let o = PHImageRequestOptions()
        o.isNetworkAccessAllowed = network; o.deliveryMode = .highQualityFormat; o.isSynchronous = false; o.version = .original
        return await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
            let box = OnceBox(cont)
            o.progressHandler = { _, _, _, _ in box.touch() }
            let rid = PHImageManager.default().requestImageDataAndOrientation(for: a, options: o) { data, _, _, _ in
                guard let d = data, let src = CGImageSourceCreateWithData(d as CFData, nil),
                      let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
                      let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any],
                      let lens = (exif[kCGImagePropertyExifLensModel] as? String)?.lowercased(), !lens.isEmpty
                else { box.fire(""); return }
                box.fire(lens.contains("front") ? "front" : "back")
            }
            watchStall(box, stall: ICloudTimeout.photo) { if box.fire("") { PHImageManager.default().cancelImageRequest(rid) } }
        }
    }

    /// For the judge (and the subject examples): the FULL-resolution image or nil (never a smaller stand-in).
    static func ciImage(_ id: String, side: CGFloat, purpose: FetchPurpose = .judge) async -> CIImage? {
        guard case .full(let ui) = await read(id, side: side, purpose: purpose) else { return nil }
        return ui.cgImage.map { CIImage(cgImage: $0) } ?? CIImage(image: ui)
    }

    /// Creates a NEW album and adds the photos. Only called from the Save button.
    /// True when the person shared only some photos: iOS then does not let apps create albums.
    static var isLimited: Bool { PHPhotoLibrary.authorizationStatus(for: .readWrite) == .limited }

    static func saveAlbum(named name: String, ids: [String]) async throws {
        let assets = PHAsset.fetchAssets(withLocalIdentifiers: ids, options: nil)
        try await PHPhotoLibrary.shared().performChanges {
            let req = PHAssetCollectionChangeRequest.creationRequestForAssetCollection(withTitle: name)
            req.addAssets(assets)
        }
    }
}

/// What changed in the library while the app runs (PHPhotoLibraryChangeObserver). `full`: PhotoKit could not say
/// exactly what changed -> compare the whole library with the index.
struct LibraryChange: Sendable { var inserted: [LibraryAsset] = []; var removed: [String] = []; var full = false }

/// Watches the library from the first fetch on; calls `onChange` (on PhotoKit's background queue) for new / removed
/// photos and videos. Edits of existing photos are not re-indexed (their vectors stay those of the original).
final class LibraryObserver: NSObject, PHPhotoLibraryChangeObserver, @unchecked Sendable {   // `fetch` behind the lock
    private let lock = NSLock()
    private var fetch: PHFetchResult<PHAsset>
    private let onChange: @Sendable (LibraryChange) -> Void

    init(onChange: @escaping @Sendable (LibraryChange) -> Void) {
        self.onChange = onChange
        fetch = PhotoLibrary.fetchAll()
        super.init()
        PHPhotoLibrary.shared().register(self)
    }

    deinit { PHPhotoLibrary.shared().unregisterChangeObserver(self) }

    func photoLibraryDidChange(_ changeInstance: PHChange) {
        lock.lock()
        guard let d = changeInstance.changeDetails(for: fetch) else { lock.unlock(); return }
        fetch = d.fetchResultAfterChanges
        lock.unlock()
        var c = LibraryChange()
        if !d.hasIncrementalChanges { c.full = true }
        else {
            c.inserted = d.insertedObjects.compactMap(PhotoLibrary.libraryAsset)
            c.removed = d.removedObjects.map(\.localIdentifier)
        }
        if c.full || !c.inserted.isEmpty || !c.removed.isEmpty { onChange(c) }
    }
}
