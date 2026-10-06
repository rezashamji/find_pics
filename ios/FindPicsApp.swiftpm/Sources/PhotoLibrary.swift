// The photo library, through PhotoKit. READ-ONLY except one thing: after the person taps "Save as album", a NEW album is
// created and the found photos are ADDED to it. Nothing is ever deleted, moved or edited (no delete API is called).
import CoreImage
import CoreLocation
import ImageIO
import Photos
import UIKit

struct LibraryAsset: Identifiable, Hashable {
    let id: String              // PHAsset.localIdentifier
    let isVideo: Bool
    let created: Date?
    let location: CLLocation?
}

enum PhotoLibrary {
    static func requestAccess() async -> Bool {
        let s = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        return s == .authorized || s == .limited
    }

    /// Every photo and video, newest first.
    static func allAssets() -> [LibraryAsset] {
        let opts = PHFetchOptions()
        opts.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        let r = PHAsset.fetchAssets(with: opts)
        var out = [LibraryAsset](); out.reserveCapacity(r.count)
        r.enumerateObjects { a, _, _ in
            guard a.mediaType == .image || a.mediaType == .video else { return }
            out.append(LibraryAsset(id: a.localIdentifier, isVideo: a.mediaType == .video, created: a.creationDate,
                                    location: a.location))
        }
        return out
    }

    static func asset(_ id: String) -> PHAsset? {
        PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil).firstObject
    }

    /// An image of the asset at roughly `side` px (a video's poster frame), for the models or the grid.
    static func image(_ id: String, side: CGFloat) async -> UIImage? {
        guard let a = asset(id) else { return nil }
        let o = PHImageRequestOptions()
        o.deliveryMode = .highQualityFormat; o.isNetworkAccessAllowed = false   // iCloud-only originals: use what is on the phone
        o.resizeMode = .exact; o.isSynchronous = false
        return await withCheckedContinuation { cont in
            var done = false
            PHImageManager.default().requestImage(for: a, targetSize: CGSize(width: side, height: side),
                                                  contentMode: .aspectFit, options: o) { img, info in
                if done { return }
                if (info?[PHImageResultIsDegradedKey] as? Bool) == true { return }
                done = true; cont.resume(returning: img)
            }
        }
    }

    /// "front" / "back" from the original's EXIF lens model ("iPhone 13 Pro front camera 2.71mm f/2.2"); "" when the
    /// file has none (screenshots, saved images) or the original is only in iCloud (no download just for this).
    static func camera(_ id: String) async -> String {
        guard let a = asset(id) else { return "" }
        let o = PHImageRequestOptions()
        o.isNetworkAccessAllowed = false; o.deliveryMode = .highQualityFormat; o.isSynchronous = false
        return await withCheckedContinuation { cont in
            PHImageManager.default().requestImageDataAndOrientation(for: a, options: o) { data, _, _, _ in
                guard let d = data, let src = CGImageSourceCreateWithData(d as CFData, nil),
                      let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
                      let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any],
                      let lens = (exif[kCGImagePropertyExifLensModel] as? String)?.lowercased(), !lens.isEmpty
                else { cont.resume(returning: ""); return }
                cont.resume(returning: lens.contains("front") ? "front" : "back")
            }
        }
    }

    static func ciImage(_ id: String, side: CGFloat) async -> CIImage? {
        guard let ui = await image(id, side: side) else { return nil }
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
