import CoreML
@preconcurrency import FindPicsCore
import Foundation

/// Compile each bundled .mlpackage ONCE and keep the result in Application Support (FindPicsCore.compiledModelKey says
/// why: compiling on every launch wrote ~640 MB per launch and drew an iOS disk-writes report). An .mlmodelc in the
/// bundle is used as it is. The cache is excluded from iCloud backup: it is rebuilt from the bundle when missing.
enum CompiledModels {
    static func url(forBundled url: URL, name: String) throws -> URL {
        if url.pathExtension == "mlmodelc" { return url }
        let fm = FileManager.default
        let dir = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("compiled_models", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        var files = [(path: String, size: Int)]()
        if let e = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) {
            for case let f as URL in e {
                let v = try? f.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
                guard v?.isRegularFile == true else { continue }
                files.append((String(f.path.dropFirst(url.path.count)), v?.fileSize ?? 0))
            }
        }
        let manifest = (try? String(contentsOf: url.appendingPathComponent("Manifest.json"), encoding: .utf8)) ?? ""
        let key = compiledModelKey(name: name, files: files, manifest: manifest)
        let target = dir.appendingPathComponent(key + ".mlmodelc", isDirectory: true)
        if fm.fileExists(atPath: target.appendingPathComponent("coremldata.bin").path)
            || fm.fileExists(atPath: target.appendingPathComponent("model.mil").path) {
            return target
        }
        let t0 = Date()
        let compiled = try MLModel.compileModel(at: url)          // a temporary directory iOS may delete
        try? fm.removeItem(at: target)
        try fm.moveItem(at: compiled, to: target)
        var rv = URLResourceValues(); rv.isExcludedFromBackup = true
        var t = target; try? t.setResourceValues(rv)
        // older compiled copies of the same model (an earlier build's) are dead weight
        let prefix = name.replacingOccurrences(of: "/", with: "_") + "-"
        for old in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        where old.hasPrefix(prefix) && old != target.lastPathComponent {
            try? fm.removeItem(at: dir.appendingPathComponent(old))
        }
        Judge.logMem(String(format: "COREML compiled %@ once in %.1f s -> %@", name, Date().timeIntervalSince(t0), key))
        return target
    }
}
