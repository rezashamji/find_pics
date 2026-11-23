// Which compiled Core ML model a bundled .mlpackage maps to (11-23). Free of Core ML so it is tested on Linux; the app
// (CompiledModels.swift) does the compiling and the file moves.
//
// MEASURED 11-23 03:47 on Reza's phone: iOS filed a disk-writes report for find pics - 4.29 GB written in 2,000 s,
// against a budget of 49.71 KB/s over a day - and the heaviest stack ran through Core ML into libcopyfile. The app
// ships its three image/face models as .mlpackage (~640 MB) and called MLModel.compileModel on EVERY launch, which
// writes a fresh compiled copy each time - including every launch iOS makes for the charger task.
import Foundation

/// A stable name for one bundled package's compiled form: the model name plus a fingerprint of the package's files
/// (relative path and size of each, and the manifest's text). A new model in a new build gets a new key and is
/// compiled once; the same model across rebuilds and launches reuses the compiled copy.
public func compiledModelKey(name: String, files: [(path: String, size: Int)], manifest: String) -> String {
    let listing = files.sorted { $0.path < $1.path }.map { "\($0.path):\($0.size)" }.joined(separator: ";")
    let h = fnv1a64(name + "|" + listing + "|" + manifest)
    return name.replacingOccurrences(of: "/", with: "_") + "-" + String(h, radix: 16)
}
