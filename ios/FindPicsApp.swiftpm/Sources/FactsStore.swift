import Foundation
import FindPicsCore

/// What the night pass has written, kept BESIDE the index rather than inside it.
///
/// WHY BESIDE (11-05): the main store is a memory-mapped binary file holding 187,231 entries that took a week to
/// build. Adding a per-photo field to it is a schema migration, and a migration that goes wrong costs Reza that
/// week. The verification the cluster asked for (300 photos, three queries) does not need the real schema - it
/// needs the numbers. So: a side file, reversible by deleting it, with the migration done later once the facts
/// path has earned its place.
///
/// Layout: "FPFACTS1" + UInt32 count, then per photo: UInt16 id length, id bytes, UInt16 line length, line bytes,
/// then `dim` Float16. fp16 because that is what the cluster sized the real thing at (768 B per photo, 144 MB for
/// a 187k library) and the scorer was trained against fp16 inputs.
actor FactsStore {
    struct Row: Sendable { let line: String; let vector: [Float] }

    private(set) var rows: [String: Row] = [:]
    private var loaded = false
    private let dim: Int
    private let file: URL

    init(dim: Int = 384, name: String = "facts.bin") {
        self.dim = dim
        self.file = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(name)
    }

    var count: Int { rows.count }
    func has(_ id: String) -> Bool { rows[id] != nil }
    func row(_ id: String) -> Row? { rows[id] }

    @discardableResult func load() -> Bool {
        if loaded { return true }
        defer { loaded = true }
        guard let d = try? Data(contentsOf: file), d.count > 12,
              String(data: d[0..<8], encoding: .ascii) == "FPFACTS1" else { return false }
        var o = 12
        func u16() -> Int? {
            guard o + 2 <= d.count else { return nil }
            let v = Int(d[o]) | Int(d[o + 1]) << 8; o += 2; return v
        }
        let n = Int(d[8]) | Int(d[9]) << 8 | Int(d[10]) << 16 | Int(d[11]) << 24
        for _ in 0..<n {
            guard let idLen = u16(), o + idLen <= d.count,
                  let id = String(data: d[o..<(o + idLen)], encoding: .utf8) else { break }
            o += idLen
            guard let lineLen = u16(), o + lineLen <= d.count,
                  let line = String(data: d[o..<(o + lineLen)], encoding: .utf8) else { break }
            o += lineLen
            guard o + 2 * dim <= d.count else { break }
            var v = [Float](repeating: 0, count: dim)
            for k in 0..<dim {
                let bits = UInt16(d[o + 2 * k]) | UInt16(d[o + 2 * k + 1]) << 8
                v[k] = Float(Float16(bitPattern: bits))
            }
            o += 2 * dim
            rows[id] = Row(line: line, vector: v)
        }
        return true
    }

    func put(id: String, line: String, vector: [Float]) {
        _ = load()
        guard vector.count == dim else { return }
        rows[id] = Row(line: line, vector: vector)
    }

    /// Written whole rather than appended: a half-written append would make the file unreadable, and rewriting
    /// 144 MB occasionally is cheaper than losing the pass.
    func save() {
        guard loaded else { return }
        var d = Data("FPFACTS1".utf8)
        var n = UInt32(rows.count).littleEndian
        withUnsafeBytes(of: &n) { d.append(contentsOf: $0) }
        for (id, r) in rows {
            let idB = Data(id.utf8), lineB = Data(r.line.utf8)
            var a = UInt16(idB.count).littleEndian, b = UInt16(lineB.count).littleEndian
            withUnsafeBytes(of: &a) { d.append(contentsOf: $0) }; d.append(idB)
            withUnsafeBytes(of: &b) { d.append(contentsOf: $0) }; d.append(lineB)
            for x in r.vector {
                var h = Float16(x).bitPattern.littleEndian
                withUnsafeBytes(of: &h) { d.append(contentsOf: $0) }
            }
        }
        try? d.write(to: file, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
    }
}
