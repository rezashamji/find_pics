// The on-phone index on disk, built for a whole library (Reza's: 187k photos). Replaces index.json + not_read.json,
// which held every vector as JSON text, were rewritten whole every 200 photos (~12 KB per entry: 2.3 GB at 187k) and
// had to be decoded whole at launch (~100 s at 17.6k entries, measured on the Mac).
//
// Files (one directory, app-private, derived data only):
//   CURRENT          the live generation number N (replaced atomically: write CURRENT.tmp, rename)
//   meta-N.snap      snapshot of every entry's metadata + the not-read list (binary, CRC-32 checked)
//   log-N.log        append-only journal since the snapshot: put / delete / not-read records, each length + CRC
//                    framed, grouped by COMMIT records; only committed groups are applied on open
//   img-V.vec        image vectors, one row per unit (photo, or video frame): Float16 x imageDim, never rewritten
//   face-W.vec       face fingerprints, one row per face: Float16 (or Float32) x faceDim
// Writes: new vectors are appended to the .vec files; `commit` fsyncs them, then appends the journal records and a
// COMMIT record and fsyncs the log. A replaced / deleted entry leaves dead rows (garbage) behind; compaction writes a
// new generation (fresh snapshot, empty log, and, when garbage is high, compacted .vec files) and switches CURRENT.
// Crash safety: whatever is past the last COMMIT (a half-written record, rows not yet committed) is cut off on open;
// a compaction killed before the CURRENT rename leaves the old generation intact (the new files are deleted on open).
// Reads: the .vec files are memory-mapped (mmap, read-only); StoredRows reads rows a block at a time (Float16 ->
// Float32 per block) so search RAM stays bounded; mapped clean pages are not the app's dirty memory.
import Foundation
#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

public enum IndexStoreError: Error, Equatable {
    /// A file exists but cannot be opened now (e.g. iOS data protection before the first unlock): try again later.
    case unreadable(String, Int32)
    /// The files are damaged beyond what crash recovery repairs.
    case corrupt(String)
    /// Made with other vector sizes / types than asked for.
    case incompatible(String)
    case io(String, Int32)
    /// A vector of the wrong length was handed in.
    case badVector(String)
}

public enum StoredScalar: UInt8, Sendable { case f16 = 1, f32 = 2; public var bytes: Int { self == .f16 ? 2 : 4 } }

// MARK: - files

/// A read-only memory map of a file's first `length` bytes (unmapped when the last reader lets go).
public final class MappedFile: @unchecked Sendable {
    let base: UnsafeRawPointer?
    let length: Int
    init(fd: Int32, length: Int, path: String) throws {
        self.length = length; owned = false
        if length == 0 { base = nil; return }
        let p = mmap(nil, length, PROT_READ, MAP_SHARED, fd, 0)
        guard let p, p != UnsafeMutableRawPointer(bitPattern: -1) else { throw IndexStoreError.io("mmap " + path, errno) }
        base = UnsafeRawPointer(p)
        _ = madvise(p, length, MADV_SEQUENTIAL)      // searches scan rows in order: read ahead
    }
    /// Read a whole file into memory with read() (snapshot / log / CURRENT / old JSON parsing: sequential, once).
    static func read(_ path: String) throws -> MappedFile {
        let fd = Posix.openRead(path)
        guard fd >= 0 else { throw Posix.openError(path) }
        defer { close(fd) }
        return try MappedFile(readingAll: fd, length: try Posix.size(fd, path), path: path)
    }
    /// Map a whole file read-only (the old index.json: hundreds of MB, scanned once; mapped clean pages, not app memory).
    static func map(_ path: String) throws -> MappedFile {
        let fd = Posix.openRead(path)
        guard fd >= 0 else { throw Posix.openError(path) }
        defer { close(fd) }
        return try MappedFile(fd: fd, length: try Posix.size(fd, path), path: path)
    }
    private let owned: Bool
    /// A private copy of the file's first `length` bytes (freed on deinit).
    init(readingAll fd: Int32, length: Int, path: String) throws {
        self.length = length; owned = true
        if length == 0 { base = nil; return }
        let m = UnsafeMutableRawPointer.allocate(byteCount: length, alignment: 16)
        var done = 0
        while done < length {
            let n = pread(fd, m + done, length - done, off_t(done))
            if n < 0 && errno == EINTR { continue }
            if n <= 0 { m.deallocate(); throw IndexStoreError.io("read " + path, errno) }
            done += n
        }
        base = UnsafeRawPointer(m)
    }
    var bytes: UnsafeRawBufferPointer { UnsafeRawBufferPointer(start: base, count: base == nil ? 0 : length) }
    deinit {
        guard let b = base else { return }
        if owned { UnsafeMutableRawPointer(mutating: b).deallocate() } else { munmap(UnsafeMutableRawPointer(mutating: b), length) }
    }
}

enum Posix {
    static func openRead(_ path: String) -> Int32 { open(path, O_RDONLY) }
    static func openError(_ path: String) -> IndexStoreError {
        let e = errno
        return (e == EPERM || e == EACCES) ? .unreadable(path, e) : .io("open " + path, e)
    }
    static func openRW(_ path: String, create: Bool, truncate: Bool = false) throws -> Int32 {
        var flags = O_RDWR
        if create { flags |= O_CREAT }
        if truncate { flags |= O_TRUNC }
        let fd = open(path, flags, 0o600)
        guard fd >= 0 else { throw openError(path) }
        return fd
    }
    static func size(_ fd: Int32, _ path: String) throws -> Int {
        let s = lseek(fd, 0, SEEK_END)
        guard s >= 0 else { throw IndexStoreError.io("seek " + path, errno) }
        return Int(s)
    }
    static func pwriteAll(_ fd: Int32, _ p: UnsafeRawBufferPointer, at offset: Int, _ path: String) throws {
        var done = 0
        while done < p.count {
            let n = pwrite(fd, p.baseAddress! + done, p.count - done, off_t(offset + done))
            if n < 0 { if errno == EINTR { continue }; throw IndexStoreError.io("write " + path, errno) }
            done += n
        }
    }
    static func sync(_ fd: Int32, _ path: String) throws {
        guard fsync(fd) == 0 else { throw IndexStoreError.io("fsync " + path, errno) }
    }
    static func truncate(_ fd: Int32, _ size: Int, _ path: String) throws {
        guard ftruncate(fd, off_t(size)) == 0 else { throw IndexStoreError.io("truncate " + path, errno) }
    }
    static func exists(_ path: String) -> Bool { access(path, F_OK) == 0 }
    /// Durably make `path` hold `bytes`: tmp file, fsync, rename (atomic on POSIX file systems), fsync the directory.
    static func atomicWrite(_ bytes: [UInt8], to path: String, dir: String) throws {
        let tmp = path + ".tmp"
        let fd = try openRW(tmp, create: true, truncate: true)
        do {
            try bytes.withUnsafeBytes { try pwriteAll(fd, $0, at: 0, tmp) }
            try sync(fd, tmp)
        } catch { close(fd); throw error }
        close(fd)
        guard rename(tmp, path) == 0 else { throw IndexStoreError.io("rename " + tmp, errno) }
        syncDir(dir)
    }
    static func syncDir(_ dir: String) {
        let fd = open(dir, O_RDONLY)
        if fd >= 0 { _ = fsync(fd); close(fd) }
    }
}

/// An append-only file of fixed-size rows.
final class VectorFile {
    let path: String
    let fd: Int32
    let rowBytes: Int
    private(set) var rows: Int
    private var map: MappedFile?
    private var mappedRows = 0

    /// Opens (creating if asked) and cuts the file back to `committedRows` (rows past the last commit are garbage).
    init(path: String, rowBytes: Int, committedRows: Int, create: Bool) throws {
        self.path = path; self.rowBytes = rowBytes
        if !create && !Posix.exists(path) { throw IndexStoreError.corrupt("missing " + path) }
        fd = try Posix.openRW(path, create: create)
        let size = try Posix.size(fd, path)
        if size < committedRows * rowBytes {
            close(fd); throw IndexStoreError.corrupt("\(path): \(size) bytes, \(committedRows) committed rows")
        }
        if size > committedRows * rowBytes { try Posix.truncate(fd, committedRows * rowBytes, path) }
        rows = committedRows
    }
    deinit { close(fd) }

    /// Appends rows (bytes = n * rowBytes); returns the first new row.
    func append(_ bytes: UnsafeRawBufferPointer) throws -> Int {
        precondition(bytes.count % rowBytes == 0)
        let first = rows
        if bytes.count > 0 { try Posix.pwriteAll(fd, bytes, at: rows * rowBytes, path) }
        rows += bytes.count / rowBytes
        return first
    }
    func sync() throws { try Posix.sync(fd, path) }
    /// A map covering every row written so far (remapped only when rows were added since the last one).
    func mapping() throws -> MappedFile? {
        if rows == 0 { return nil }
        if map == nil || mappedRows < rows {
            map = try MappedFile(fd: fd, length: rows * rowBytes, path: path)
            mappedRows = rows
        }
        return map
    }
}

// MARK: - reading rows

/// Rows of a store vector file (memory-mapped), picked by physical row; Float32 out, a block at a time.
public struct StoredRows: EmbeddingRows {
    let file: MappedFile?
    let scalar: StoredScalar
    public let dim: Int
    let physical: [Int32]
    public var count: Int { physical.count }
    /// No rows.
    public init() { file = nil; scalar = .f32; dim = 0; physical = [] }
    init(file: MappedFile?, scalar: StoredScalar, dim: Int, physical: [Int32]) {
        self.file = file; self.scalar = scalar; self.dim = dim; self.physical = physical
    }
    /// The physical row behind logical row i.
    public func physicalRow(_ i: Int) -> Int { Int(physical[i]) }

    public func withRows<R>(_ start: Int, _ n: Int, _ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        precondition(start >= 0 && n >= 0 && start + n <= physical.count)
        let rb = dim * scalar.bytes
        guard n > 0, let base = file?.base else { return try body(UnsafeBufferPointer(start: nil, count: 0)) }
        if scalar == .f32 && n == 1 {
            let p = (base + Int(physical[start]) * rb).assumingMemoryBound(to: Float.self)
            return try body(UnsafeBufferPointer(start: p, count: dim))
        }
        return try withUnsafeTemporaryAllocation(of: Float.self, capacity: n * dim) { buf in
            let dst = buf.baseAddress!
            for k in 0..<n {
                let src = base + Int(physical[start + k]) * rb
                if scalar == .f16 { MatrixMath.halfToFloat(src, dst + k * dim, dim) }
                else { (dst + k * dim).update(from: src.assumingMemoryBound(to: Float.self), count: dim) }
            }
            return try body(UnsafeBufferPointer(buf))
        }
    }
}

/// Every face people searches use (the shipped model's, or `model`'s), with where each is.
public struct LibraryFaces: Sendable {
    public let emb: StoredRows
    public let item: [String]
    public let px: [Float]
    public let det: [Float]
    public let box: [StoredFace]
    /// When each face's photo was taken (bounded "Who is X?" grouping: faceGroups' sample).
    public let taken: [Double?]
    public init(emb: StoredRows = StoredRows(), item: [String] = [], px: [Float] = [], det: [Float] = [], box: [StoredFace] = [],
                taken: [Double?] = []) {
        self.emb = emb; self.item = item; self.px = px; self.det = det; self.box = box; self.taken = taken
    }
}

// MARK: - binary encoding

struct BinWriter {
    var buf: [UInt8] = []
    mutating func u8(_ v: UInt8) { buf.append(v) }
    mutating func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { buf.append(contentsOf: $0) } }
    mutating func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { buf.append(contentsOf: $0) } }
    mutating func i64(_ v: Int) { u64(UInt64(bitPattern: Int64(v))) }
    mutating func f64(_ v: Double) { u64(v.bitPattern) }
    mutating func f32(_ v: Float) { u32(v.bitPattern) }
    mutating func bool(_ v: Bool) { u8(v ? 1 : 0) }
    mutating func str(_ s: String) {
        var s = s
        s.withUTF8 { p in u32(UInt32(p.count)); buf.append(contentsOf: p) }
    }
    mutating func opt<T>(_ v: T?, _ w: (inout BinWriter, T) -> Void) { if let v { u8(1); w(&self, v) } else { u8(0) } }
    /// A framed record: u32 length of (type + payload), u8 type, payload, u32 CRC-32 of (type + payload).
    mutating func record(_ type: UInt8, _ body: (inout BinWriter) -> Void) {
        let at = buf.count
        u32(0); u8(type)
        body(&self)
        let len = buf.count - at - 4
        withUnsafeBytes(of: UInt32(len).littleEndian) { for k in 0..<4 { buf[at + k] = $0[k] } }
        let crc = buf.withUnsafeBytes { CRC32.of(UnsafeRawBufferPointer(rebasing: $0[(at + 4)...])) }
        u32(crc)
    }
}

struct BinReader {
    let p: UnsafeRawBufferPointer
    var pos: Int
    init(_ p: UnsafeRawBufferPointer, pos: Int = 0) { self.p = p; self.pos = pos }
    var remaining: Int { p.count - pos }
    struct Short: Error {}
    @inline(__always) mutating func need(_ n: Int) throws { if n < 0 || pos + n > p.count { throw Short() } }
    mutating func u8() throws -> UInt8 { try need(1); defer { pos += 1 }; return p[pos] }
    mutating func u32() throws -> UInt32 { try need(4); defer { pos += 4 }; return UInt32(littleEndian: p.loadUnaligned(fromByteOffset: pos, as: UInt32.self)) }
    mutating func u64() throws -> UInt64 { try need(8); defer { pos += 8 }; return UInt64(littleEndian: p.loadUnaligned(fromByteOffset: pos, as: UInt64.self)) }
    mutating func i64() throws -> Int { Int(Int64(bitPattern: try u64())) }
    mutating func f64() throws -> Double { Double(bitPattern: try u64()) }
    mutating func f32() throws -> Float { Float(bitPattern: try u32()) }
    mutating func bool() throws -> Bool { try u8() != 0 }
    mutating func str() throws -> String {
        let n = Int(try u32()); try need(n)
        defer { pos += n }
        return String(decoding: UnsafeRawBufferPointer(rebasing: p[pos..<(pos + n)]), as: UTF8.self)
    }
    mutating func opt<T>(_ r: (inout BinReader) throws -> T) throws -> T? { try u8() == 0 ? nil : try r(&self) }
}

enum CRC32 {
    static let table: [UInt32] = (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }
    static func update(_ crc: UInt32, _ p: UnsafeRawBufferPointer) -> UInt32 {
        var c = ~crc
        table.withUnsafeBufferPointer { t in for b in p { c = t[Int((c ^ UInt32(b)) & 0xff)] ^ (c >> 8) } }
        return ~c
    }
    static func of(_ p: UnsafeRawBufferPointer) -> UInt32 { update(0, p) }
}

/// Repeated strings (place names, camera, face model) share one copy in RAM.
struct Interner {
    var seen: [String: String] = [:]
    mutating func callAsFunction(_ s: String?) -> String? {
        guard let s else { return nil }
        if let t = seen[s] { return t }
        seen[s] = s; return s
    }
}

extension IndexRecord {
    func encode(_ w: inout BinWriter) {
        w.str(id)
        w.u8((isVideo ? 1 : 0) | (facesKnown ? 2 : 0))
        w.opt(taken) { $0.f64($1) }
        w.opt(localMinutes) { $0.i64($1) }
        w.opt(lat) { $0.f64($1) }
        w.opt(lon) { $0.f64($1) }
        w.opt(place) { $0.str($1) }
        w.opt(camera) { $0.str($1) }
        w.opt(isScreenshot) { $0.bool($1) }
        w.opt(lowRes) { $0.bool($1) }
        w.opt(faceModel) { $0.str($1) }
        w.opt(imageVersion) { $0.i64($1) }
        w.opt(faceSide) { $0.f64($1) }
        w.i64(vectorRow); w.i64(frameRow)
        w.opt(frameTs) { w, ts in w.u32(UInt32(ts.count)); for t in ts { w.f64(t) } }
        w.u32(UInt32(faces.count))
        for f in faces {
            w.u8(UInt8(f.box.count)); for b in f.box { w.f64(b) }
            w.f64(f.imageW); w.f64(f.imageH); w.f32(f.confidence)
            w.opt(f.frameT) { $0.f64($1) }
            w.i64(f.frame ?? -1); w.i64(f.row)
        }
    }

    static func decode(_ r: inout BinReader, _ intern: inout Interner) throws -> IndexRecord {
        let id = try r.str()
        let flags = try r.u8()
        let taken = try r.opt { try $0.f64() }
        let lm = try r.opt { try $0.i64() }
        let lat = try r.opt { try $0.f64() }, lon = try r.opt { try $0.f64() }
        let place = intern(try r.opt { try $0.str() }), camera = intern(try r.opt { try $0.str() })
        let shot = try r.opt { try $0.bool() }, low = try r.opt { try $0.bool() }
        let model = intern(try r.opt { try $0.str() })
        let iv = try r.opt { try $0.i64() }
        let side = try r.opt { try $0.f64() }
        let vr = try r.i64(), fr = try r.i64()
        let ts: [Double]? = try r.opt { r in
            let n = Int(try r.u32()); try r.need(n * 8)
            return try (0..<n).map { _ in try r.f64() }
        }
        let nf = Int(try r.u32()); try r.need(nf * 40)
        var faces = [StoredFace](); faces.reserveCapacity(nf)
        for _ in 0..<nf {
            let nb = Int(try r.u8())
            let box = try (0..<nb).map { _ in try r.f64() }
            let w = try r.f64(), h = try r.f64(), c = try r.f32()
            let t = try r.opt { try $0.f64() }
            let k = try r.i64(), row = try r.i64()
            faces.append(StoredFace(box: box, imageW: w, imageH: h, confidence: c, frameT: t, frame: k < 0 ? nil : k, row: row))
        }
        return IndexRecord(id: id, isVideo: flags & 1 != 0, taken: taken, localMinutes: lm, lat: lat, lon: lon, place: place,
                           camera: camera, isScreenshot: shot, lowRes: low, faceModel: model, imageVersion: iv, faceSide: side,
                           vectorRow: vr, frameRow: fr, frameTs: ts, facesKnown: flags & 2 != 0, faces: faces)
    }
}

extension ReadOutcome {
    var code: UInt8 { switch self { case .waitingForICloud: return 1; case .downloadFailed: return 2; case .unreadable: return 3 } }
    init?(code: UInt8) {
        switch code { case 1: self = .waitingForICloud; case 2: self = .downloadFailed; case 3: self = .unreadable; default: return nil }
    }
}

// MARK: - the store

/// What opening found (journaled to measure launch time and crash recovery).
public struct IndexStoreOpenStats: Sendable {
    public var seconds: Double = 0
    public var entries = 0
    public var logRecords = 0
    /// Bytes of the log past its last COMMIT that were cut off (a save killed mid-write).
    public var logBytesCut = 0
    /// Uncommitted vector rows cut off.
    public var imageRowsCut = 0, faceRowsCut = 0
}

public final class IndexStore {
    public struct Config: Equatable, Sendable {
        public var imageDim: Int
        public var faceDim: Int
        public var imageScalar: StoredScalar
        public var faceScalar: StoredScalar
        /// Rewrite the snapshot when the log grows past max(this, the snapshot's size).
        public var minLogBytesToCompact: Int
        /// Rewrite the vector files when dead rows are over this fraction of a file (and at least minGarbageRows).
        public var garbageFraction: Double
        public var minGarbageRows: Int
        public init(imageDim: Int, faceDim: Int, imageScalar: StoredScalar = .f16, faceScalar: StoredScalar = .f16,
                    minLogBytesToCompact: Int = 8 << 20, garbageFraction: Double = 0.3, minGarbageRows: Int = 20_000) {
            self.imageDim = imageDim; self.faceDim = faceDim; self.imageScalar = imageScalar; self.faceScalar = faceScalar
            self.minLogBytesToCompact = minLogBytesToCompact; self.garbageFraction = garbageFraction
            self.minGarbageRows = minGarbageRows
        }
    }

    public let dir: URL
    public let config: Config
    public private(set) var records: [String: IndexRecord] = [:]
    public private(set) var notRead: [String: ReadOutcome] = [:]
    public private(set) var openStats = IndexStoreOpenStats()

    private var generation: UInt64
    private var imageFileId: UInt64, faceFileId: UInt64
    private var image: VectorFile, face: VectorFile
    private var logFD: Int32
    private var logSize: Int
    private var snapshotBytes: Int
    private var pending = BinWriter()
    private var pendingOps = 0
    private(set) var liveImageRows = 0, liveFaceRows = 0

    static let snapMagic: [UInt8] = Array("FPIXSNP1".utf8), logMagic: [UInt8] = Array("FPIXLOG1".utf8)
    static let formatVersion: UInt32 = 1
    static let logHeaderBytes = 16
    enum Op: UInt8 { case put = 1, delete = 2, notRead = 3, commit = 4 }

    private var path: String { dir.path }
    private func file(_ name: String) -> String { dir.appendingPathComponent(name).path }
    private static func snapName(_ g: UInt64) -> String { "meta-\(g).snap" }
    private static func logName(_ g: UInt64) -> String { "log-\(g).log" }
    private static func imgName(_ v: UInt64) -> String { "img-\(v).vec" }
    private static func faceName(_ v: UInt64) -> String { "face-\(v).vec" }

    private init(dir: URL, config: Config, generation: UInt64, imageFileId: UInt64, faceFileId: UInt64, image: VectorFile,
                 face: VectorFile, logFD: Int32, logSize: Int, snapshotBytes: Int) {
        self.dir = dir; self.config = config; self.generation = generation; self.imageFileId = imageFileId
        self.faceFileId = faceFileId; self.image = image; self.face = face; self.logFD = logFD; self.logSize = logSize
        self.snapshotBytes = snapshotBytes
    }
    deinit { close(logFD) }

    // MARK: open

    /// Opens the store in `dir` (created empty if there is none), applying every committed journal record and cutting
    /// off anything a killed save left half-written.
    public static func open(dir: URL, config: Config) throws -> IndexStore {
        let t0 = Date()
        let fm = FileManager.default
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let cur = dir.appendingPathComponent("CURRENT").path
        if !Posix.exists(cur) {
            // nothing committed here yet: anything else in the folder is a creation that never finished
            for f in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? [] { try? fm.removeItem(atPath: dir.appendingPathComponent(f).path) }
            try create(dir: dir, config: config)
        }
        let curFile = try MappedFile.read(cur)
        guard let g = UInt64(String(decoding: curFile.bytes, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw IndexStoreError.corrupt("CURRENT")
        }
        // snapshot
        let snapPath = dir.appendingPathComponent(snapName(g)).path
        guard Posix.exists(snapPath) else { throw IndexStoreError.corrupt("missing " + snapPath) }
        let snap = try MappedFile.read(snapPath)
        var st = try parseSnapshot(snap.bytes, path: snapPath)
        guard st.generation == g else { throw IndexStoreError.corrupt("snapshot generation \(st.generation) != \(g)") }
        guard st.imageDim == config.imageDim, st.faceDim == config.faceDim, st.imageScalar == config.imageScalar,
              st.faceScalar == config.faceScalar else {
            throw IndexStoreError.incompatible("store \(st.imageDim)/\(st.imageScalar) faces \(st.faceDim)/\(st.faceScalar)")
        }
        // log: apply committed groups, cut the rest
        let logPath = dir.appendingPathComponent(logName(g)).path
        var stats = IndexStoreOpenStats()
        let logFD = try Posix.openRW(logPath, create: true)
        var logSize = try Posix.size(logFD, logPath)
        if logSize < logHeaderBytes {
            var w = BinWriter(); w.buf = logMagic; w.u64(g)
            try w.buf.withUnsafeBytes { try Posix.pwriteAll(logFD, $0, at: 0, logPath) }
            try Posix.truncate(logFD, logHeaderBytes, logPath)
            try Posix.sync(logFD, logPath)
            logSize = logHeaderBytes
        } else {
            let lm = try MappedFile(readingAll: logFD, length: logSize, path: logPath)
            let good = try replay(lm.bytes, generation: g, into: &st, stats: &stats, path: logPath)
            if good < logSize {
                stats.logBytesCut = logSize - good
                try Posix.truncate(logFD, good, logPath); try Posix.sync(logFD, logPath)
                logSize = good
            }
        }
        // vector files: cut rows past the last commit
        let imgPath = dir.appendingPathComponent(imgName(st.imageFileId)).path
        let facePath = dir.appendingPathComponent(faceName(st.faceFileId)).path
        let imgSize0 = (try? fm.attributesOfItem(atPath: imgPath)[.size] as? Int) ?? 0
        let faceSize0 = (try? fm.attributesOfItem(atPath: facePath)[.size] as? Int) ?? 0
        let image = try VectorFile(path: imgPath, rowBytes: config.imageDim * config.imageScalar.bytes, committedRows: st.imageRows, create: false)
        let face = try VectorFile(path: facePath, rowBytes: config.faceDim * config.faceScalar.bytes, committedRows: st.faceRows, create: false)
        stats.imageRowsCut = max(0, imgSize0 / image.rowBytes - st.imageRows)
        stats.faceRowsCut = max(0, faceSize0 / face.rowBytes - st.faceRows)
        // leftovers of a compaction that never switched over, an old generation not yet deleted, CURRENT.tmp
        let keep: Set<String> = ["CURRENT", snapName(g), logName(g), imgName(st.imageFileId), faceName(st.faceFileId)]
        for f in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? [] where !keep.contains(f) {
            try? fm.removeItem(atPath: dir.appendingPathComponent(f).path)
        }
        let s = IndexStore(dir: dir, config: config, generation: g, imageFileId: st.imageFileId, faceFileId: st.faceFileId,
                           image: image, face: face, logFD: logFD, logSize: logSize, snapshotBytes: snap.length)
        s.records = st.records; s.notRead = st.notRead
        for r in s.records.values { s.liveImageRows += r.ownedImageRows; s.liveFaceRows += r.faces.count }
        stats.entries = s.records.count
        stats.seconds = Date().timeIntervalSince(t0)
        s.openStats = stats
        return s
    }

    /// Generation 1, empty.
    private static func create(dir: URL, config: Config) throws {
        let st = SnapState(generation: 1, imageDim: config.imageDim, faceDim: config.faceDim, imageScalar: config.imageScalar,
                           faceScalar: config.faceScalar, imageFileId: 1, faceFileId: 1, imageRows: 0, faceRows: 0)
        for n in [imgName(1), faceName(1)] {
            let fd = try Posix.openRW(dir.appendingPathComponent(n).path, create: true, truncate: true)
            try Posix.sync(fd, n); close(fd)
        }
        _ = try writeSnapshot(st, records: [:], notRead: [:], to: dir.appendingPathComponent(snapName(1)).path)
        var w = BinWriter(); w.buf = logMagic; w.u64(1)
        let lp = dir.appendingPathComponent(logName(1)).path
        let fd = try Posix.openRW(lp, create: true, truncate: true)
        try w.buf.withUnsafeBytes { try Posix.pwriteAll(fd, $0, at: 0, lp) }
        try Posix.sync(fd, lp); close(fd)
        try Posix.atomicWrite(Array("1\n".utf8), to: dir.appendingPathComponent("CURRENT").path, dir: dir.path)
    }

    struct SnapState {
        var generation: UInt64
        var imageDim: Int, faceDim: Int
        var imageScalar: StoredScalar, faceScalar: StoredScalar
        var imageFileId: UInt64, faceFileId: UInt64
        var imageRows: Int, faceRows: Int
        var records: [String: IndexRecord] = [:]
        var notRead: [String: ReadOutcome] = [:]
    }

    private static func parseSnapshot(_ p: UnsafeRawBufferPointer, path: String) throws -> SnapState {
        guard p.count >= 12, Array(p[0..<8]) == snapMagic else { throw IndexStoreError.corrupt("not a snapshot: " + path) }
        let body = UnsafeRawBufferPointer(rebasing: p[0..<(p.count - 4)])
        let crc = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: p.count - 4, as: UInt32.self))
        guard CRC32.of(body) == crc else { throw IndexStoreError.corrupt("snapshot checksum: " + path) }
        var r = BinReader(body, pos: 8)
        do {
            guard try r.u32() == formatVersion else { throw IndexStoreError.incompatible("snapshot format") }
            let id = Int(try r.u32()), isc = try r.u8(), fd = Int(try r.u32()), fsc = try r.u8()
            guard let iS = StoredScalar(rawValue: isc), let fS = StoredScalar(rawValue: fsc) else { throw IndexStoreError.corrupt("scalar") }
            var st = SnapState(generation: try r.u64(), imageDim: id, faceDim: fd, imageScalar: iS, faceScalar: fS,
                               imageFileId: try r.u64(), faceFileId: try r.u64(), imageRows: Int(try r.u64()), faceRows: Int(try r.u64()))
            let n = Int(try r.u64()), m = Int(try r.u64())
            var intern = Interner()
            st.records.reserveCapacity(n)
            for _ in 0..<n { let e = try IndexRecord.decode(&r, &intern); st.records[e.id] = e }
            st.notRead.reserveCapacity(m)
            for _ in 0..<m {
                let id = try r.str()
                guard let o = ReadOutcome(code: try r.u8()) else { throw IndexStoreError.corrupt("not-read code") }
                st.notRead[id] = o
            }
            return st
        } catch is BinReader.Short { throw IndexStoreError.corrupt("snapshot truncated: " + path) }
    }

    /// Applies the log's committed groups to `st`; returns the byte offset just past the last good COMMIT.
    private static func replay(_ p: UnsafeRawBufferPointer, generation g: UInt64, into st: inout SnapState,
                               stats: inout IndexStoreOpenStats, path: String) throws -> Int {
        guard p.count >= logHeaderBytes, Array(p[0..<8]) == logMagic,
              UInt64(littleEndian: p.loadUnaligned(fromByteOffset: 8, as: UInt64.self)) == g else {
            throw IndexStoreError.corrupt("log header: " + path)
        }
        enum Pending { case put(IndexRecord), delete(String), notRead(String, ReadOutcome?) }
        var batch = [Pending](), good = logHeaderBytes, pos = logHeaderBytes
        var intern = Interner()
        while p.count - pos >= 9 {
            let len = Int(UInt32(littleEndian: p.loadUnaligned(fromByteOffset: pos, as: UInt32.self)))
            guard len >= 1, p.count - pos - 4 >= len + 4 else { break }
            let payload = UnsafeRawBufferPointer(rebasing: p[(pos + 4)..<(pos + 4 + len)])
            let crc = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: pos + 4 + len, as: UInt32.self))
            guard CRC32.of(payload) == crc else { break }
            var r = BinReader(payload, pos: 1)
            guard let op = Op(rawValue: payload[0]) else { break }
            do {
                switch op {
                case .put: batch.append(.put(try IndexRecord.decode(&r, &intern)))
                case .delete: batch.append(.delete(try r.str()))
                case .notRead: let id = try r.str(); batch.append(.notRead(id, ReadOutcome(code: try r.u8())))
                case .commit:
                    let ir = Int(try r.u64()), fr = Int(try r.u64())
                    guard ir >= st.imageRows, fr >= st.faceRows else { throw IndexStoreError.corrupt("commit rows go back") }
                    // check the whole group first: a group is applied entirely or not at all
                    for b in batch {
                        guard case .put(let e) = b else { continue }
                        let n = e.frameTs?.count ?? 0
                        let last = n > 0 ? max(e.vectorRow, e.frameRow + n - 1) : e.vectorRow
                        guard e.vectorRow >= 0, e.frameRow >= 0, last < ir, e.faces.allSatisfy({ $0.row >= 0 && $0.row < fr }) else {
                            throw IndexStoreError.corrupt("rows past commit")
                        }
                    }
                    for b in batch {
                        switch b {
                        case .put(let e): st.records[e.id] = e
                        case .delete(let id): st.records[id] = nil
                        case .notRead(let id, let o): st.notRead[id] = o
                        }
                    }
                    stats.logRecords += batch.count
                    batch = []; st.imageRows = ir; st.faceRows = fr
                    good = pos + 8 + len
                }
            } catch { break }
            pos += 8 + len
        }
        return good
    }

    @discardableResult
    private static func writeSnapshot(_ st: SnapState, records: [String: IndexRecord], notRead: [String: ReadOutcome],
                                      to path: String) throws -> Int {
        let fd = try Posix.openRW(path, create: true, truncate: true)
        defer { close(fd) }
        var w = BinWriter(); w.buf = snapMagic
        w.u32(formatVersion)
        w.u32(UInt32(st.imageDim)); w.u8(st.imageScalar.rawValue); w.u32(UInt32(st.faceDim)); w.u8(st.faceScalar.rawValue)
        w.u64(st.generation); w.u64(st.imageFileId); w.u64(st.faceFileId)
        w.u64(UInt64(st.imageRows)); w.u64(UInt64(st.faceRows))
        w.u64(UInt64(records.count)); w.u64(UInt64(notRead.count))
        var crc: UInt32 = 0, off = 0
        func flush(_ force: Bool) throws {
            guard force || w.buf.count >= 4 << 20 else { return }
            try w.buf.withUnsafeBytes { b in crc = CRC32.update(crc, b); try Posix.pwriteAll(fd, b, at: off, path) }
            off += w.buf.count; w.buf.removeAll(keepingCapacity: true)
        }
        for e in records.values { e.encode(&w); try flush(false) }
        for (id, o) in notRead { w.str(id); w.u8(o.code); try flush(false) }
        try flush(true)
        w.u32(crc)
        try w.buf.withUnsafeBytes { try Posix.pwriteAll(fd, $0, at: off, path) }
        off += 4
        try Posix.sync(fd, path)
        return off
    }

    // MARK: writes

    private func encodeRows(_ rows: [[Float]], dim: Int, scalar: StoredScalar, what: String) throws -> [UInt8] {
        var out = [UInt8](); out.reserveCapacity(rows.count * dim * scalar.bytes)
        for v in rows {
            guard v.count == dim else { throw IndexStoreError.badVector("\(what): \(v.count) values, the store has \(dim)") }
            if scalar == .f16 { for x in v { let h = halfBits(x); out.append(UInt8(h & 0xff)); out.append(UInt8(h >> 8)) } }
            else { for x in v { withUnsafeBytes(of: x.bitPattern.littleEndian) { out.append(contentsOf: $0) } } }
        }
        return out
    }

    private func appendFaces(_ list: [(DetectedFace, Int?)]) throws -> [StoredFace] {
        let bytes = try encodeRows(list.map { $0.0.embedding }, dim: config.faceDim, scalar: config.faceScalar, what: "face")
        let first = try bytes.withUnsafeBytes { try face.append($0) }
        return list.enumerated().map { k, x in
            StoredFace(box: x.0.box, imageW: x.0.imageW, imageH: x.0.imageH, confidence: x.0.confidence, frameT: x.0.frameT,
                       frame: x.1, row: first + k)
        }
    }

    private func set(_ e: IndexRecord) {
        if let old = records[e.id] { liveImageRows -= old.ownedImageRows; liveFaceRows -= old.faces.count }
        records[e.id] = e
        liveImageRows += e.ownedImageRows; liveFaceRows += e.faces.count
        pending.record(Op.put.rawValue) { e.encode(&$0) }
        pendingOps += 1
    }

    /// Adds or replaces an entry with its vectors (appended to the vector files now; durable at the next commit).
    public func put(_ e: FullIndexEntry) throws {
        var units: [[Float]], frameTs: [Double]? = nil, vectorIsFrame0 = false
        if let fr = e.frames {
            frameTs = fr.map(\.t)
            if let f0 = fr.first, f0.vector == e.vector { units = fr.map(\.vector); vectorIsFrame0 = true }
            else { units = [e.vector] + fr.map(\.vector) }
        } else { units = [e.vector] }
        let bytes = try encodeRows(units, dim: config.imageDim, scalar: config.imageScalar, what: "image")
        var faceList: [(DetectedFace, Int?)] = (e.faces ?? []).map { ($0, nil) }
        for (k, u) in (e.frames ?? []).enumerated() { faceList += u.faces.map { ($0, k) } }
        for (f, _) in faceList where f.embedding.count != config.faceDim {
            throw IndexStoreError.badVector("face: \(f.embedding.count) values, the store has \(config.faceDim)")
        }
        let first = try bytes.withUnsafeBytes { try image.append($0) }
        let faces = try appendFaces(faceList)
        set(IndexRecord(id: e.id, isVideo: e.isVideo, taken: e.taken, localMinutes: e.localMinutes, lat: e.lat, lon: e.lon,
                        place: e.place, camera: e.camera, isScreenshot: e.isScreenshot, lowRes: e.lowRes, faceModel: e.faceModel,
                        imageVersion: e.imageVersion, faceSide: e.faceSide, vectorRow: first,
                        frameRow: (frameTs == nil || vectorIsFrame0) ? first : first + 1, frameTs: frameTs,
                        facesKnown: e.faces != nil, faces: faces))
    }

    /// Replaces an entry's faces (image vectors, dates, place kept): `photo` = the photo's faces (nil: not looked for),
    /// `frames` = each sampled frame's faces (videos; nil keeps none). New fingerprints are appended.
    public func replaceFaces(_ id: String, photo: [DetectedFace]?, frames: [[DetectedFace]]?, faceModel: String?,
                             faceSide: Double?) throws {
        guard var e = records[id] else { return }
        if let fr = frames, fr.count != (e.frameTs?.count ?? 0) {
            throw IndexStoreError.badVector("\(fr.count) frames of faces for \(e.frameTs?.count ?? 0) frames")
        }
        var list: [(DetectedFace, Int?)] = (photo ?? []).map { ($0, nil) }
        for (k, fs) in (frames ?? []).enumerated() { list += fs.map { ($0, k) } }
        for (f, _) in list where f.embedding.count != config.faceDim {
            throw IndexStoreError.badVector("face: \(f.embedding.count) values, the store has \(config.faceDim)")
        }
        e.faces = try appendFaces(list)
        e.facesKnown = photo != nil
        e.faceModel = faceModel; e.faceSide = faceSide
        set(e)
    }

    /// The frames pass (LazyVideo.swift): a video's cover-frame vector and faces are replaced by its sampled frames'
    /// vectors and faces in ONE journal record (a reader never sees half of each); dates, place, camera kept. The old
    /// rows become dead rows (compaction). No-op when the entry is gone or `frames` is empty.
    public func replaceWithFrames(_ id: String, frames: [FrameUnit], faceModel: String?, imageVersion: Int?) throws {
        guard let e = records[id], !frames.isEmpty else { return }
        try put(FullIndexEntry(id: e.id, isVideo: e.isVideo, taken: e.taken, localMinutes: e.localMinutes, lat: e.lat, lon: e.lon,
                               vector: frames[0].vector, faces: nil, place: e.place, frames: frames, camera: e.camera,
                               isScreenshot: e.isScreenshot, lowRes: nil, faceModel: faceModel, imageVersion: imageVersion,
                               faceSide: nil))
    }

    /// Changes metadata only (e.g. lowRes, place); vector rows and faces must be the entry's own.
    public func update(_ e: IndexRecord) {
        guard let old = records[e.id], old.vectorRow == e.vectorRow, old.frameRow == e.frameRow, old.faces == e.faces,
              old.frameTs == e.frameTs else { return }
        if old != e { set(e) }
    }

    public func remove(_ id: String) {
        guard let old = records.removeValue(forKey: id) else { return }
        liveImageRows -= old.ownedImageRows; liveFaceRows -= old.faces.count
        pending.record(Op.delete.rawValue) { $0.str(id) }
        pendingOps += 1
    }

    public func setNotRead(_ id: String, _ why: ReadOutcome?) {
        guard notRead[id] != why else { return }
        notRead[id] = why
        pending.record(Op.notRead.rawValue) { $0.str(id); $0.u8(why?.code ?? 0) }
        pendingOps += 1
    }

    /// Changes since the last commit (they are in RAM; a kill before commit loses them, never more).
    public var uncommitted: Int { pendingOps }

    /// Makes every change so far durable: vector files fsynced, then the journal records + COMMIT appended and fsynced.
    /// Then compacts when the log or the dead rows have grown (compactIfNeeded).
    public func commit() throws {
        guard pendingOps > 0 else { return }
        try image.sync(); try face.sync()
        var w = pending
        w.record(Op.commit.rawValue) { $0.u64(UInt64(image.rows)); $0.u64(UInt64(face.rows)) }
        try w.buf.withUnsafeBytes { try Posix.pwriteAll(logFD, $0, at: logSize, file(Self.logName(generation))) }
        try Posix.sync(logFD, file(Self.logName(generation)))
        logSize += w.buf.count
        pending = BinWriter(); pendingOps = 0
        try compactIfNeeded()
    }

    public var deadImageRows: Int { image.rows - liveImageRows }
    public var deadFaceRows: Int { face.rows - liveFaceRows }
    public var logBytes: Int { logSize }

    /// Compacts if the log outgrew the snapshot or a vector file's dead rows passed Config.garbageFraction (only that
    /// file is rewritten: e.g. the face upgrade replaces most faces but no image vector).
    public func compactIfNeeded() throws {
        let gImg = deadImageRows > max(config.minGarbageRows, Int(Double(image.rows) * config.garbageFraction))
        let gFace = deadFaceRows > max(config.minGarbageRows, Int(Double(face.rows) * config.garbageFraction))
        if gImg || gFace || logSize > max(config.minLogBytesToCompact, snapshotBytes) {
            try compact(rewriteImages: gImg, rewriteFaces: gFace)
        }
    }

    /// compact(rewriteImages: v, rewriteFaces: v).
    public func compact(rewriteVectors v: Bool) throws { try compact(rewriteImages: v, rewriteFaces: v) }

    /// A new generation: snapshot of everything (log emptied); a rewritten vector file holds live rows only, in the old
    /// order. Commits pending changes first. Old files are deleted after the switch (readers holding a StoredRows keep
    /// their mapping of the old file until they let go).
    public func compact(rewriteImages: Bool, rewriteFaces: Bool) throws {
        if pendingOps > 0 {      // commit without recursing into compactIfNeeded
            try image.sync(); try face.sync()
            var w = pending
            w.record(Op.commit.rawValue) { $0.u64(UInt64(image.rows)); $0.u64(UInt64(face.rows)) }
            try w.buf.withUnsafeBytes { try Posix.pwriteAll(logFD, $0, at: logSize, file(Self.logName(generation))) }
            try Posix.sync(logFD, file(Self.logName(generation)))
            logSize += w.buf.count; pending = BinWriter(); pendingOps = 0
        }
        let g = generation + 1
        var recs = records
        let imgId = rewriteImages ? g : imageFileId, faceId = rewriteFaces ? g : faceFileId
        let ni = rewriteImages ? try VectorFile(path: file(Self.imgName(g)), rowBytes: image.rowBytes, committedRows: 0, create: true) : nil
        let nf = rewriteFaces ? try VectorFile(path: file(Self.faceName(g)), rowBytes: face.rowBytes, committedRows: 0, create: true) : nil
        if ni != nil || nf != nil {
            let im = try image.mapping(), fm = try face.mapping()
            var ibuf = [UInt8](), fbuf = [UInt8]()
            var nextImg = 0, nextFace = 0
            func copy(_ m: MappedFile?, _ row: Int, _ rb: Int, into buf: inout [UInt8]) {
                buf.append(contentsOf: UnsafeRawBufferPointer(start: m!.base! + row * rb, count: rb))
            }
            func flush(_ f: VectorFile?, _ buf: inout [UInt8]) throws {
                if let f { try buf.withUnsafeBytes { _ = try f.append($0) } }
                buf.removeAll(keepingCapacity: true)
            }
            for e0 in records.values.sorted(by: { $0.vectorRow < $1.vectorRow }) {   // keeps the file order
                var e = e0
                if ni != nil {
                    let nts = e.frameTs?.count ?? 0
                    let sep = e.frameTs != nil && e.frameRow != e.vectorRow
                    copy(im, e.vectorRow, image.rowBytes, into: &ibuf)
                    let newVector = nextImg; nextImg += 1
                    var newFrame = newVector
                    if sep { newFrame = nextImg; for k in 0..<nts { copy(im, e.frameRow + k, image.rowBytes, into: &ibuf) }; nextImg += nts }
                    else if nts > 1 { for k in 1..<nts { copy(im, e.frameRow + k, image.rowBytes, into: &ibuf) }; nextImg += nts - 1 }
                    e.vectorRow = newVector; e.frameRow = newFrame
                    if ibuf.count >= 4 << 20 { try flush(ni, &ibuf) }
                }
                if nf != nil {
                    for k in e.faces.indices { copy(fm, e.faces[k].row, face.rowBytes, into: &fbuf); e.faces[k].row = nextFace; nextFace += 1 }
                    if fbuf.count >= 4 << 20 { try flush(nf, &fbuf) }
                }
                recs[e.id] = e
            }
            try flush(ni, &ibuf); try flush(nf, &fbuf)
            try ni?.sync(); try nf?.sync()
        }
        let st = SnapState(generation: g, imageDim: config.imageDim, faceDim: config.faceDim, imageScalar: config.imageScalar,
                           faceScalar: config.faceScalar, imageFileId: imgId, faceFileId: faceId,
                           imageRows: ni?.rows ?? image.rows, faceRows: nf?.rows ?? face.rows)
        let snapBytes = try Self.writeSnapshot(st, records: recs, notRead: notRead, to: file(Self.snapName(g)))
        let lp = file(Self.logName(g))
        let lfd = try Posix.openRW(lp, create: true, truncate: true)
        var w = BinWriter(); w.buf = Self.logMagic; w.u64(g)
        do {
            try w.buf.withUnsafeBytes { try Posix.pwriteAll(lfd, $0, at: 0, lp) }
            try Posix.sync(lfd, lp)
            Posix.syncDir(path)
            // the switch: before this rename the old generation is the store; after it, the new one
            try Posix.atomicWrite(Array("\(g)\n".utf8), to: file("CURRENT"), dir: path)
        } catch { close(lfd); throw error }
        let oldSnap = file(Self.snapName(generation)), oldLog = file(Self.logName(generation))
        let oldImg = file(Self.imgName(imageFileId)), oldFace = file(Self.faceName(faceFileId))
        close(logFD)
        logFD = lfd; logSize = Self.logHeaderBytes; snapshotBytes = snapBytes; generation = g
        unlink(oldSnap); unlink(oldLog)
        records = recs
        if let ni { image = ni; imageFileId = imgId; unlink(oldImg); liveImageRows = ni.rows }
        if let nf { face = nf; faceFileId = faceId; unlink(oldFace); liveFaceRows = nf.rows }
    }

    // MARK: reads

    /// Image-vector units of these items: one per photo, one per sampled video frame (unitItem = position in `ids`,
    /// unitT = frame time), as mapped rows (nothing copied).
    public func units(ids: [String]) throws -> (vectors: StoredRows, unitItem: [Int], unitT: [Double?]) {
        var rows = [Int32](), ui = [Int](), ut = [Double?]()
        rows.reserveCapacity(ids.count); ui.reserveCapacity(ids.count); ut.reserveCapacity(ids.count)
        for (k, id) in ids.enumerated() {
            guard let e = records[id] else { continue }
            if let ts = e.frameTs { for (j, t) in ts.enumerated() { rows.append(Int32(e.frameRow + j)); ui.append(k); ut.append(t) } }
            else { rows.append(Int32(e.vectorRow)); ui.append(k); ut.append(nil) }
        }
        return (StoredRows(file: try image.mapping(), scalar: config.imageScalar, dim: config.imageDim, physical: rows), ui, ut)
    }

    /// Face fingerprints at these face-file rows.
    public func faceRows(_ rows: [Int]) throws -> StoredRows {
        StoredRows(file: try face.mapping(), scalar: config.faceScalar, dim: config.faceDim, physical: rows.map { Int32($0) })
    }

    /// Every face people searches use, of face model `model` (IndexRecord.searchFaces; nil faceModel = legacy).
    public func allFaces(model: String, legacyModel: String = legacyFaceModel) throws -> LibraryFaces {
        var rows = [Int](), item = [String](), px = [Float](), det = [Float](), box = [StoredFace](), taken = [Double?]()
        for (id, e) in records where e.hasFaces && (e.faceModel ?? legacyModel) == model {
            for f in e.searchFaces {
                rows.append(f.row); item.append(id); px.append(Float(f.px)); det.append(f.confidence); box.append(f); taken.append(e.taken)
            }
        }
        return LibraryFaces(emb: try faceRows(rows), item: item, px: px, det: det, box: box, taken: taken)
    }

    /// The entry's image vector (Float32 copy).
    public func vector(_ id: String) throws -> [Float]? {
        guard let e = records[id] else { return nil }
        return StoredRows(file: try image.mapping(), scalar: config.imageScalar, dim: config.imageDim, physical: [Int32(e.vectorRow)])[0]
    }

    /// These faces with their fingerprints (Float32 copies).
    public func detectedFaces(_ fs: [StoredFace]) throws -> [DetectedFace] {
        let v = try faceRows(fs.map(\.row))
        return fs.enumerated().map { k, f in
            DetectedFace(box: f.box, imageW: f.imageW, imageH: f.imageH, confidence: f.confidence, embedding: v[k], frameT: f.frameT)
        }
    }

    /// The entry as it was handed in (vectors at the store's precision).
    public func full(_ id: String) throws -> FullIndexEntry? {
        guard let e = records[id] else { return nil }
        let img = StoredRows(file: try image.mapping(), scalar: config.imageScalar, dim: config.imageDim, physical: [Int32(e.vectorRow)])
        var frames: [FrameUnit]? = nil
        if let ts = e.frameTs {
            let fv = StoredRows(file: try image.mapping(), scalar: config.imageScalar, dim: config.imageDim,
                                physical: ts.indices.map { Int32(e.frameRow + $0) })
            frames = try ts.enumerated().map { k, t in
                FrameUnit(t: t, vector: fv[k], faces: try detectedFaces(e.faces.filter { $0.frame == k }))
            }
        }
        return FullIndexEntry(id: e.id, isVideo: e.isVideo, taken: e.taken, localMinutes: e.localMinutes, lat: e.lat, lon: e.lon,
                              vector: img[0], faces: e.facesKnown ? try detectedFaces(e.faces.filter { $0.frame == nil }) : nil,
                              place: e.place, frames: frames, camera: e.camera, isScreenshot: e.isScreenshot, lowRes: e.lowRes,
                              faceModel: e.faceModel, imageVersion: e.imageVersion, faceSide: e.faceSide)
    }

    /// Bytes on disk (all files of the live generation).
    public var diskBytes: Int {
        image.rows * image.rowBytes + face.rows * face.rowBytes + logSize + snapshotBytes
    }
    public var imageRows: Int { image.rows }
    public var faceRowCount: Int { face.rows }
}

// MARK: - migration from index.json

public struct IndexMigrationReport: Sendable, Equatable {
    public var entries = 0
    /// Objects in index.json that could not be decoded (their photos are simply indexed again).
    public var skipped = 0
    /// Ids that appeared twice (the first one is kept, as the old loader did).
    public var duplicates = 0
    public var notRead = 0
    /// index.json ended early (cut off): the entries read until then are kept.
    public var truncated = false
    public var jsonBytes = 0
    public var seconds: Double = 0
}

extension IndexStore {
    /// Opens the store in `dir`; on the first launch of this build, converts the old index.json / not_read.json into it
    /// once (streamed: one entry decoded at a time from the mapped file; `progress` 0...1 by bytes read), switches to it
    /// atomically (the converted store is built in "<dir>.migrating" and renamed), then deletes the old JSON files
    /// (app-private derived data, never photos). A conversion killed half-way is simply redone on the next launch.
    public static func openMigrating(dir: URL, legacyIndex: URL, legacyNotRead: URL, config: Config,
                                     progress: (Double) -> Void = { _ in }) throws -> (IndexStore, IndexMigrationReport?) {
        let fm = FileManager.default
        let hasLegacy = Posix.exists(legacyIndex.path) || Posix.exists(legacyNotRead.path)
        if Posix.exists(dir.appendingPathComponent("CURRENT").path) {
            let s = try open(dir: dir, config: config)
            if hasLegacy { try? fm.removeItem(at: legacyIndex); try? fm.removeItem(at: legacyNotRead) }   // converted before
            return (s, nil)
        }
        guard hasLegacy else { return (try open(dir: dir, config: config), nil) }
        let t0 = Date()
        let tmp = URL(fileURLWithPath: dir.path + ".migrating")
        try? fm.removeItem(at: tmp)
        var rep = IndexMigrationReport()
        do {
            let s = try open(dir: tmp, config: config)
            if Posix.exists(legacyIndex.path) {
                let m = try MappedFile.map(legacyIndex.path)
                rep.jsonBytes = m.length
                let dec = JSONDecoder()
                var lastTick = 0
                var entries = 0, skipped = 0, duplicates = 0
                let complete = try forEachJSONObject(m.bytes) { obj, end in
                    if let e = try? dec.decode(FullIndexEntry.self, from: Data(obj)) {
                        if s.records[e.id] != nil { duplicates += 1 }
                        else {
                            do { try s.put(e); entries += 1 } catch IndexStoreError.badVector(_) { skipped += 1 }
                        }
                    } else { skipped += 1 }
                    if s.uncommitted >= 2000 { try s.commit() }
                    if end - lastTick >= (1 << 20) { lastTick = end; progress(Double(end) / Double(max(m.length, 1))) }
                }
                rep.truncated = !complete; rep.entries = entries; rep.skipped = skipped; rep.duplicates = duplicates
            }
            if Posix.exists(legacyNotRead.path) {
                let d = try MappedFile.read(legacyNotRead.path)
                if let n = try? JSONDecoder().decode([String: ReadOutcome].self, from: Data(d.bytes)) {
                    for (id, o) in n { s.setNotRead(id, o) }
                    rep.notRead = n.count
                }
            }
            try s.commit()
            try s.compact(rewriteVectors: false)       // launch then reads one snapshot, not a long log
        }
        progress(1)
        if Posix.exists(dir.path) { try fm.removeItem(at: dir) }      // a folder without CURRENT: nothing committed
        guard rename(tmp.path, dir.path) == 0 else { throw IndexStoreError.io("rename " + tmp.path, errno) }
        Posix.syncDir(dir.deletingLastPathComponent().path)
        try? fm.removeItem(at: legacyIndex); try? fm.removeItem(at: legacyNotRead)
        rep.seconds = Date().timeIntervalSince(t0)
        return (try open(dir: dir, config: config), rep)
    }

    /// Calls `body` with each top-level object of a JSON array (its bytes and the offset just past it), without parsing
    /// the whole array. Returns false if the array ends early (a cut-off file); objects before that are delivered.
    static func forEachJSONObject(_ p: UnsafeRawBufferPointer, _ body: (UnsafeRawBufferPointer, Int) throws -> Void) throws -> Bool {
        var i = 0
        func ws() { while i < p.count, p[i] == 0x20 || p[i] == 0x0a || p[i] == 0x0d || p[i] == 0x09 { i += 1 } }
        ws()
        guard i < p.count, p[i] == UInt8(ascii: "[") else { return p.count == 0 }
        i += 1
        while true {
            ws()
            guard i < p.count else { return false }
            if p[i] == UInt8(ascii: "]") { return true }
            if p[i] == UInt8(ascii: ",") { i += 1; continue }
            guard p[i] == UInt8(ascii: "{") else { return false }
            let start = i
            var depth = 0, inStr = false, esc = false
            while i < p.count {
                let c = p[i]
                if inStr {
                    if esc { esc = false } else if c == 0x5c { esc = true } else if c == 0x22 { inStr = false }
                } else if c == 0x22 { inStr = true }
                else if c == 0x7b || c == 0x5b { depth += 1 }
                else if c == 0x7d || c == 0x5d { depth -= 1; if depth == 0 { i += 1; break } }
                i += 1
            }
            guard depth == 0 else { return false }
            try body(UnsafeRawBufferPointer(rebasing: p[start..<i]), i)
        }
    }
}
