// The binary index store (IndexStore.swift): round trip, append / replace / delete / reopen, a save killed mid-write,
// compaction, migration from the old index.json, parity of the blocked search / face math with the [[Float]] path,
// and a synthetic whole-library load benchmark (FP_STORE_BENCH=1: 187k entries; default 10k).
import Foundation
@testable import FindPicsCore
import XCTest

/// Scratch folders: inside the repo's .cache on Linux (the cluster rule: nothing outside the project), temp elsewhere.
func storeTestDir(_ name: String) -> URL {
    #if os(Linux)
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent(".cache/storetests")
    #else
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("findpics-storetests")
    #endif
    let d = root.appendingPathComponent(name + "-" + UUID().uuidString)
    try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
    return d
}

struct LCG {
    var s: UInt64
    mutating func next() -> Float {
        s = s &* 6364136223846793005 &+ 1442695040888963407
        return Float(Int64(bitPattern: s >> 11) % 2_000_001 - 1_000_000) / 1_000_000
    }
    mutating func unit(_ d: Int) -> [Float] {
        let v = (0..<d).map { _ in next() }
        let n = sqrt(v.reduce(0) { $0 + $1 * $1 }) + 1e-9
        return v.map { $0 / n }
    }
}

func normed(_ v: [Float]) -> [Float] { let n = sqrt(v.reduce(0) { $0 + $1 * $1 }); return v.map { $0 / n } }

func f16(_ v: [Float]) -> [Float] { v.map { halfToFloatScalar(halfBits($0)) } }

/// The entry as a Float16 store gives it back.
func rounded(_ e: FullIndexEntry, faces16: Bool = true) -> FullIndexEntry {
    var x = e
    x.vector = f16(e.vector)
    func rf(_ f: DetectedFace) -> DetectedFace {
        DetectedFace(box: f.box, imageW: f.imageW, imageH: f.imageH, confidence: f.confidence,
                     embedding: faces16 ? f16(f.embedding) : f.embedding, frameT: f.frameT)
    }
    x.faces = e.faces.map { $0.map(rf) }
    x.frames = e.frames.map { $0.map { FrameUnit(t: $0.t, vector: f16($0.vector), faces: $0.faces.map(rf)) } }
    return x
}

final class IndexStoreTests: XCTestCase {
    let cfg = IndexStore.Config(imageDim: 8, faceDim: 4)
    var rng = LCG(s: 7)

    func face(_ t: Double? = nil) -> DetectedFace {
        DetectedFace(box: [10.5, 20.25, 110, 140], imageW: 448, imageH: 336, confidence: 0.93, embedding: rng.unit(4), frameT: t)
    }
    func photo(_ id: String, faces: [DetectedFace]? = nil) -> FullIndexEntry {
        FullIndexEntry(id: id, isVideo: false, taken: 1.7e9 + Double(id.count), localMinutes: 615, lat: 48.85, lon: 2.35,
                       vector: rng.unit(8), faces: faces, place: "Paris, Ile-de-France, FR, France", camera: "front",
                       isScreenshot: false, lowRes: nil, faceModel: "auraface_flip", imageVersion: 2, faceSide: 448)
    }
    func video(_ id: String, frames n: Int, vectorIsFrame0: Bool = true) -> FullIndexEntry {
        let fr = (0..<n).map { k in FrameUnit(t: Double(k) * 1.5, vector: rng.unit(8), faces: k == 1 ? [face(), face()] : []) }
        return FullIndexEntry(id: id, isVideo: true, taken: nil, localMinutes: nil, lat: nil, lon: nil,
                              vector: vectorIsFrame0 && n > 0 ? fr[0].vector : rng.unit(8), frames: fr, camera: nil,
                              isScreenshot: nil, lowRes: true, faceModel: nil)
    }

    func testRoundTrip() throws {
        for faceScalar in [StoredScalar.f16, .f32] {
            var c = cfg; c.faceScalar = faceScalar
            let dir = storeTestDir("roundtrip")
            let all = [photo("p1", faces: [face(), face()]), photo("p2", faces: []), photo("p3", faces: nil),
                       video("v1", frames: 3), video("v2", frames: 2, vectorIsFrame0: false), video("v3", frames: 0)]
            do {
                let s = try IndexStore.open(dir: dir, config: c)
                for e in all { try s.put(e) }
                try s.setNotRead("gone", .waitingForICloud)
                try s.commit()
                for e in all { XCTAssertEqual(try s.full(e.id), rounded(e, faces16: faceScalar == .f16)) }   // before reopen
            }
            let s = try IndexStore.open(dir: dir, config: c)
            XCTAssertEqual(s.records.count, all.count)
            for e in all { XCTAssertEqual(try s.full(e.id), rounded(e, faces16: faceScalar == .f16), e.id) }
            XCTAssertEqual(s.notRead, ["gone": .waitingForICloud])
            // what the app reads off records
            XCTAssertTrue(s.records["p1"]!.hasFaces); XCTAssertTrue(s.records["p1"]!.hasPhotoFaces)
            XCTAssertFalse(s.records["p2"]!.hasFaces); XCTAssertTrue(s.records["p2"]!.facesKnown)
            XCTAssertFalse(s.records["p3"]!.facesKnown)
            XCTAssertTrue(s.records["v1"]!.hasFaces); XCTAssertFalse(s.records["v1"]!.hasPhotoFaces)
            XCTAssertEqual(s.records["v1"]!.searchFaces.map(\.frameT), [1.5, 1.5])
            // units as the old PhotoIndex.units listed them: a photo's vector; a video's frames (none for v3)
            let u = try s.units(ids: ["p1", "v1", "v2", "v3", "missing"])
            XCTAssertEqual(u.unitItem, [0, 1, 1, 1, 2, 2])
            XCTAssertEqual(u.unitT, [nil, 0, 1.5, 3, 0, 1.5])
            XCTAssertEqual(u.vectors[2], f16(all[3].frames![1].vector))
            XCTAssertEqual(u.vectors[4], f16(all[4].frames![0].vector))
            XCTAssertEqual(try s.vector("v2"), f16(all[4].vector))
            XCTAssertEqual(try s.vector("v3"), f16(all[5].vector))
        }
    }

    func testAppendReplaceDeleteReopen() throws {
        let dir = storeTestDir("ard")
        let a = photo("a", faces: [face()]), b = photo("b"), c = video("c", frames: 2)
        let b2 = photo("b", faces: [face(), face()])
        let newFaces = [face(), face(), face()]
        do {
            let s = try IndexStore.open(dir: dir, config: cfg)
            for e in [a, b, c] { try s.put(e) }
            try s.commit()
            try s.put(b2)                                                                 // re-indexed
            try s.replaceFaces("a", photo: newFaces, frames: nil, faceModel: "x", faceSide: 1280)   // face upgrade
            s.remove("c")
            s.setNotRead("c", nil); s.setNotRead("d", .downloadFailed)
            try s.commit()
            XCTAssertEqual(s.deadImageRows, 1 + 2)          // b's first vector, c's two frames
            XCTAssertEqual(s.deadFaceRows, 1 + 2)           // a's first face, c's frame faces
        }
        let s = try IndexStore.open(dir: dir, config: cfg)
        XCTAssertEqual(Set(s.records.keys), ["a", "b"])
        XCTAssertEqual(try s.full("b"), rounded(b2))
        var a2 = a; a2.faces = newFaces; a2.faceModel = "x"; a2.faceSide = 1280
        XCTAssertEqual(try s.full("a"), rounded(a2))
        XCTAssertEqual(s.notRead, ["d": .downloadFailed])
        XCTAssertEqual(s.deadImageRows, 3); XCTAssertEqual(s.deadFaceRows, 3)
        // appends after a reopen continue the files
        try s.put(photo("e")); try s.commit()
        let s2 = try IndexStore.open(dir: dir, config: cfg)
        XCTAssertEqual(Set(s2.records.keys), ["a", "b", "e"])
        XCTAssertEqual(try s2.full("b"), rounded(b2))
    }

    /// A kill at any point of a save: the store reopens with exactly the last committed state.
    func testCrashMidWrite() throws {
        let dir = storeTestDir("crash")
        let a = photo("a", faces: [face()]), b = photo("b", faces: [face()])
        do {
            let s = try IndexStore.open(dir: dir, config: cfg)
            try s.put(a); try s.commit()
            try s.put(b)                         // appended, never committed: the app was killed here
        }
        var s = try IndexStore.open(dir: dir, config: cfg)
        XCTAssertEqual(Set(s.records.keys), ["a"])
        XCTAssertEqual(s.openStats.imageRowsCut, 1); XCTAssertEqual(s.openStats.faceRowsCut, 1)
        XCTAssertEqual(try s.full("a"), rounded(a))
        // killed while the journal group was being written: every cut length of the last group
        try s.put(b); try s.setNotRead("z", .unreadable); try s.commit()
        let log = dir.appendingPathComponent("log-1.log")
        let full = try Data(contentsOf: log)
        let vecs = try ["img-1.vec", "face-1.vec"].map { ($0, try Data(contentsOf: dir.appendingPathComponent($0))) }
        s = try IndexStore.open(dir: dir, config: cfg)
        XCTAssertEqual(Set(s.records.keys), ["a", "b"])
        let groupStart = try XCTUnwrap(lastCommitGroupStart(full))
        for cut in stride(from: groupStart, to: full.count, by: 7) {
            try full.prefix(cut).write(to: log)
            let r = try IndexStore.open(dir: dir, config: cfg)
            XCTAssertEqual(Set(r.records.keys), ["a"], "cut at \(cut)")
            XCTAssertEqual(r.notRead, [:])
            XCTAssertEqual(try r.full("a"), rounded(a))
        }
        // garbage after the last commit (a torn write): ignored and cut off (files as they were before the cuts above)
        for (n, d) in vecs { try d.write(to: dir.appendingPathComponent(n)) }
        try (full + Data([0x13, 0, 0, 0, 1, 2, 3])).write(to: log)
        s = try IndexStore.open(dir: dir, config: cfg)
        XCTAssertEqual(Set(s.records.keys), ["a", "b"]); XCTAssertEqual(s.openStats.logBytesCut, 7)
        // a compaction killed before the CURRENT switch: its files are ignored and deleted
        for n in ["meta-2.snap", "log-2.log", "img-2.vec", "face-2.vec", "CURRENT.tmp"] {
            try Data([1, 2, 3]).write(to: dir.appendingPathComponent(n))
        }
        s = try IndexStore.open(dir: dir, config: cfg)
        XCTAssertEqual(Set(s.records.keys), ["a", "b"])
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: dir.path)),
                       ["CURRENT", "meta-1.snap", "log-1.log", "img-1.vec", "face-1.vec"])
        // a damaged snapshot is reported, never half-loaded
        let snap = dir.appendingPathComponent("meta-1.snap")
        var d = try Data(contentsOf: snap); d[d.count / 2] ^= 0xff; try d.write(to: snap)
        XCTAssertThrowsError(try IndexStore.open(dir: dir, config: cfg)) { e in
            guard case IndexStoreError.corrupt = e else { return XCTFail("\(e)") }
        }
    }

    /// Byte offset where the last COMMIT group of a log starts (just after the previous COMMIT record).
    func lastCommitGroupStart(_ d: Data) -> Int? {
        var pos = 16, commits = [16]
        let b = [UInt8](d)
        while pos + 9 <= b.count {
            let len = Int(b[pos]) | Int(b[pos + 1]) << 8 | Int(b[pos + 2]) << 16 | Int(b[pos + 3]) << 24
            if b[pos + 4] == 4 { commits.append(pos + 8 + len) }
            pos += 8 + len
        }
        return commits.count >= 2 ? commits[commits.count - 2] : nil
    }

    func testCompaction() throws {
        let dir = storeTestDir("compact")
        var c = cfg; c.minGarbageRows = 0; c.garbageFraction = 0.3; c.minLogBytesToCompact = 1 << 30
        var latest = [String: FullIndexEntry]()
        let s = try IndexStore.open(dir: dir, config: c)
        for k in 0..<10 { let e = k % 3 == 0 ? video("e\(k)", frames: 3, vectorIsFrame0: k != 3) : photo("e\(k)", faces: [face()]); latest[e.id] = e; try s.put(e) }
        try s.commit()
        let before = try s.units(ids: ["e1"]).vectors          // a reader's view from before the compaction
        let e1v = before[0]
        for k in 0..<5 { let e = photo("e\(k)", faces: [face(), face()]); latest[e.id] = e; try s.put(e) }
        s.remove("e9"); latest["e9"] = nil
        try s.commit()                                          // dead rows > 30%: vectors rewritten
        XCTAssertEqual(s.deadImageRows, 0); XCTAssertEqual(s.deadFaceRows, 0)
        XCTAssertEqual(before[0], e1v)                          // the old mapping still reads the old rows
        let names = Set(try FileManager.default.contentsOfDirectory(atPath: dir.path))
        XCTAssertEqual(names, ["CURRENT", "meta-2.snap", "log-2.log", "img-2.vec", "face-2.vec"])
        for (id, e) in latest { XCTAssertEqual(try s.full(id), rounded(e)) }
        try s.put(photo("new")); latest["new"] = try s.full("new").map { _ in photo("x") }
        try s.commit()
        let r = try IndexStore.open(dir: dir, config: c)
        for (id, e) in latest where id != "new" { XCTAssertEqual(try r.full(id), rounded(e)) }
        XCTAssertNotNil(try r.full("new"))
        // snapshot-only compaction when the log outgrows the snapshot
        var c2 = c; c2.minLogBytesToCompact = 0; c2.minGarbageRows = 1 << 30
        let s2 = try IndexStore.open(dir: dir, config: c2)
        s2.setNotRead("q", .waitingForICloud); try s2.commit()
        XCTAssertGreaterThan(s2.logBytes, 16)                   // a small log stays a log
        for k in 0..<300 { s2.setNotRead("n\(k)", .downloadFailed) }
        try s2.commit()                                         // now bigger than the snapshot: folded into a new one
        XCTAssertEqual(s2.logBytes, 16)
        let r2 = try IndexStore.open(dir: dir, config: c2)
        XCTAssertEqual(r2.notRead["q"], .waitingForICloud); XCTAssertEqual(r2.notRead.count, 301)
        XCTAssertEqual(r2.records.count, latest.count)
    }

    func testMigrationFromJSON() throws {
        let base = storeTestDir("migrate")
        let dir = base.appendingPathComponent("index_store")
        let ij = base.appendingPathComponent("index.json"), nj = base.appendingPathComponent("not_read.json")
        let entries = [photo("p1", faces: [face()]), photo("p2", faces: nil), video("v1", frames: 3), photo("p1", faces: [])]
        var text = String(decoding: try JSONEncoder().encode(entries), as: UTF8.self)
        // one damaged object ({"id": 5} does not decode) and a string holding brackets / quotes
        text = text.replacingOccurrences(of: "[{", with: "[{\"id\":5,\"note\":\"}]\\\"{[\"},{", options: [], range: text.range(of: "[{"))
        try Data(text.utf8).write(to: ij)
        try JSONEncoder().encode(["x": ReadOutcome.waitingForICloud, "y": .unreadable]).write(to: nj)
        // leftovers of a conversion killed before: redone from scratch
        try FileManager.default.createDirectory(at: URL(fileURLWithPath: dir.path + ".migrating"), withIntermediateDirectories: true)
        try Data([9]).write(to: URL(fileURLWithPath: dir.path + ".migrating/CURRENT"))
        var ticks = [Double]()
        let (s, rep) = try IndexStore.openMigrating(dir: dir, legacyIndex: ij, legacyNotRead: nj,
                                                    config: IndexStore.Config(imageDim: 8, faceDim: 4)) { ticks.append($0) }
        let r = try XCTUnwrap(rep)
        XCTAssertEqual(r.entries, 3); XCTAssertEqual(r.duplicates, 1); XCTAssertEqual(r.skipped, 1); XCTAssertEqual(r.notRead, 2)
        XCTAssertFalse(r.truncated)
        XCTAssertEqual(ticks.last, 1)
        XCTAssertEqual(try s.full("p1"), rounded(entries[0]))          // the first p1, as the old loader kept
        XCTAssertEqual(try s.full("v1"), rounded(entries[2]))
        XCTAssertEqual(s.notRead, ["x": .waitingForICloud, "y": .unreadable])
        XCTAssertFalse(FileManager.default.fileExists(atPath: ij.path)); XCTAssertFalse(FileManager.default.fileExists(atPath: nj.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.path + ".migrating"))
        let (s2, rep2) = try IndexStore.openMigrating(dir: dir, legacyIndex: ij, legacyNotRead: nj, config: IndexStore.Config(imageDim: 8, faceDim: 4))
        XCTAssertNil(rep2); XCTAssertEqual(s2.records.count, 3)
        // a cut-off index.json: the entries before the cut are kept
        let base2 = storeTestDir("migrate-cut")
        let full = try JSONEncoder().encode([photo("a"), photo("b"), photo("c")])
        try full.prefix(full.count - 20).write(to: base2.appendingPathComponent("index.json"))   // inside the last object
        let (s3, rep3) = try IndexStore.openMigrating(dir: base2.appendingPathComponent("index_store"),
                                                      legacyIndex: base2.appendingPathComponent("index.json"),
                                                      legacyNotRead: base2.appendingPathComponent("not_read.json"), config: IndexStore.Config(imageDim: 8, faceDim: 4))
        XCTAssertEqual(Set(s3.records.keys), ["a", "b"]); XCTAssertEqual(rep3?.truncated, true)
    }

    func testHalfConversionMatchesFloat16() {
        var r = LCG(s: 3)
        var xs: [Float] = [0, -0, 1, -1, 65504, 65520, 1e-8, 6.1e-5, 5.96e-8, 2.98e-8, .infinity, -.infinity, 0.33333334, 1e5]
        for _ in 0..<20000 { xs.append(r.next() * pow(2, Float(Int(r.next() * 20)))) }
        for x in xs {
            let h = halfBits(x)
            #if !(os(macOS) && arch(x86_64))
            XCTAssertEqual(h, Float16(x).bitPattern, "\(x)")
            XCTAssertEqual(halfToFloatScalar(h), Float(Float16(bitPattern: h)))
            #endif
            var out: Float = 0
            withUnsafeBytes(of: h) { MatrixMath.halfToFloat($0.baseAddress!, &out, 1) }
            XCTAssertEqual(out.bitPattern, halfToFloatScalar(h).bitPattern)
        }
    }

    /// The blocked search / face math on store rows gives the [[Float]] results: identical for Float32 rows (same sums
    /// in the same order off Apple), within Float16 rounding for Float16 rows, with the same rankings.
    func testParityWithInMemoryPath() throws {
        let d = 64, n = 1200
        var r = LCG(s: 11)
        let lib = (0..<n).map { _ in r.unit(d) }
        let unitItem = (0..<n).map { $0 < 1000 ? $0 : 1000 + ($0 - 1000) / 5 }     // 40 "videos" of 5 frames
        let nItems = 1040
        let looks = [r.unit(d), r.unit(d)], avoid = [r.unit(d)]
        for sc in [StoredScalar.f32, .f16] {
            let dir = storeTestDir("parity")
            let s = try IndexStore.open(dir: dir, config: IndexStore.Config(imageDim: d, faceDim: d, imageScalar: sc, faceScalar: sc))
            for (k, v) in lib.enumerated() {
                try s.put(FullIndexEntry(id: String(format: "u%05d", k), isVideo: false, taken: nil, localMinutes: nil, lat: nil,
                                         lon: nil, vector: v, faces: [DetectedFace(box: [0, 0, 50, 50], imageW: 100, imageH: 100, confidence: 0.9, embedding: v)]))
            }
            try s.commit()
            let ids = (0..<n).map { String(format: "u%05d", $0) }
            let rows = try s.units(ids: ids).vectors
            let ref = sc == .f32 ? lib : lib.map(f16)
            // look scores
            let a = lookScores(units: lib, unitItem: unitItem, nItems: nItems, looks: looks, avoid: avoid)
            let b = lookScores(units: rows, unitItem: unitItem, nItems: nItems, looks: looks, avoid: avoid)
            let bRef = lookScores(units: ref, unitItem: unitItem, nItems: nItems, looks: looks, avoid: avoid)
            #if !canImport(Accelerate)
            XCTAssertEqual(b, bRef)                                   // the store path IS the in-memory path on the same values
            #endif
            let maxDiff = zip(a, b).map { abs($0 - $1) }.max()!
            XCTAssertLessThan(maxDiff, sc == .f32 ? 1e-5 : 3e-3)
            XCTAssertEqual(topOverlap(a, b, 50), sc == .f32 ? 50 : 48, accuracy: sc == .f32 ? 0 : 2)
            // subject scores: blocked (small blocks to exercise the edges) vs the reference
            let refs = [lib[17], r.unit(d)]
            // (against the reference on the SAME stored values: with Float16 rows, near-tied neighbours of random 64-d
            // vectors can swap, which moves that unit's smoothed score; eval/f16_store_effect.py measures real vectors)
            let sa = subjectScores(units: ref, unitItem: unitItem, nItems: nItems, refs: refs)
            let sb = subjectScoresBlocked(units: rows, unitItem: unitItem, nItems: nItems, refs: refs, queryBlock: 300, libraryBlock: 700)
            XCTAssertLessThan(zip(sa, sb).map { abs($0 - $1) }.max()!, 1e-4)
            XCTAssertGreaterThanOrEqual(topOverlap(sa, sb, 20), 19)
            // faces: matchPerson + groups + locateRefs on store rows
            let faces = try s.allFaces(model: legacyFaceModel)
            let order = faces.item.map { Int($0.dropFirst())! }
            let memFaces = order.map { lib[$0] }
            let fItem = order.map { unitItem[$0] }
            let chk = order.map { $0 % 7 != 0 }
            let pr = FaceProfile(id: "t", group: 0.3, expand: 0.3, accept: 0.25, other: 0.25, maybe: 0.2, pickFloor: 0.2, consensus: 0.2)
            let m1 = matchPerson(faces: memFaces, faceItem: fItem, faceChecked: chk, nItems: nItems, inScope: [Bool](repeating: true, count: nItems),
                                 refs: [lib[5]], profile: pr, expandRounds: 2)
            let m2 = matchPerson(faces: faces.emb, faceItem: fItem, faceChecked: chk, nItems: nItems, inScope: [Bool](repeating: true, count: nItems),
                                 refs: [ref[5]], profile: pr, expandRounds: 2)
            if sc == .f32 { XCTAssertEqual(m1, m2) }
            else { XCTAssertGreaterThanOrEqual(Double(Set(m1.members).intersection(m2.members).count), 0.97 * Double(max(m1.members.count, 1))) }
            let at = locateRefs([lib[3], lib[900], r.unit(d)], in: faces.emb)
            XCTAssertEqual(at.map { $0.map { order[$0] } }, [3, 900, nil])
            let px = [Float](repeating: 50, count: memFaces.count), det = [Float](repeating: 0.9, count: memFaces.count)
            let g1 = faceGroupsPreferChecked(faces: memFaces.map { sc == .f32 ? $0 : f16($0) }, faceItem: fItem, facePx: px, det: det, checked: chk, accept: 0.35)
            let g2 = faceGroupsPreferChecked(faces: faces.emb, faceItem: fItem, facePx: px, det: det, checked: chk, accept: 0.35)
            XCTAssertEqual(g1, g2)
        }
    }

    func topOverlap(_ a: [Float], _ b: [Float], _ k: Int) -> Double {
        let ta = Set(a.indices.sorted { a[$0] > a[$1] }.prefix(k)), tb = Set(b.indices.sorted { b[$0] > b[$1] }.prefix(k))
        return Double(ta.intersection(tb).count)
    }

    /// Whole-library load time (FP_STORE_BENCH=1: Reza's 187k; default 10k so `swift test` stays quick).
    func testLoadBenchmark() throws {
        let n = ProcessInfo.processInfo.environment["FP_STORE_BENCH"] != nil ? 187_000 : 10_000
        let dir = storeTestDir("bench\(n)")
        defer { try? FileManager.default.removeItem(at: dir) }
        var r = LCG(s: 5)
        let pool = (0..<512).map { _ in r.unit(1024) }, fpool = (0..<512).map { _ in r.unit(512) }
        let places = (0..<400).map { "Place \($0), Region \($0 % 37), US, United States" }
        let cfg = IndexStore.Config(imageDim: 1024, faceDim: 512)
        var nFaces = 0, nUnits = 0
        let t0 = Date()
        do {
            let s = try IndexStore.open(dir: dir, config: cfg)
            for k in 0..<n {
                let nf = [0, 0, 1, 1, 1, 2, 3][k % 7]
                func fc() -> DetectedFace { DetectedFace(box: [100, 120, 220, 260], imageW: 1280, imageH: 960, confidence: 0.95, embedding: fpool[(k * 7 + nFaces) % 512]) }
                let isVideo = k % 20 == 0
                let id = String(format: "%08X-%04X-%04X-%04X-%012X/L0/001", k, k % 65536, 4660, 22136, k * 2654435761 % 0xffffffffffff)
                var e = FullIndexEntry(id: id, isVideo: isVideo, taken: 1.4e9 + Double(k) * 1000, localMinutes: k % 1440,
                                       lat: 40 + Double(k % 100) / 100, lon: -71 - Double(k % 50) / 100, vector: pool[k % 512],
                                       faces: isVideo ? nil : (0..<nf).map { _ in fc() }, place: places[k % 400],
                                       camera: k % 9 == 0 ? "front" : "back", isScreenshot: k % 31 == 0, lowRes: k % 13 == 0 ? true : nil,
                                       faceModel: "auraface_flip", imageVersion: 2, faceSide: 1280)
                if isVideo {
                    e.frames = (0..<6).map { j in FrameUnit(t: Double(j) * 2, vector: pool[(k + j) % 512], faces: j == 2 ? [fc()] : []) }
                    e.vector = e.frames![0].vector
                    nFaces += 1; nUnits += 6
                } else { nFaces += nf; nUnits += 1 }
                try s.put(e)
                if k % 200 == 199 { try s.commit() }                 // the app's save cadence
                if k % 200 == 199 && k % 10000 == 9999 { s.setNotRead("nr\(k)", .waitingForICloud) }
            }
            try s.commit()
        }
        let build = Date().timeIntervalSince(t0)
        let rss0 = memoryLine()
        let t1 = Date()
        let s = try IndexStore.open(dir: dir, config: cfg)
        let load = Date().timeIntervalSince(t1)
        let rss1 = memoryLine()
        XCTAssertEqual(s.records.count, n)
        // the app's biggest search read: every unit scored against two looks
        let t2 = Date()
        let ids = Array(s.records.keys)
        let u = try s.units(ids: ids)
        let sc = lookScores(units: u.vectors, unitItem: u.unitItem, nItems: ids.count, looks: [pool[1], pool[2]], avoid: [])
        let search = Date().timeIntervalSince(t2)
        let t2b = Date()
        _ = lookScores(units: u.vectors, unitItem: u.unitItem, nItems: ids.count, looks: [pool[1], pool[2]], avoid: [])
        let searchWarm = Date().timeIntervalSince(t2b)
        XCTAssertEqual(sc.count, ids.count)
        let t3 = Date()
        let f = try s.allFaces(model: "auraface_flip")
        let m = matchPerson(faces: f.emb, faceItem: f.item.indices.map { $0 }, faceChecked: f.item.map { _ in true }, nItems: f.item.count,
                            inScope: f.item.map { _ in true }, refs: [normed(zip(fpool[3], fpool[4]).map { $0 + 0.3 * $1 })], expandRounds: 0)
        let people = Date().timeIntervalSince(t3)
        let t3b = Date()
        _ = matchPerson(faces: f.emb, faceItem: f.item.indices.map { $0 }, faceChecked: f.item.map { _ in true }, nItems: f.item.count,
                        inScope: f.item.map { _ in true }, refs: [fpool[3]], expandRounds: 0)
        let peopleWarm = Date().timeIntervalSince(t3b)
        // "Who is X?" rebuild (after each face-upgrade chunk) and a subject search, at this size (FP_STORE_BENCH only:
        // tens of billions of multiply-adds, minutes in a Debug build)
        if ProcessInfo.processInfo.environment["FP_STORE_BENCH"] != nil {
        let t6 = Date()
        let groups = faceGroupsPreferChecked(faces: f.emb, faceItem: f.item.indices.map { $0 }, facePx: f.px, det: f.det,
                                             checked: f.item.map { _ in true }, accept: 0.62, faceTaken: f.taken)
        let groupSec = Date().timeIntervalSince(t6)
        let t7 = Date()
        let subj = subjectScoresCandidates(units: u.vectors, unitItem: u.unitItem, nItems: ids.count, refs: [pool[7], pool[8], pool[9]])
        let subjSec = Date().timeIntervalSince(t7)
        // the 20k x 20k x 512 product alone (Accelerate on the phone; the plain-order loop here)
        let gm = min(faceGroupCap, f.item.count)
        let Eg = (0..<(256 * 512)).map { Float($0 % 7) }, Fg = [Float](repeating: 0.5, count: gm * 512)
        var Sg = [Float](repeating: 0, count: 256 * gm)
        let t8 = Date()
        Eg.withUnsafeBufferPointer { a in Fg.withUnsafeBufferPointer { b in Sg.withUnsafeMutableBufferPointer { c in
            MatrixMath.gemmNT(a.baseAddress!, m: 256, b.baseAddress!, n: gm, d: 512, c.baseAddress!)
        } } }
        let gemmBlock = Date().timeIntervalSince(t8)
        print(String(format: "STORE BENCH faceGroupsPreferChecked over %d faces (cap %d): %.2f s, %d groups (largest %d photos); "
                     + "the cap^2 product alone here ~%.1f s (256-row block %.3f s x %d blocks); subjectScoresCandidates "
                     + "(K=%d) over %d units: %.2f s", f.item.count, faceGroupCap, groupSec, groups.count, groups.first?.items.count ?? 0,
                     gemmBlock * Double((gm + 255) / 256), gemmBlock, (gm + 255) / 256, subjectCandidateUnits, u.vectors.count, subjSec))
        XCTAssertEqual(subj.count, ids.count)
        }
        let rss2 = memoryLine()
        // the old format at a smaller size, for comparison (JSON of FullIndexEntry = old index.json)
        let jn = min(n, 2000)
        let sample = try ids.prefix(jn).map { try s.full($0)! }
        let js = try JSONEncoder().encode(sample)
        let t4 = Date()
        _ = try JSONDecoder().decode([FullIndexEntry].self, from: js)
        let jsonSec = Date().timeIntervalSince(t4)
        print(String(format: "STORE BENCH n=%d units=%d faces=%d: build %.1f s, disk %.0f MB (snapshot+log %.1f MB); OPEN %.3f s; "
                     + "lookScores over all units %.3f s; matchPerson over all faces %.3f s (%d members)",
                     n, nUnits, nFaces, build, Double(s.diskBytes) / 1e6, Double(s.logBytes) / 1e6, load, search, people, m.members.count))
        print(String(format: "STORE BENCH second (page-cache warm) pass: lookScores %.3f s, matchPerson %.3f s", searchWarm, peopleWarm))
        print("STORE BENCH memory before open: \(rss0)\nSTORE BENCH memory after open:  \(rss1)\nSTORE BENCH memory after searches: \(rss2)")
        print(String(format: "STORE BENCH old JSON format: %d entries = %.1f MB decoded in %.3f s (%.1f KB/entry; x%d -> %.0f s, %.1f GB at n)",
                     jn, Double(js.count) / 1e6, jsonSec, Double(js.count) / Double(jn) / 1e3, n / jn, jsonSec * Double(n) / Double(jn),
                     Double(js.count) * Double(n) / Double(jn) / 1e9))
        // from a fresh snapshot (as after compaction / migration)
        try s.compact(rewriteVectors: false)
        let t5 = Date()
        let s2 = try IndexStore.open(dir: dir, config: cfg)
        print(String(format: "STORE BENCH open from snapshot only: %.3f s (%d entries)", Date().timeIntervalSince(t5), s2.records.count))
    }

    /// VmRSS / RssAnon / RssFile (Linux); RssAnon is the phone's "footprint" analogue (mapped file pages are RssFile).
    func memoryLine() -> String {
        #if os(Linux)
        guard let t = try? String(contentsOfFile: "/proc/self/status", encoding: .utf8) else { return "?" }
        return t.split(separator: "\n").filter { $0.hasPrefix("VmRSS") || $0.hasPrefix("RssAnon") || $0.hasPrefix("RssFile") }
            .map { $0.replacingOccurrences(of: "\t", with: " ") }.joined(separator: ", ")
        #else
        return "n/a"
        #endif
    }
}
