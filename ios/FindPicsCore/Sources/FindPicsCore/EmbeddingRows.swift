// Vectors as rows, wherever they live: [[Float]] in memory, or rows of the index store's memory-mapped Float16 /
// Float32 files (IndexStore.swift). Search and face code read rows through this protocol, a block at a time, so a
// 187k-photo library is never copied into RAM as [[Float]] (1024 Float32 per photo = 766 MB; the app has ~1.1 GB).
// Math: MatrixMath.gemmNT is cblas_sgemm on Apple (Accelerate) and a plain loop elsewhere whose sums run in the same
// order as FaceMatch.dot, so on Linux the blocked functions give bit-identical scores to the [[Float]] ones.
import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

public protocol EmbeddingRows: Sendable {
    var count: Int { get }
    var dim: Int { get }
    /// Rows [start, start + n) as Float32, row-major n x dim; the buffer is valid inside `body` only.
    func withRows<R>(_ start: Int, _ n: Int, _ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R
}

extension Array: EmbeddingRows where Element == [Float] {
    public var dim: Int { first?.count ?? 0 }
    public func withRows<R>(_ start: Int, _ n: Int, _ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        if n == 1 { return try self[start].withUnsafeBufferPointer(body) }
        var buf = [Float](); buf.reserveCapacity(n * dim)
        for r in start..<(start + n) { buf += self[r] }
        return try buf.withUnsafeBufferPointer(body)
    }
}

extension EmbeddingRows {
    /// One row, copied out as Float32.
    public subscript(_ i: Int) -> [Float] { withRows(i, 1) { Array($0) } }
    /// These rows only, in this order (no copy).
    public func subset(_ rows: [Int]) -> RowSubset<Self> { RowSubset(base: self, rows: rows) }
    /// Every row in blocks of `size` (the last one shorter): (first row, rows in block, block n x dim).
    public func forEachBlock(size: Int = 1024, _ body: (Int, Int, UnsafeBufferPointer<Float>) throws -> Void) rethrows {
        var s = 0
        while s < count {
            let n = Swift.min(size, count - s)
            try withRows(s, n) { try body(s, n, $0) }
            s += n
        }
    }
}

/// Rows of `base` picked by index (e.g. the checked faces only), without copying them.
public struct RowSubset<Base: EmbeddingRows>: EmbeddingRows {
    public let base: Base
    public let rows: [Int]
    public var count: Int { rows.count }
    public var dim: Int { base.dim }
    public func withRows<R>(_ start: Int, _ n: Int, _ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        if n == 1 { return try base.withRows(rows[start], 1, body) }
        let d = dim
        return try withUnsafeTemporaryAllocation(of: Float.self, capacity: Swift.max(n * d, 1)) { buf in
            for k in 0..<n {
                base.withRows(rows[start + k], 1) { r in
                    for j in 0..<d { buf[k * d + j] = r[j] }
                }
            }
            return try body(UnsafeBufferPointer(buf))
        }
    }
}

public enum MatrixMath {
    /// C (m x n, row-major) = A (m x d) * B (n x d)^T.
    public static func gemmNT(_ A: UnsafePointer<Float>, m: Int, _ B: UnsafePointer<Float>, n: Int, d: Int,
                              _ C: UnsafeMutablePointer<Float>) {
        guard m > 0, n > 0 else { return }
        #if canImport(Accelerate)
        if d > 0 {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(m), Int32(n), Int32(d), 1, A, Int32(d), B, Int32(d),
                        0, C, Int32(n))
            return
        }
        #endif
        for i in 0..<m {
            let a = A + i * d
            for j in 0..<n {
                let b = B + j * d
                var s: Float = 0
                for k in 0..<d { s += a[k] * b[k] }
                C[i * n + j] = s
            }
        }
    }

    /// `rows` (n x d) as one contiguous Float32 array (small sets: references, query vectors).
    public static func flat(_ rows: [[Float]], d: Int) -> [Float] {
        var out = [Float](); out.reserveCapacity(rows.count * d)
        for r in rows { out += r }
        return out
    }

    /// Float16 -> Float32 for n values.
    @inline(__always)
    public static func halfToFloat(_ src: UnsafeRawPointer, _ dst: UnsafeMutablePointer<Float>, _ n: Int) {
        #if canImport(Accelerate)
        var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
        var t = vImage_Buffer(data: UnsafeMutableRawPointer(dst), height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
        vImageConvert_Planar16FtoPlanarF(&s, &t, 0)
        #else
        let p = src.assumingMemoryBound(to: UInt16.self)
        halfTable.withUnsafeBufferPointer { lut in for i in 0..<n { dst[i] = lut[Int(p[i])] } }
        #endif
    }

    /// Float32 -> Float16 bit patterns (IEEE round-to-nearest-even, as vImage and the hardware do).
    public static func floatToHalfBits(_ v: Float) -> UInt16 { halfBits(v) }

    #if !canImport(Accelerate)
    /// Every Float16 bit pattern as Float32 (256 KB, built once): Linux x86 has no fast Float16 conversion.
    static let halfTable: [Float] = (0..<65536).map { halfToFloatScalar(UInt16($0)) }
    #endif
}

/// Float32 -> Float16 bits, round to nearest even; overflow -> inf; NaN stays NaN. Portable (no Float16 type needed).
func halfBits(_ f: Float) -> UInt16 {
    let x = f.bitPattern
    let sign = UInt16((x >> 16) & 0x8000)
    let exp = Int((x >> 23) & 0xff), man = x & 0x7f_ffff
    if exp == 0xff { return sign | 0x7c00 | (man != 0 ? 0x200 : 0) }          // inf / NaN
    let e = exp - 127 + 15
    if e >= 0x1f { return sign | 0x7c00 }                                       // overflow
    if e <= 0 {                                                                 // subnormal half (or zero)
        if e < -10 { return sign }
        let m = man | 0x80_0000
        let shift = UInt32(14 - e)
        var h = m >> shift
        let rem = m & ((1 << shift) - 1), half = UInt32(1) << (shift - 1)
        if rem > half || (rem == half && (h & 1) == 1) { h += 1 }
        return sign | UInt16(h)
    }
    var h = UInt32(e) << 10 | (man >> 13)
    let rem = man & 0x1fff
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) { h += 1 }               // may carry into the exponent: correct
    return sign | UInt16(h)
}

/// Float16 bits -> Float32 (exact).
func halfToFloatScalar(_ h: UInt16) -> Float {
    let sign = UInt32(h & 0x8000) << 16
    let exp = Int((h >> 10) & 0x1f), man = UInt32(h & 0x3ff)
    if exp == 0 {
        if man == 0 { return Float(bitPattern: sign) }
        return (sign != 0 ? -1 : 1) * Float(man) * pow(2, -24)                  // subnormal
    }
    if exp == 0x1f { return Float(bitPattern: sign | 0x7f80_0000 | (man << 13)) }
    return Float(bitPattern: sign | UInt32(exp - 15 + 127) << 23 | (man << 13))
}
