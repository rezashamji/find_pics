// Pillow's Image.resize(size, Image.BILINEAR) for 8-bit pixels, bit-exact (Pillow src/libImaging/Resample.c:
// precompute_coeffs + normalize_coeffs_8bpc + ResampleHorizontal_8bpc / ResampleVertical_8bpc).
// Why: the server's open_clip preprocess for PE-Core-B-16 is PIL resize((224, 224), BILINEAR) ("squash", no crop).
// Pillow's bilinear WIDENS its triangle filter by the shrink factor (a 1280 px photo -> 224: each output pixel averages
// ~11 source pixels per axis). Core Image's affine transform takes one 2x2 bilinear tap per output pixel instead, which
// aliases; that alone put the phone's Self-check image cosines at 0.93 / 0.98 (eval/coreml_preproc_parity.py).
// Golden test: Tests/FindPicsCoreTests/PILResizeTests.swift (fixtures from eval/pil_resize_fixtures.py).

public enum PILResize {
    static let precisionBits = 32 - 8 - 2

    /// For each output index: first source index, tap count, and fixed-point weights (Pillow's 8-bit path).
    static func coeffs(inSize: Int, outSize: Int) -> (bounds: [(Int, Int)], ksize: Int, kk: [Int]) {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = 1.0 * filterscale                    // bilinear (triangle) filter support = 1
        let ksize = Int(support.rounded(.up)) * 2 + 1
        var bounds = [(Int, Int)](); bounds.reserveCapacity(outSize)
        var kk = [Int](repeating: 0, count: outSize * ksize)
        let ss = 1.0 / filterscale
        var k = [Double](repeating: 0, count: ksize)
        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            var ww = 0.0
            var xmin = Int(center - support + 0.5)          // C (int) cast: truncation toward zero
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax -= xmin
            for x in 0..<ksize { k[x] = 0 }
            for x in 0..<max(xmax, 0) {
                var t = (Double(x + xmin) - center + 0.5) * ss
                if t < 0 { t = -t }
                let w = t < 1 ? 1 - t : 0
                k[x] = w; ww += w
            }
            for x in 0..<max(xmax, 0) where ww != 0 { k[x] /= ww }
            for x in 0..<ksize {
                let v = k[x] * Double(1 << precisionBits)
                kk[xx * ksize + x] = v < 0 ? Int(-0.5 + v) : Int(0.5 + v)
            }
            bounds.append((xmin, max(xmax, 0)))
        }
        return (bounds, ksize, kk)
    }

    @inline(__always) static func clip8(_ v: Int) -> UInt8 {
        let s = v >> precisionBits
        return s < 0 ? 0 : (s > 255 ? 255 : UInt8(s))
    }

    /// `src`: width x height pixels, `channels` interleaved bytes per pixel (e.g. 4 for RGBA8 / RGBX), rows top-down.
    /// Returns toWidth x toHeight pixels in the same layout. All channels are resampled independently (alpha ignored,
    /// as Pillow does for RGB, whose pixels are stored as RGBX).
    public static func bilinear(_ src: [UInt8], width: Int, height: Int, channels: Int,
                                toWidth: Int, toHeight: Int) -> [UInt8] {
        precondition(src.count >= width * height * channels && width > 0 && height > 0 && toWidth > 0 && toHeight > 0)
        let (hb, hk, hkk) = coeffs(inSize: width, outSize: toWidth)
        let (vb, vk, vkk) = coeffs(inSize: height, outSize: toHeight)
        // Pillow resamples only the rows the vertical pass reads (ybox_first ..< ybox_last); same result.
        let yFirst = vb[0].0, yLast = vb[toHeight - 1].0 + vb[toHeight - 1].1
        let rows = yLast - yFirst
        let half = 1 << (precisionBits - 1)
        var tmp = [UInt8](repeating: 0, count: max(rows, 0) * toWidth * channels)
        src.withUnsafeBufferPointer { s in
            tmp.withUnsafeMutableBufferPointer { t in
                for yy in 0..<rows {
                    let rowIn = (yy + yFirst) * width * channels, rowOut = yy * toWidth * channels
                    for xx in 0..<toWidth {
                        let (xmin, xmax) = hb[xx], kb = xx * hk
                        for c in 0..<channels {
                            var acc = half
                            for x in 0..<xmax { acc += Int(s[rowIn + (x + xmin) * channels + c]) * hkk[kb + x] }
                            t[rowOut + xx * channels + c] = clip8(acc)
                        }
                    }
                }
            }
        }
        var out = [UInt8](repeating: 0, count: toWidth * toHeight * channels)
        tmp.withUnsafeBufferPointer { t in
            out.withUnsafeMutableBufferPointer { o in
                for yy in 0..<toHeight {
                    let (ymin0, ymax) = vb[yy], ymin = ymin0 - yFirst, kb = yy * vk
                    for xx in 0..<toWidth {
                        for c in 0..<channels {
                            var acc = half
                            for y in 0..<ymax { acc += Int(t[((y + ymin) * toWidth + xx) * channels + c]) * vkk[kb + y] }
                            o[(yy * toWidth + xx) * channels + c] = clip8(acc)
                        }
                    }
                }
            }
        }
        return out
    }
}
