import Foundation
import XCTest
@testable import FindPicsCore

struct ResizeCase: Decodable { let w, h, tw, th: Int; let kind: String; let seed: UInt64; let fnv: String; let idx: [Int]; let vals: [UInt8] }

final class PILResizeTests: XCTestCase {
    /// Same inputs as eval/pil_resize_fixtures.py (LCG noise / integer pattern), RGB.
    static func make(_ c: ResizeCase) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: c.w * c.h * 3)
        if c.kind == "noise" {
            var s = c.seed
            for i in 0..<out.count { s = s &* 6364136223846793005 &+ 1442695040888963407; out[i] = UInt8(s >> 56) }
        } else {
            for y in 0..<c.h { for x in 0..<c.w { for ch in 0..<3 {
                out[(y * c.w + x) * 3 + ch] = UInt8((x * 3 + y * 5 + ch * 71 + (x * y) % 17) & 255)
            } } }
        }
        return out
    }

    /// Bit-exact with Pillow's resize(BILINEAR) (the server's open_clip preprocess) on 11 sizes, down and up.
    func testMatchesPillow() throws {
        let url = Bundle.module.url(forResource: "pil_resize", withExtension: "json", subdirectory: "Fixtures")!
        let cases = try JSONDecoder().decode([ResizeCase].self, from: Data(contentsOf: url))
        XCTAssertEqual(cases.count, 11)
        for c in cases {
            let rgb = Self.make(c)
            let got = PILResize.bilinear(rgb, width: c.w, height: c.h, channels: 3, toWidth: c.tw, toHeight: c.th)
            var h: UInt64 = 0xcbf29ce484222325
            for v in got { h = (h ^ UInt64(v)) &* 0x100000001b3 }
            let bad = zip(c.idx, c.vals).filter { got[$0.0] != $0.1 }.count
            XCTAssertEqual(String(h), c.fnv, "\(c.w)x\(c.h) -> \(c.tw)x\(c.th) \(c.kind): \(bad)/\(c.idx.count) sampled bytes differ")
            // 4 channels (RGBA8 from CIContext) give the same RGB as 3
            if c.w * c.h <= 640 * 480 {
                var rgba = [UInt8](repeating: 255, count: c.w * c.h * 4)
                for p in 0..<(c.w * c.h) { for ch in 0..<3 { rgba[p * 4 + ch] = rgb[p * 3 + ch] } }
                let g4 = PILResize.bilinear(rgba, width: c.w, height: c.h, channels: 4, toWidth: c.tw, toHeight: c.th)
                for p in 0..<(c.tw * c.th) { for ch in 0..<3 where g4[p * 4 + ch] != got[p * 3 + ch] {
                    XCTFail("RGBA differs at \(p)"); return
                } }
            }
        }
    }
}
