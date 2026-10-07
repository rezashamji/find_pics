import Foundation
import XCTest
@testable import FindPicsCore

/// Timing only (prints, no assertion): how long PILResize takes per photo at the app's read sizes, so a slow phone
/// index pass (MAC M12: ~4 s/photo in a Debug build) can be attributed or ruled out. Run with -c debug and -c release.
final class PILResizeSpeedTests: XCTestCase {
    func testResizeTiming() {
        for (w, h) in [(480, 360), (1280, 960), (1600, 1200)] {
            let src = [UInt8](repeating: 128, count: w * h * 4)
            let t0 = Date()
            let n = 3
            for _ in 0..<n { _ = PILResize.bilinear(src, width: w, height: h, channels: 4, toWidth: 224, toHeight: 224) }
            print("PILRESIZE_TIMING \(w)x\(h) -> 224: \(String(format: "%.1f", Date().timeIntervalSince(t0) / Double(n) * 1000)) ms")
        }
    }
}
