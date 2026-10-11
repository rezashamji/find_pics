import Foundation
import XCTest
@testable import FindPicsCore

private actor Log {
    var events: [String] = []
    var loads: [String: Int] = [:]
    func add(_ e: String) { events.append(e) }
    func loaded(_ k: String) { loads[k, default: 0] += 1; events.append("load \(k)") }
}

final class PrefetchTests: XCTestCase {
    /// Values come back by key; a prefetched key is loaded once; a miss loads on the spot; keys no longer wanted are
    /// dropped (counted), never handed out for another key.
    func testValuesByKeyAndCounts() async {
        let log = Log()
        let pf = Prefetcher<String> { k in await log.loaded(k); return "img-\(k)" }
        await pf.want(["a", "b"])
        let a = await pf.take("a")
        XCTAssertEqual(a, "img-a")
        await pf.want(["c"])                        // b was a wrong guess
        let c = await pf.take("c"), d = await pf.take("d")
        XCTAssertEqual(c, "img-c"); XCTAssertEqual(d, "img-d")
        let hits = await pf.hits, misses = await pf.misses, dropped = await pf.dropped, held = await pf.held
        XCTAssertEqual(hits, 2); XCTAssertEqual(misses, 1); XCTAssertEqual(dropped, 1); XCTAssertEqual(held, 0)
        let loads = await log.loads
        XCTAssertEqual(loads["a"], 1); XCTAssertEqual(loads["c"], 1); XCTAssertEqual(loads["d"], 1)
    }

    /// The point of it: while photo k is being judged, photo k+1 is already loading (its load starts before judge k
    /// ends), so a judge loop over n photos waits for one load, not n.
    func testNextLoadOverlapsTheCurrentJudgeCall() async throws {
        let log = Log()
        let pf = Prefetcher<Int> { k in
            await log.add("start \(k)"); try? await Task.sleep(nanoseconds: 50_000_000); return Int(k)!
        }
        let keys = (0..<6).map(String.init)
        let t0 = Date()
        for (i, k) in keys.enumerated() {
            await pf.want(Array(keys[i..<min(i + 2, keys.count)]))       // current + next, as Search.swift does
            let v = await pf.take(k)
            XCTAssertEqual(v, i)
            try await Task.sleep(nanoseconds: 100_000_000)                    // the "model call"
            await log.add("judged \(k)")
        }
        let elapsed = Date().timeIntervalSince(t0)
        let ev = await log.events
        for i in 1..<keys.count {
            let s = ev.firstIndex(of: "start \(i)")!, j = ev.firstIndex(of: "judged \(i - 1)")!
            XCTAssertLessThan(s, j, "photo \(i) must start loading while photo \(i - 1) is judged")
        }
        let hits = await pf.hits
        XCTAssertEqual(hits, 6)
        print("PREFETCH 6 photos, load 0.05 s + judge 0.1 s each: \(String(format: "%.2f", elapsed)) s (serial would be 0.90)")
    }

    /// Qwen3-VL prompt shape: [header] <vs> [image x N] <ve> [question] [assistant header]. Same photo, two questions:
    /// same prefix, different tails. Another size (N differs) or no image: no reuse.
    func testImagePrefixSplit() {
        let vs = 151652, ve = 151653, img = 151655
        func prompt(_ n: Int, _ q: [Int]) -> [Int] { [151644, 872, 198, vs] + Array(repeating: img, count: n) + [ve] + q + [151645, 198, 151644, 77091, 198] }
        let a = prompt(588, [3872, 1052, 264, 5562]), b = prompt(588, [1972, 432, 4766])
        let n = imagePrefixLength(a, visionEnd: ve)!
        XCTAssertEqual(n, 4 + 588 + 1)
        XCTAssertEqual(imagePrefixLength(b, visionEnd: ve), n)
        let pre = Array(a[..<n])
        XCTAssertEqual(Array(tailAfterPrefix(b, prefix: pre)!), [1972, 432, 4766, 151645, 198, 151644, 77091, 198])
        XCTAssertEqual(Array(tailAfterPrefix(a, prefix: pre)!), Array(a[n...]))
        XCTAssertNil(tailAfterPrefix(prompt(784, [1972]), prefix: pre))          // a square photo: other image tokens
        XCTAssertNil(tailAfterPrefix(pre, prefix: pre))                           // nothing to append
        XCTAssertNil(tailAfterPrefix(a, prefix: []))
        XCTAssertNil(imagePrefixLength([151644, 872, 198, 3872], visionEnd: ve))  // text only
        XCTAssertNil(imagePrefixLength([1, 2, ve], visionEnd: ve))                // nothing after the image
    }
}
