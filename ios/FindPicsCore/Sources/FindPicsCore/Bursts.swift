// Near-identical shots of one moment shown as one stack: port of src/findpics/bursts.py (display only).
import Foundation

public let burstSim: Float = 0.90       // image-vector cosine
public let burstMinutes = 10.0          // and taken at most this far apart

/// Group id per item, in album order (the first item of each group is its cover). `vectors` are L2-normalized image
/// vectors; `taken` is seconds since 1970 (nil = no date: never stacked).
public func burstIds(vectors: [[Float]], taken: [Double?], sim: Float = burstSim, minutes: Double = burstMinutes) -> [Int] {
    let n = vectors.count
    var gid = [Int](repeating: -1, count: n), g = 0
    func linked(_ a: Int, _ b: Int) -> Bool {
        guard let ta = taken[a], let tb = taken[b], abs(ta - tb) / 60 <= minutes else { return false }
        var dot: Float = 0
        for k in 0..<vectors[a].count { dot += vectors[a][k] * vectors[b][k] }
        return dot >= sim
    }
    for i in 0..<n where gid[i] < 0 {
        gid[i] = g; var stack = [i]
        while let j = stack.popLast() {
            for k in 0..<n where gid[k] < 0 && linked(j, k) { gid[k] = g; stack.append(k) }
        }
        g += 1
    }
    return gid
}
