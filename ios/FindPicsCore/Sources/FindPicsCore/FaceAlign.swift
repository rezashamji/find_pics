// Face alignment for the face model: the 2x3 similarity transform (rotation + uniform scale + shift) that maps the 5
// landmarks (eyes, nose tip, mouth corners) onto the ArcFace template in a 112x112 crop. Port of insightface
// estimate_norm = skimage SimilarityTransform.estimate (Umeyama 1991, least squares, no reflection).
import Foundation

public let arcfaceTemplate: [(Double, Double)] = [(38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366),
                                                  (41.5493, 92.3655), (70.7299, 92.2041)]

/// Returns [[a, -b, tx], [b, a, ty]] mapping image points to crop points (crop = M * [x, y, 1]).
public func similarityTransform(from src: [(Double, Double)], to dst: [(Double, Double)] = arcfaceTemplate) -> [[Double]] {
    let n = Double(src.count)
    let sx = src.map { $0.0 }.reduce(0, +) / n, sy = src.map { $0.1 }.reduce(0, +) / n
    let dx = dst.map { $0.0 }.reduce(0, +) / n, dy = dst.map { $0.1 }.reduce(0, +) / n
    var sxx = 0.0, sxy = 0.0, syx = 0.0, syy = 0.0, varS = 0.0
    for (s, d) in zip(src, dst) {
        let ax = s.0 - sx, ay = s.1 - sy, bx = d.0 - dx, by = d.1 - dy
        sxx += bx * ax; sxy += bx * ay; syx += by * ax; syy += by * ay   // covariance A = dst^T src / n
        varS += ax * ax + ay * ay
    }
    sxx /= n; sxy /= n; syx /= n; syy /= n; varS /= n
    // 2x2 Umeyama in closed form: rotation angle and scale from the covariance (no reflection allowed)
    let a = sxx + syy, b = syx - sxy
    let norm = sqrt(a * a + b * b)
    let c = a / norm, s = b / norm, scale = norm / varS
    let r00 = scale * c, r01 = -scale * s, r10 = scale * s, r11 = scale * c
    return [[r00, r01, dx - (r00 * sx + r01 * sy)], [r10, r11, dy - (r10 * sx + r11 * sy)]]
}
