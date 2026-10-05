// The judge sees ONE person: crop around the matched face (head + torso) with a red box on that face, so "the person
// in the red box" is unambiguous in group photos. Port of engine.person_crop / person_crop_boxed + vlm.draw_box math.
import Foundation

public struct Rect: Equatable { public var x1, y1, x2, y2: Double
    public init(_ x1: Double, _ y1: Double, _ x2: Double, _ y2: Double) { self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2 } }

/// Face box (in image pixels) -> (crop rect in the image, red box rect inside the crop). Image size w x h.
public func personCrop(face f: Rect, imageW w: Double, imageH h: Double) -> (crop: Rect, box: Rect) {
    let fw = f.x2 - f.x1, fh = f.y2 - f.y1
    // engine.person_crop: int() truncation of the crop corners, like PIL crop(tuple(int(v) ...))
    let crop = Rect(Double(Int(max(0, f.x1 - 1.6 * fw))), Double(Int(max(0, f.y1 - 0.6 * fh))),
                    Double(Int(min(w, f.x2 + 1.6 * fw))), Double(Int(min(h, f.y2 + 4.5 * fh))))
    let cx = max(0, f.x1 - 1.6 * fw), cy = max(0, f.y1 - 0.6 * fh)            // person_crop_boxed uses the float corner
    let bx1 = f.x1 - cx, by1 = f.y1 - cy, bx2 = f.x2 - cx, by2 = f.y2 - cy
    // vlm.draw_box widens the face box to head/shoulders, clipped to the crop
    let cw = crop.x2 - crop.x1, ch = crop.y2 - crop.y1, bw = bx2 - bx1, bh = by2 - by1
    let box = Rect(max(0, bx1 - 0.6 * bw), max(0, by1 - 0.5 * bh), min(cw, bx2 + 0.6 * bw), min(ch, by2 + 1.2 * bh))
    return (crop, box)
}

/// Line width of the red box (vlm.draw_box: max(3, int(0.006 * longest side))).
public func boxLineWidth(cropW: Double, cropH: Double) -> Int { max(3, Int(0.006 * max(cropW, cropH))) }
