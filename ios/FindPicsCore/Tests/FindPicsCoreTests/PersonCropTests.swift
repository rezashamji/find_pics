import Foundation
import XCTest
@testable import FindPicsCore

struct CropCase: Decodable { let face: [Double]; let W: Double; let H: Double; let crop_size: [Int]; let red: [Int]? }

final class PersonCropTests: XCTestCase {
    /// Same crop size and red-box position as engine.person_crop_boxed (PIL) on 150 random faces / image sizes.
    func testMatchesPython() throws {
        let url = Bundle.module.url(forResource: "crops", withExtension: "json", subdirectory: "Fixtures")!
        for c in try JSONDecoder().decode([CropCase].self, from: Data(contentsOf: url)) {
            let (crop, box) = personCrop(face: Rect(c.face[0], c.face[1], c.face[2], c.face[3]), imageW: c.W, imageH: c.H)
            XCTAssertEqual(Int(crop.x2 - crop.x1), c.crop_size[0]); XCTAssertEqual(Int(crop.y2 - crop.y1), c.crop_size[1])
            if let r = c.red {   // PIL draws the outline inward from the box edges: outer edge = box corner (rounded)
                XCTAssertEqual(Double(r[0]), box.x1, accuracy: 1.01); XCTAssertEqual(Double(r[1]), box.y1, accuracy: 1.01)
                XCTAssertEqual(Double(r[2]), min(box.x2, crop.x2 - crop.x1 - 1), accuracy: 1.01)
                XCTAssertEqual(Double(r[3]), min(box.y2, crop.y2 - crop.y1 - 1), accuracy: 1.01)
            }
        }
    }
}
