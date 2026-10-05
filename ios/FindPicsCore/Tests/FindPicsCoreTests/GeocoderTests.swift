import Foundation
import XCTest
@testable import FindPicsCore

struct GeoCase: Decodable { let lat: Double; let lon: Double; let name: String; let place: String }

final class GeocoderTests: XCTestCase {
    /// Same place text as ingest.nearest_place (the server's, sphere distance) on 1,500 points near cities + 500 random.
    func testMatchesReverseGeocoder() throws {
        let g = try Geocoder()
        let url = Bundle.module.url(forResource: "geo", withExtension: "json", subdirectory: "Fixtures")!
        var bad = 0
        let cases = try JSONDecoder().decode([GeoCase].self, from: Data(contentsOf: url))
        for c in cases {
            let n = g.placeText(lat: c.lat, lon: c.lon)
            if n != c.place { bad += 1; if bad <= 5 { print("GEO mismatch \(c.lat),\(c.lon): \(n ?? "-") vs \(c.place)") } }
        }
        print("GEO mismatches \(bad) of \(cases.count)")
        XCTAssertLessThanOrEqual(bad, 2)   // float ties between equidistant cities
        XCTAssertEqual(g.placeText(lat: 48.8566, lon: 2.3522)?.hasSuffix("FR, France"), true)
    }
}
