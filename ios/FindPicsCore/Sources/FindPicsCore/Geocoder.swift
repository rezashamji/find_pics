// GPS -> place name, offline: the nearest of ~144,000 GeoNames cities (rg_cities1000.csv, the same table the server's
// reverse_geocoder uses; GeoNames, CC-BY 4.0) + the country's name (countries.tsv) so "photos from France" can match.
// Nearest by straight-line distance between points on the unit sphere, like reverse_geocoder's KD-tree.
import Foundation

public final class Geocoder {
    struct City { let x, y, z: Double; let name, admin1, admin2, cc: String }
    var cells = [Int: [Int]]()
    var cities = [City]()
    public let countries: [String: String]

    public init(citiesURL: URL? = nil, countriesURL: URL? = nil) throws {
        let cu = citiesURL ?? Bundle.module.url(forResource: "rg_cities1000", withExtension: "csv")!
        let text = try String(contentsOf: cu, encoding: .utf8)
        // "\r\n" is ONE Character in Swift, so split on newline scalars, not on the Character "\n"
        for (i, line) in text.components(separatedBy: .newlines).filter({ !$0.isEmpty }).enumerated() where i > 0 {
            let f = Geocoder.csvFields(line)
            guard f.count >= 6, let lat = Double(f[0]), let lon = Double(f[1]) else { continue }
            let (x, y, z) = Geocoder.xyz(lat, lon)
            cells[Geocoder.cell(lat, lon), default: []].append(cities.count)
            cities.append(City(x: x, y: y, z: z, name: f[2], admin1: f[3], admin2: f[4], cc: f[5]))
        }
        var c = [String: String]()
        let ct = try String(contentsOf: countriesURL ?? Bundle.module.url(forResource: "countries", withExtension: "tsv")!, encoding: .utf8)
        for line in ct.components(separatedBy: .newlines) { let f = line.split(separator: "\t"); if f.count >= 2 { c[String(f[0])] = String(f[1]) } }
        countries = c
    }

    /// One CSV line -> fields ("quoted, with commas" stays one field; "" inside quotes = a quote).
    static func csvFields(_ line: String) -> [String] {
        var out = [String](), cur = "", quoted = false
        let it = Array(line); var i = 0
        while i < it.count {
            let c = it[i]
            if quoted {
                if c == "\"" { if i + 1 < it.count && it[i + 1] == "\"" { cur.append("\""); i += 1 } else { quoted = false } } else { cur.append(c) }
            } else if c == "\"" { quoted = true } else if c == "," { out.append(cur); cur = "" } else { cur.append(c) }
            i += 1
        }
        out.append(cur)
        return out
    }

    static func xyz(_ lat: Double, _ lon: Double) -> (Double, Double, Double) {
        let a = lat * .pi / 180, b = lon * .pi / 180
        return (cos(a) * cos(b), cos(a) * sin(b), sin(a))
    }
    static func cell(_ lat: Double, _ lon: Double) -> Int {
        let la = Int(floor(lat)) + 90, lo = ((Int(floor(lon)) % 360) + 360) % 360
        return la * 360 + lo
    }

    /// Nearest city: (name, admin1, admin2, country code).
    public func nearest(lat: Double, lon: Double) -> (name: String, admin1: String, admin2: String, cc: String)? {
        let (x, y, z) = Geocoder.xyz(lat, lon)
        var best = -1, bestD = Double.infinity
        let la0 = Int(floor(lat)), lo0 = Int(floor(lon))
        var r = 0
        while r <= 180 {
            for dla in -r...r { for dlo in -r...r where max(abs(dla), abs(dlo)) == r {
                let la = la0 + dla; if la < -90 || la > 89 { continue }
                for i in cells[(la + 90) * 360 + (((lo0 + dlo) % 360) + 360) % 360] ?? [] {
                    let c = cities[i], d = (c.x - x) * (c.x - x) + (c.y - y) * (c.y - y) + (c.z - z) * (c.z - z)
                    if d < bestD { bestD = d; best = i }
                }
            } }
            // a city in ring r+1 is >= r cells away: >= r degrees of latitude, or r degrees of longitude, which near the
            // poles shrinks by cos(latitude); stop once even that cannot beat the best
            if best >= 0 {
                let phi = min(89.999, abs(lat) + Double(r) + 1) * .pi / 180
                let minAngle = Double(r) * .pi / 180 * cos(phi)
                if minAngle > 0 && 4 * pow(sin(minAngle / 2), 2) > bestD { break }
            }
            r += 1
        }
        guard best >= 0 else { return nil }
        let c = cities[best]
        return (c.name, c.admin1, c.admin2, c.cc)
    }

    /// The place text the search matches against: "Paris, Paris, Ile-de-France, FR, France".
    public func placeText(lat: Double, lon: Double) -> String? {
        guard let n = nearest(lat: lat, lon: lon) else { return nil }
        return [n.name, n.admin2, n.admin1, n.cc, countries[n.cc] ?? ""].filter { !$0.isEmpty }.joined(separator: ", ")
    }
}
