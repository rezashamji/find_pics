func add(_ a: UnsafePointer<Float>, _ s: Int, _ b: UnsafePointer<Float>, _ s2: Int, _ c: UnsafeMutablePointer<Float>, _ s3: Int, _ n: UInt) {}
func t(units: [[Float]]) -> [Float] { var v = [Float](repeating: 0, count: 4); for j in 0..<2 { add(v, 1, units[j], 1, &v, 1, 4) }; return v }
func smul(_ a: UnsafePointer<Float>, _ s: Int, _ b: UnsafePointer<Float>, _ c: UnsafeMutablePointer<Float>, _ s3: Int, _ n: UInt) {}
func normalized(_ v: inout [Float]) { smul(v, 1, [2], &v, 1, UInt(v.count)) }
