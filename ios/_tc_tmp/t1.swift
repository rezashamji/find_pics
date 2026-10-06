import Foundation
func f() throws -> URL { URL(fileURLWithPath: "/") }
func g(_ u: URL) throws -> Int { 1 }
let u0 = URL(fileURLWithPath: "/a")
func t1() throws { let c = u0.path == "x" ? u0 : try f(); _ = c }
func t2() throws { _ = try g(u0.path == "x" ? u0 : try f()) }
func cam() async -> String { "" }
struct E { let camera: String? }
func t3() async { let isV = false; _ = E(camera: isV ? nil : await cam()) }
func opt() async -> Int? { 1 }
func pair() async throws -> (Int, Double) { (1, 2) }
func t5() async { if let (a, b) = try? await pair() { _ = a; _ = b } }
func norm(_ v: inout [Float]) { var s: Float = 0; s += 1; add(v, &v); _ = s }
func add(_ a: UnsafePointer<Float>, _ b: UnsafeMutablePointer<Float>) {}
func run(update: @escaping @Sendable (Int) -> Void) async {}
