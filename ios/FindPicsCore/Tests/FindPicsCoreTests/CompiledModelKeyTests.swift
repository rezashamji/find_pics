import XCTest
@testable import FindPicsCore

final class CompiledModelKeyTests: XCTestCase {
    private let files = [(path: "Data/weights/weight.bin", size: 178_000_000), (path: "Manifest.json", size: 600)]

    func testSamePackageSameKeyWhateverTheOrder() {
        let a = compiledModelKey(name: "Models/pe_core_image", files: files, manifest: "{m}")
        let b = compiledModelKey(name: "Models/pe_core_image", files: files.reversed(), manifest: "{m}")
        XCTAssertEqual(a, b)
        XCTAssertTrue(a.hasPrefix("Models_pe_core_image-"))
    }

    func testAChangedModelGetsANewKey() {
        let a = compiledModelKey(name: "x", files: files, manifest: "{m}")
        XCTAssertNotEqual(a, compiledModelKey(name: "x", files: [(path: "Data/weights/weight.bin", size: 1), files[1]],
                                              manifest: "{m}"))
        XCTAssertNotEqual(a, compiledModelKey(name: "x", files: files, manifest: "{m2}"))
        XCTAssertNotEqual(a, compiledModelKey(name: "y", files: files, manifest: "{m}"))
    }
}
