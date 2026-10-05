// swift-tools-version:5.9
// Core search logic of the find_pics iPhone app, platform-free so it compiles and is tested on Linux too.
// Apple-only parts (PhotoKit, Core ML, MLX, SwiftUI) live in the app target and call into this package.
import PackageDescription

let package = Package(
    name: "FindPicsCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "FindPicsCore", targets: ["FindPicsCore"])],
    targets: [
        .target(name: "FindPicsCore", resources: [.copy("Resources/bpe_simple_vocab_16e6.txt"), .copy("Resources/planner_template.txt")]),
        .testTarget(name: "FindPicsCoreTests", dependencies: ["FindPicsCore"], resources: [.copy("Fixtures")]),
    ]
)
