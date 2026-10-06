// swift-tools-version: 6.2
// find_pics iPhone app. Open this .swiftpm folder in Xcode 27 (File > Open > FindPicsApp.swiftpm) and press Run on your iPhone.
// (.swiftpm: Xcode only provides AppleProductTypes / .iOSApplication to app packages in a .swiftpm folder.)
// The app reads your photo library on the phone; photos never leave it. Core search logic: ../FindPicsCore (tested on
// Linux against the Python engine). Untested on a device until the first build: expect compile fixes.
import AppleProductTypes
import PackageDescription

let package = Package(
    name: "FindPics",
    platforms: [.iOS("17.0")],
    products: [
        .iOSApplication(
            name: "find pics",
            targets: ["FindPicsApp"],
            bundleIdentifier: "com.rezashamji.findpics",
            displayVersion: "0.1",
            bundleVersion: "1",
            appIcon: .placeholder(icon: .magnifyingGlass),
            accentColor: .presetColor(.blue),
            supportedDeviceFamilies: [.phone],
            supportedInterfaceOrientations: [.portrait],
            capabilities: [
                .photoLibrary(purposeString: "find pics searches your photos ON this phone. Nothing is uploaded."),
                .photoLibraryAdd(purposeString: "find pics can save a search result as a new album (only when you tap Save)."),
            ]
        )
    ],
    dependencies: [
        .package(path: "../FindPicsCore"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3")),
        // required next to MLXHuggingFace (mlx-swift-lm README): the model-download macro uses HubClient + AutoTokenizer
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),   // `import MLX` in Judge
    ],
    targets: [
        .executableTarget(
            name: "FindPicsApp",
            dependencies: [
                "FindPicsCore",
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources",
            resources: [.copy("Models"), .copy("SelfCheck")]
        )
    ]
)
