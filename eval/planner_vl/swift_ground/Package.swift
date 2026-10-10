// swift-tools-version:5.9
// Replays the app's post-planner code (FindPicsCore: unanswerable/dropUnanswerable, ground, keepPartialUndo, and the
// prompt builder) on planner outputs recorded by eval/eval_planner_vl.py, and compares with the Python results.
import PackageDescription
let package = Package(
    name: "GroundReplay",
    platforms: [.macOS(.v14)],
    dependencies: [.package(path: "../../../ios/FindPicsCore")],
    targets: [.executableTarget(name: "GroundReplay", dependencies: [.product(name: "FindPicsCore", package: "FindPicsCore")])]
)
