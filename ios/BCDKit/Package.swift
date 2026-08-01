// swift-tools-version: 6.0
import PackageDescription

// BCDKit — the app's core logic as a standalone SwiftPM package.
//
// Deliberately builds on the macOS *host* (no Xcode, no iOS SDK required) so the whole
// core is verifiable with `swift build && swift test` on any machine, including this
// Intel Mac. Platform-specific integrations (VisionKit scanner, Foundation Models) are
// behind `#if canImport(...)` so they compile into the iOS app but no-op on the host.
let package = Package(
    name: "BCDKit",
    platforms: [
        .iOS(.v18),   // deployment floor; iOS 26 features are runtime/#if gated
        .macOS(.v13), // host build for tests
    ],
    products: [
        .library(name: "BCDKit", targets: ["BCDKit"]),
    ],
    targets: [
        .target(name: "BCDKit"),
        .testTarget(name: "BCDKitTests", dependencies: ["BCDKit"]),
    ]
)
