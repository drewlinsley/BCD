import Foundation

/// The on-device detection seam. The app drives a `ScanEngine`; the HUD renders whatever
/// detections it emits. VisionKit's `DataScannerViewController` is the real
/// implementation on iOS 18+, but it only exists on iOS — so it lives behind
/// `#if canImport(VisionKit)`. On the macOS host (tests, this Intel Mac) the mock is used,
/// keeping the whole ranking/overlay pipeline verifiable without a device.
public protocol ScanEngine: AnyObject, Sendable {
    /// Stream of detection frames. Each element is the full set currently in view, so the
    /// HUD can diff and re-anchor overlays.
    var frames: AsyncStream<[DetectedText]> { get }
    func start() async
    func stop()
}

/// Deterministic engine for tests and previews. Emits a scripted sequence of frames.
public final class MockScanEngine: ScanEngine, @unchecked Sendable {
    private let scripted: [[DetectedText]]
    private var continuation: AsyncStream<[DetectedText]>.Continuation?
    public let frames: AsyncStream<[DetectedText]>

    public init(scripted: [[DetectedText]]) {
        self.scripted = scripted
        var cont: AsyncStream<[DetectedText]>.Continuation!
        self.frames = AsyncStream { cont = $0 }
        self.continuation = cont
    }

    public func start() async {
        for frame in scripted {
            continuation?.yield(frame)
        }
        continuation?.finish()
    }

    public func stop() { continuation?.finish() }
}
