import Foundation

/// The on-device detection seam. The app drives a `ScanEngine`; the coordinator tracks
/// what it emits into objects and the HUD renders those. Two real implementations exist
/// on iOS — VisionKit's `DataScannerViewController` (text + barcode, no segmentation) and
/// an AVCapture + Vision pipeline that also segments cans/bottles — and both live behind
/// `#if canImport(...)`. On the macOS host (tests, this Intel Mac) the mock is used,
/// keeping the whole tracking/resolution/overlay pipeline verifiable without a device.
public protocol ScanEngine: AnyObject, Sendable {
    /// Stream of detection frames. Each element is the full set currently in view, so the
    /// HUD can diff and re-anchor overlays.
    var frames: AsyncStream<[DetectedText]> { get }
    func start() async
    func stop()

    /// A still of what the camera is seeing right now, JPEG-encoded, or nil when this engine
    /// cannot produce one. The scan path only asks for it when text alone has failed, so an
    /// engine that returns nil simply never escalates.
    func captureFrame() async -> Data?

    /// Width/height of the content the boxes are normalized to, when that differs from
    /// the view (an aspect-fill camera buffer). nil means boxes are already in view space.
    var contentAspect: Double? { get }
}

public extension ScanEngine {
    // The mock and any future text-only engine inherit this: no picture, no escalation.
    func captureFrame() async -> Data? { nil }
    var contentAspect: Double? { nil }
}

/// Engines that segment the scene adopt this. `latestRegions` is what the segmenter last
/// saw — the coarse stage's "these are the cans" — and the coordinator reads it alongside
/// each text frame so lines can be assigned to the object they sit on.
public protocol RegionProvider: AnyObject {
    var latestRegions: [ObjectRegion]? { get }
}

/// The fine stage's hook: read text again, carefully, on one object's crop. Engines that
/// hold frames (or can capture a still) adopt this; the coordinator calls it only for
/// objects the coarse pass could not resolve.
public protocol FineTextReader: AnyObject, Sendable {
    func readText(in box: BoundingBox) async -> [String]
}

/// Engines whose recognizer accepts a custom vocabulary adopt this; the app feeds it the
/// catalog lexicon from `/v1/lexicon`.
public protocol LexiconConsumer: AnyObject {
    var lexicon: [String] { get set }
}

/// Deterministic engine for tests and previews. Emits a scripted sequence of frames.
public final class MockScanEngine: ScanEngine, RegionProvider, @unchecked Sendable {
    private let scripted: [[DetectedText]]
    private var continuation: AsyncStream<[DetectedText]>.Continuation?
    public let frames: AsyncStream<[DetectedText]>
    /// Regions to report alongside every frame (a scripted segmenter).
    public var latestRegions: [ObjectRegion]?
    /// Optional fine-read script: what a closer look at any object "reads".
    public var fineReads: [String] = []
    public private(set) var fineReadCount = 0

    public init(scripted: [[DetectedText]], regions: [ObjectRegion]? = nil) {
        self.scripted = scripted
        self.latestRegions = regions
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

extension MockScanEngine: FineTextReader {
    public func readText(in box: BoundingBox) async -> [String] {
        fineReadCount += 1
        return fineReads
    }
}
