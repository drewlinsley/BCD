import Foundation

/// The on-device detection seam. The app drives a `ScanEngine`; the coordinator tracks
/// what it emits into objects and the HUD renders those. Two real implementations exist
/// on iOS — VisionKit's `DataScannerViewController` (text + barcode, no segmentation) and
/// an AVCapture + Vision pipeline that also segments cans/bottles — and both live behind
/// `#if canImport(...)`. On the macOS host (tests, this Intel Mac) the mock is used,
/// keeping the whole tracking/resolution/overlay pipeline verifiable without a device.
public protocol ScanEngine: AnyObject, Sendable {
    /// Stream of frames. Each element is the full set currently in view, so the tracker
    /// can diff and re-anchor.
    var frames: AsyncStream<ScanFrame> { get }
    func start() async
    func stop()

    /// Width/height of the content the boxes are normalized to, when that differs from
    /// the view (an aspect-fill camera buffer). nil means boxes are already in view space.
    var contentAspect: Double? { get }
}

public extension ScanEngine {
    var contentAspect: Double? { nil }
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
public final class MockScanEngine: ScanEngine, @unchecked Sendable {
    private let scripted: [ScanFrame]
    private var continuation: AsyncStream<ScanFrame>.Continuation?
    public let frames: AsyncStream<ScanFrame>
    /// Optional fine-read script: object box -> extra lines "read" on a closer look.
    public var fineReads: [String] = []
    public private(set) var fineReadCount = 0

    public init(frames: [ScanFrame]) {
        self.scripted = frames
        var cont: AsyncStream<ScanFrame>.Continuation!
        self.frames = AsyncStream { cont = $0 }
        self.continuation = cont
    }

    public convenience init(scripted: [[DetectedText]]) {
        self.init(frames: scripted.map { ScanFrame(texts: $0) })
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
