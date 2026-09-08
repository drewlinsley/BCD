import Foundation

#if canImport(Vision) && canImport(AVFoundation) && os(iOS)
import AVFoundation
import Vision
import CoreImage
import CoreVideo
import UIKit

/// The fuller on-device pipeline: our own `AVCaptureSession` feeding Vision directly.
/// Compared with `DataScannerViewController` this buys three things the scan problem
/// needs:
///
///   * **Coarse segmentation.** Every N frames, `GenerateForegroundInstanceMaskRequest`
///     lifts the salient objects (the cans and bottles in a fridge or on a bar) and
///     `ClassifyImageRequest` on each instance keeps only drink containers. Those boxes
///     become `ObjectRegion`s; OCR lines are assigned to the region they fall in, so a
///     can is one object no matter how its label wraps.
///   * **Raw OCR.** `RecognizeTextRequest` with language correction *off* for the fast
///     pass. The correction is what turns "ALCHEMIST" into "Chemist"; the server matcher
///     is typo-tolerant and would rather see the raw glyphs.
///   * **Frame access** for the fine stage: `readText(in:)` crops the last frame and
///     runs the careful `VisionTextReader` (accurate, lexicon-steered) on that object.
///
/// Boxes are normalized to the camera buffer, which the preview shows aspect-fill —
/// `contentAspect` tells the HUD how to map them. Compiled only into the iOS app.
///
/// Enable with `BCD_SCAN_ENGINE=vision` (see Local.xcconfig); the VisionKit scanner stays
/// the default until this has been exercised on a device.
@available(iOS 18.0, *)
public final class VisionFrameScanEngine: NSObject, ScanEngine, @unchecked Sendable {
    public struct Config: Sendable {
        /// Run OCR + barcode on every Nth frame (30 fps in, ~10-15 Vision passes/s out).
        public var ocrEveryNFrames = 2
        /// Run instance segmentation on every Nth frame — it costs ~10x an OCR pass.
        public var segmentEveryNFrames = 12
        /// Classify each lifted instance and drop the ones that aren't drink containers.
        public var classifyRegions = true
        public var maxRegions = 8
        /// Ignore instances smaller than this fraction of the frame (caps, coasters).
        public var minRegionArea = 0.008
        public init() {}
    }

    public let frames: AsyncStream<[DetectedText]>
    public let session = AVCaptureSession()
    public var contentAspect: Double? { lock.withLock { _contentAspect } }

    private var continuation: AsyncStream<[DetectedText]>.Continuation?
    private let config: Config
    private let sessionQueue = DispatchQueue(label: "bcd.capture.session")
    private let videoQueue = DispatchQueue(label: "bcd.capture.video", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let lock = NSLock()
    private var _lexicon: [String] = []
    private var _contentAspect: Double?
    private var configured = false
    private var frameIndex = 0
    private var busy = false
    private var lastRegions: [ObjectRegion] = []
    private var lastFrame: PixelBufferBox?

    public init(config: Config = Config()) {
        self.config = config
        var cont: AsyncStream<[DetectedText]>.Continuation!
        self.frames = AsyncStream { cont = $0 }
        self.continuation = cont
        super.init()
    }

    public func start() async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            sessionQueue.async {
                self.configureIfNeeded()
                if !self.session.isRunning { self.session.startRunning() }
                cont.resume()
            }
        }
    }

    public func stop() {
        continuation?.finish()
        sessionQueue.async { if self.session.isRunning { self.session.stopRunning() } }
    }

    /// A preview layer bound to this engine's session; the HUD sits on top of it.
    @MainActor public func makePreviewLayer() -> AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        return layer
    }

    // MARK: - capture setup

    private func configureIfNeeded() {
        guard !configured else { return }
        configured = true
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        // 1080p is plenty for label text at arm's length and keeps Vision passes fast.
        if session.canSetSessionPreset(.hd1920x1080) { session.sessionPreset = .hd1920x1080 }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video,
                                                   position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { return }
        session.addInput(input)
        // Continuous autofocus matters more than usual: label text is small.
        if (try? device.lockForConfiguration()) != nil {
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            device.unlockForConfiguration()
        }
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                    kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else { return }
        session.addOutput(output)
        // Deliver portrait-up buffers so Vision boxes and the preview agree without
        // per-frame orientation bookkeeping.
        if let conn = output.connection(with: .video), conn.isVideoRotationAngleSupported(90) {
            conn.videoRotationAngle = 90
        }
    }

    // MARK: - per-frame analysis

    private func process(_ box: PixelBufferBox, segment: Bool) async {
        let buffer = box.buffer
        let width = Double(CVPixelBufferGetWidth(buffer))
        let height = Double(CVPixelBufferGetHeight(buffer))
        if width > 0, height > 0 { lock.withLock { _contentAspect = width / height } }

        var textReq = RecognizeTextRequest()
        textReq.recognitionLevel = .fast
        textReq.usesLanguageCorrection = false
        let barcodeReq = DetectBarcodesRequest()

        var texts: [DetectedText] = []
        let unit = CGSize(width: 1, height: 1)
        if let observations = try? await textReq.perform(on: buffer) {
            for o in observations {
                guard let top = o.topCandidates(1).first, !top.string.isEmpty else { continue }
                let r = o.boundingBox.toImageCoordinates(unit, origin: .upperLeft)
                texts.append(DetectedText(text: top.string, kind: "text",
                                          x: r.minX, y: r.minY, w: r.width, h: r.height,
                                          confidence: Double(top.confidence)))
            }
        }
        if let codes = try? await barcodeReq.perform(on: buffer) {
            for c in codes {
                guard let payload = c.payloadString, !payload.isEmpty else { continue }
                let r = c.boundingBox.toImageCoordinates(unit, origin: .upperLeft)
                texts.append(DetectedText(text: payload, kind: "barcode",
                                          symbology: String(describing: c.symbology),
                                          x: r.minX, y: r.minY, w: r.width, h: r.height))
            }
        }

        if segment, let fresh = try? await segmentObjects(in: buffer) {
            lock.withLock { lastRegions = fresh }
        }
        // Regions travel beside the text stream (`RegionProvider`), read by the coordinator
        // as it ingests each frame, so a text-only engine and this one share one protocol.
        continuation?.yield(texts)
    }

    /// Coarse stage: lift foreground instances, keep the drink containers, box them.
    private func segmentObjects(in buffer: CVPixelBuffer) async throws -> [ObjectRegion] {
        let request = GenerateForegroundInstanceMaskRequest()
        let maybe: InstanceMaskObservation? = try await request.perform(on: buffer)
        guard let observation = maybe else { return [] }
        let boxes = Self.instanceBoxes(of: observation)
        let handler = ImageRequestHandler(buffer)
        var regions: [ObjectRegion] = []
        let ordered = boxes.sorted { $0.value.area > $1.value.area }
        for (index, box) in ordered where box.area >= config.minRegionArea {
            if regions.count >= config.maxRegions { break }
            var label: String? = "object"
            var confidence: Double?
            if config.classifyRegions,
               let crop = try? observation.generateMaskedImage(
                   for: IndexSet(integer: index), imageFrom: handler,
                   croppedToInstancesExtent: true),
               let classes = try? await ClassifyImageRequest().perform(on: crop) {
                if let hit = classes.first(where: { Self.isDrinkContainer($0.identifier) }),
                   hit.confidence >= 0.1 {
                    label = Self.containerLabel(hit.identifier)
                    confidence = Double(hit.confidence)
                } else if let top = classes.first, top.confidence >= 0.5 {
                    continue  // confidently something else: a hand, a face, a menu
                }
            }
            regions.append(ObjectRegion(id: "seg-\(index)", box: box, label: label,
                                        confidence: confidence))
        }
        return regions
    }

    /// Bounding box per instance.
    ///
    /// Vision hands back one mask at a time (`generateMask(for:)`) rather than a single
    /// index-coded buffer, so each instance is lifted on its own and reduced to the extent of
    /// its non-zero pixels. `allInstancesMask` is the union of every instance and cannot tell
    /// them apart, which is why it is not what this uses.
    static func instanceBoxes(of observation: InstanceMaskObservation) -> [Int: BoundingBox] {
        var out: [Int: BoundingBox] = [:]
        for index in observation.allInstances {
            guard let mask = try? observation.generateMask(for: IndexSet(integer: index)),
                  let box = boundingBox(ofNonZeroIn: mask) else { continue }
            out[index] = box
        }
        return out
    }

    /// Extent of a mask's non-zero pixels, normalized 0-1. Vision returns a one-component
    /// mask; whether that component is 8-bit or float depends on the request, so both are
    /// read rather than assuming the byte layout and silently boxing noise.
    static func boundingBox(ofNonZeroIn mask: CVPixelBuffer) -> BoundingBox? {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(mask) else { return nil }
        let w = CVPixelBufferGetWidth(mask), h = CVPixelBufferGetHeight(mask)
        guard w > 0, h > 0 else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(mask)
        let isFloat = CVPixelBufferGetPixelFormatType(mask) == kCVPixelFormatType_OneComponent32Float
        var minX = w, minY = h, maxX = -1, maxY = -1
        for y in 0..<h {
            let row = base + y * stride
            for x in 0..<w {
                let on: Bool = isFloat
                    ? row.assumingMemoryBound(to: Float.self)[x] > 0.5
                    : row.assumingMemoryBound(to: UInt8.self)[x] != 0
                guard on else { continue }
                if x < minX { minX = x }
                if x > maxX { maxX = x }
                if y < minY { minY = y }
                if y > maxY { maxY = y }
            }
        }
        guard maxX >= minX, maxY >= minY else { return nil }
        return BoundingBox(x: Double(minX) / Double(w), y: Double(minY) / Double(h),
                           w: Double(maxX - minX + 1) / Double(w),
                           h: Double(maxY - minY + 1) / Double(h))
    }

    static let containerWords: Set<String> = [
        "beer", "bottle", "can", "beverage", "drink", "wine", "liquor", "alcohol", "soda",
        "ale", "lager", "whiskey", "whisky", "bourbon", "champagne", "cocktail", "brew",
        "keg", "growler", "cider", "spirits", "flask", "jar",
    ]

    static func isDrinkContainer(_ identifier: String) -> Bool {
        let parts = identifier.lowercased().split { !$0.isLetter }.map(String.init)
        return parts.contains { containerWords.contains($0) }
    }

    static func containerLabel(_ identifier: String) -> String {
        let parts = Set(identifier.lowercased().split { !$0.isLetter }.map(String.init))
        if parts.contains("can") { return "can" }
        if parts.contains("bottle") { return "bottle" }
        return "container"
    }
}

/// `CVPixelBuffer` isn't `Sendable`; the capture delegate hands one buffer at a time to
/// exactly one analysis task, so wrapping it is sound.
struct PixelBufferBox: @unchecked Sendable {
    let buffer: CVPixelBuffer
}

@available(iOS 18.0, *)
extension VisionFrameScanEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                              from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        frameIndex += 1  // videoQueue is serial
        let index = frameIndex
        let box = PixelBufferBox(buffer: pixelBuffer)
        lock.withLock { lastFrame = box }
        guard index % config.ocrEveryNFrames == 0 else { return }
        let shouldRun: Bool = lock.withLock {
            if busy { return false }
            busy = true
            return true
        }
        guard shouldRun else { return }  // still analysing the previous frame: drop this one
        let segment = index % config.segmentEveryNFrames == 0
        Task.detached(priority: .userInitiated) { [self] in
            await self.process(box, segment: segment)
            self.lock.withLock { self.busy = false }
        }
    }
}

@available(iOS 18.0, *)
extension VisionFrameScanEngine: RegionProvider {
    /// What the segmenter last saw. Refreshed every `segmentEveryNFrames`; between
    /// refreshes the previous boxes stand, which is what the tracker's box smoothing wants.
    public var latestRegions: [ObjectRegion]? {
        let r = lock.withLock { lastRegions }
        return r.isEmpty ? nil : r
    }
}

extension VisionFrameScanEngine: LexiconConsumer {
    public var lexicon: [String] {
        get { lock.withLock { _lexicon } }
        set { lock.withLock { _lexicon = newValue } }
    }
}

@available(iOS 18.0, *)
extension VisionFrameScanEngine: FineTextReader {
    /// Fine stage: crop the most recent frame to the object and read it carefully.
    public func readText(in box: BoundingBox) async -> [String] {
        guard let frame = lock.withLock({ lastFrame }) else { return [] }
        let buffer = frame.buffer
        let W = Double(CVPixelBufferGetWidth(buffer)), H = Double(CVPixelBufferGetHeight(buffer))
        let crop = box.expanded(by: 0.04)
        // CIImage coordinates have their origin at the bottom-left.
        let rect = CGRect(x: crop.x * W, y: (1 - crop.maxY) * H, width: crop.w * W, height: crop.h * H)
            .integral
        let ci = CIImage(cvPixelBuffer: buffer)
        guard rect.width >= 8, rect.height >= 8,
              let cg = ciContext.createCGImage(ci, from: rect.intersection(ci.extent)) else { return [] }
        let reader = VisionTextReader(lexicon: lexicon)
        let texts = (try? await reader.read(cg)) ?? []
        return texts.map(\.text)
    }
}
#endif
