import SwiftUI
import BCDKit
#if canImport(VisionKit) && os(iOS)
import VisionKit
import AVFoundation
#endif

// The camera HUD — the whole product thesis in one screen. Live detections become
// overlays anchored to their bounding boxes, color-coded by predicted enjoyment. A
// persistent chat bar routes natural-language asks against the items currently in frame.

struct ScanView: View {
    @EnvironmentObject var env: AppEnvironment
    @StateObject private var model = ScanViewModel()
    @State private var ask: String = ""
    @State private var selected: ScoredCandidate?

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraLayer(engine: model.engine)  // live DataScanner on device; gradient in Sim/host
                .ignoresSafeArea()

            GeometryReader { geo in
                ForEach(model.overlays) { overlay in
                    OverlayChip(candidate: overlay.candidate)
                        .position(x: overlay.anchor.x * geo.size.width,
                                  y: overlay.anchor.y * geo.size.height)
                        .onTapGesture { selected = overlay.candidate }
                }
            }

            VStack(spacing: 8) {
                if let latency = model.lastLatencyMs {
                    Text("\(model.overlays.count) in view · \(Int(latency))ms")
                        .font(.caption).foregroundStyle(.secondary)
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                chatBar
            }
            .padding()
        }
        .task { model.configure(env: env); model.start() }
        .onDisappear { model.stop() }
        .sheet(item: $selected) { cand in
            ProductDetailView(candidate: cand)
        }
    }

    private var chatBar: some View {
        HStack {
            Image(systemName: "sparkles")
            TextField("cheapest hazy here · nothing over 6%", text: $ask)
                .textFieldStyle(.plain)
                .submitLabel(.search)
                .onSubmit { Task { await model.applyAsk(ask) } }
            if !ask.isEmpty {
                Button { ask = ""; Task { await model.applyAsk("") } } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

/// One anchored overlay: a candidate + where to draw it (normalized 0-1).
struct HUDOverlay: Identifiable {
    let id: String
    let candidate: ScoredCandidate
    let anchor: CGPoint
}

struct OverlayChip: View {
    let candidate: ScoredCandidate

    private var tint: Color {
        guard let s = candidate.personalScore else { return .gray }
        return s > 0.75 ? .green : (s > 0.5 ? .yellow : .orange)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(candidate.resolved.product.name).font(.subheadline.bold()).lineLimit(1)
            HStack(spacing: 6) {
                if let s = candidate.personalScore {
                    Label("\(Int(s * 100))", systemImage: "hand.thumbsup.fill").font(.caption2)
                }
                if candidate.coldStart {
                    Image(systemName: "flask.fill").font(.caption2)  // scored from chemistry
                }
            }
            if let reason = candidate.reason {
                Text(reason).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(tint, lineWidth: 2))
        .frame(maxWidth: 180)
    }
}

@MainActor
final class ScanViewModel: ObservableObject {
    @Published var overlays: [HUDOverlay] = []
    @Published var lastLatencyMs: Double?
    /// The engine the coordinator consumes. Exposed so the camera layer can present *this*
    /// engine's scanner view — it must be the same instance, or detections wouldn't reach the HUD.
    @Published private(set) var engine: ScanEngine?

    private var coordinator: ScanCoordinator?
    private var env: AppEnvironment?
    private var lastDetections: [String: CGPoint] = [:]

    func configure(env: AppEnvironment) {
        guard coordinator == nil else { return }
        self.env = env
        let engine = env.makeScanEngine()
        self.engine = engine
        let coord = ScanCoordinator(engine: engine, api: env.api,
                                    telemetry: env.telemetry)
        self.coordinator = coord
        // Re-render overlays whenever the coordinator publishes new candidates.
        Task { [weak self] in
            guard let self else { return }
            for await _ in coord.$candidates.values {
                self.rebuildOverlays(from: coord.candidates)
                self.lastLatencyMs = coord.lastLatencyMs
            }
        }
    }

    func start() { coordinator?.start() }
    func stop() { coordinator?.stop() }

    func applyAsk(_ ask: String) async {
        guard let env, let coord = coordinator else { return }
        guard !ask.isEmpty else { rebuildOverlays(from: coord.candidates); return }
        let order = (try? await env.llm.rerank(coord.candidates, for: ask)) ?? []
        let ranked = order.compactMap { id in
            coord.candidates.first { $0.resolved.product.id == id }
        }
        rebuildOverlays(from: ranked.isEmpty ? coord.candidates : ranked)
    }

    private func rebuildOverlays(from candidates: [ScoredCandidate]) {
        overlays = candidates.enumerated().map { idx, cand in
            // Anchor to the detection's box if we have it; else fan out down the frame.
            let anchor = CGPoint(x: 0.25 + 0.5 * Double(idx % 2),
                                 y: 0.25 + 0.12 * Double(idx))
            return HUDOverlay(id: cand.id, candidate: cand, anchor: anchor)
        }
    }
}

/// Camera layer. On a real device it presents VisionKit's `DataScannerViewController` (live
/// text + barcode) driven by the shared engine; in the Simulator or on the host — no camera —
/// it falls back to a neutral gradient so the HUD stays previewable.
struct CameraLayer: View {
    var engine: ScanEngine?

    var body: some View {
        #if canImport(VisionKit) && os(iOS)
        if #available(iOS 18.0, *), DataScannerViewController.isSupported,
           let vk = engine as? VisionKitScanEngine {
            DataScannerView(engine: vk)
        } else {
            placeholder
        }
        #else
        placeholder
        #endif
    }

    private var placeholder: some View {
        LinearGradient(colors: [.black, .gray.opacity(0.6)],
                       startPoint: .top, endPoint: .bottom)
    }
}

#if canImport(VisionKit) && os(iOS)
/// Presents the engine's `DataScannerViewController` and drives its lifecycle: request camera
/// access (this is what makes iOS show the permission prompt), then start scanning. Detections
/// flow out through the engine's delegate → `frames` → the coordinator, which the HUD renders.
@available(iOS 18.0, *)
struct DataScannerView: UIViewControllerRepresentable {
    let engine: VisionKitScanEngine

    func makeUIViewController(context: Context) -> DataScannerViewController {
        engine.makeScanner()
    }

    func updateUIViewController(_ scanner: DataScannerViewController, context: Context) {
        guard !context.coordinator.started else { return }
        context.coordinator.started = true
        let engine = self.engine  // @unchecked Sendable — safe to hand to the async closure
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { return }
            Task { await engine.start() }  // hops to @MainActor, starts the created scanner
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var started = false }
}
#endif
