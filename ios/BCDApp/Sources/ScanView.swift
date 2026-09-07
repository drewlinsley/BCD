import SwiftUI
import BCDKit
#if canImport(VisionKit) && os(iOS)
import VisionKit
import AVFoundation
#endif

// The camera HUD — the whole product thesis in one screen. Tracked objects (cans,
// bottles) become overlays anchored to their boxes, color-coded by predicted enjoyment.
// Only *resolved* objects get a name. An ambiguous object shows a "which one?" chip whose
// tap lets the user choose (our best training label); anything else is a faint outline.
// A persistent chat bar routes natural-language asks against the items currently in frame.

struct ScanView: View {
    @EnvironmentObject var env: AppEnvironment
    @StateObject private var model = ScanViewModel()
    @State private var ask: String = ""
    @State private var selected: ScoredCandidate?
    @State private var choosing: HUDOverlay?

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraLayer(engine: model.engine)  // live camera on device; gradient in Sim/host
                .ignoresSafeArea()

            GeometryReader { geo in
                ForEach(model.overlays) { overlay in
                    let rect = overlay.rect(in: geo.size, mapper: model.mapper)
                    switch overlay.kind {
                    case .resolved(let candidate, let stage):
                        OverlayChip(candidate: candidate, stage: stage)
                            .opacity(model.isHighlighted(candidate) ? 1 : 0.35)
                            .position(x: rect.midX, y: max(48, rect.minY - 30))
                            .onTapGesture { selected = candidate }
                    case .ambiguous(let shortlist):
                        AmbiguousChip(count: shortlist.count)
                            .position(x: rect.midX, y: max(48, rect.minY - 22))
                            .onTapGesture { choosing = overlay }
                    case .pending:
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(style: StrokeStyle(lineWidth: 1, dash: [5, 5]))
                            .foregroundStyle(.white.opacity(0.35))
                            .frame(width: rect.width, height: rect.height)
                            .position(x: rect.midX, y: rect.midY)
                    }
                }
            }

            VStack(spacing: 8) {
                Text(model.statusLine)
                    .font(.caption).foregroundStyle(.secondary)
                    .padding(.horizontal, 10).padding(.vertical, 4)
                    .background(.ultraThinMaterial, in: Capsule())
                chatBar
            }
            .padding()
        }
        .task { model.configure(env: env); model.start() }
        .onDisappear { model.stop() }
        .sheet(item: $selected) { cand in
            ProductDetailView(candidate: cand)
        }
        .sheet(item: $choosing) { overlay in
            ChooseProductSheet(overlay: overlay) { picked in
                model.confirm(objectId: overlay.id, candidate: picked)
                choosing = nil
            }
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

/// One anchored overlay: what to draw and where (a box in the engine's content space).
struct HUDOverlay: Identifiable {
    enum Kind {
        case resolved(ScoredCandidate, ScanCoordinator.Stage)
        case ambiguous([ScoredCandidate])
        case pending
    }
    let id: String
    let box: BoundingBox
    let kind: Kind
    let texts: [String]

    func rect(in size: CGSize, mapper: AspectFillMapper?) -> CGRect {
        let viewAspect = size.height > 0 ? Double(size.width / size.height) : 1
        let b = mapper?.toView(box, viewAspect: viewAspect) ?? box
        return CGRect(x: b.x * size.width, y: b.y * size.height,
                      width: b.w * size.width, height: b.h * size.height)
    }
}

struct OverlayChip: View {
    let candidate: ScoredCandidate
    var stage: ScanCoordinator.Stage = .coarse

    private var tint: Color {
        guard let s = candidate.personalScore else { return .gray }
        return s > 0.75 ? .green : (s > 0.5 ? .yellow : .orange)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                Text(candidate.resolved.product.name).font(.subheadline.bold()).lineLimit(1)
                if stage == .llmPick {
                    Image(systemName: "sparkles").font(.caption2).foregroundStyle(.secondary)
                } else if stage == .user {
                    Image(systemName: "person.fill").font(.caption2).foregroundStyle(.secondary)
                }
            }
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

/// The honest chip: we know roughly what this is, not exactly. Tapping shows the shortlist.
struct AmbiguousChip: View {
    let count: Int
    var body: some View {
        Label("\(count) possible · tap", systemImage: "questionmark.circle")
            .font(.caption.bold())
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.regularMaterial, in: Capsule())
            .overlay(Capsule().stroke(.secondary, lineWidth: 1))
    }
}

struct ChooseProductSheet: View {
    let overlay: HUDOverlay
    let onPick: (ScoredCandidate) -> Void
    @Environment(\.dismiss) private var dismiss

    private var shortlist: [ScoredCandidate] {
        if case .ambiguous(let cands) = overlay.kind { return cands }
        return []
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(shortlist) { c in
                        Button { onPick(c) } label: {
                            VStack(alignment: .leading) {
                                Text(c.resolved.product.name).font(.headline)
                                Text(c.resolved.producer.name).font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                } header: {
                    Text("Which one is this?")
                } footer: {
                    if !overlay.texts.isEmpty {
                        Text("Read off the label: " + overlay.texts.joined(separator: " · "))
                    }
                }
                Section {
                    Button("None of these", role: .cancel) { dismiss() }
                }
            }
            .navigationTitle("Not sure")
            .navigationBarTitleDisplayMode(.inline)
        }
        .presentationDetents([.medium])
    }
}

@MainActor
final class ScanViewModel: ObservableObject {
    @Published var overlays: [HUDOverlay] = []
    @Published var lastLatencyMs: Double?
    @Published var trackedCount = 0
    @Published var resolvedCount = 0
    /// The engine the coordinator consumes. Exposed so the camera layer can present *this*
    /// engine's view — it must be the same instance, or detections wouldn't reach the HUD.
    @Published private(set) var engine: ScanEngine?
    /// Maps engine-space boxes onto the aspect-fill preview; nil when they already match.
    private(set) var mapper: AspectFillMapper?

    private var coordinator: ScanCoordinator?
    private var env: AppEnvironment?
    private var highlighted: Set<String>?   // product ids the chat-bar ask kept; nil = all

    var statusLine: String {
        var s = "\(resolvedCount) named · \(trackedCount) in view"
        if let ms = lastLatencyMs { s += " · \(Int(ms))ms" }
        return s
    }

    func configure(env: AppEnvironment) {
        guard coordinator == nil else { return }
        self.env = env
        let engine = env.makeScanEngine()
        self.engine = engine
        let coord = ScanCoordinator(engine: engine, api: env.api, llm: env.llm,
                                    telemetry: env.telemetry)
        self.coordinator = coord
        // Re-render overlays whenever the coordinator publishes new object state.
        Task { [weak self] in
            guard let self else { return }
            for await objs in coord.$objects.values {
                self.mapper = engine.contentAspect.map(AspectFillMapper.init)
                self.rebuildOverlays(from: objs)
                self.lastLatencyMs = coord.lastLatencyMs
            }
        }
        // Teach the recognizer the catalog's names — the cheapest fix for stylized type.
        Task {
            guard let consumer = engine as? LexiconConsumer,
                  let words = try? await env.api.fetchLexicon(), !words.isEmpty else { return }
            consumer.lexicon = words
        }
    }

    func start() { coordinator?.start() }
    func stop() { coordinator?.stop() }

    func confirm(objectId: String, candidate: ScoredCandidate) {
        coordinator?.confirm(objectId: objectId, candidate: candidate)
    }

    func isHighlighted(_ c: ScoredCandidate) -> Bool {
        highlighted?.contains(c.resolved.product.id) ?? true
    }

    func applyAsk(_ ask: String) async {
        guard let env, let coord = coordinator else { return }
        guard !ask.isEmpty else { highlighted = nil; rebuildOverlays(from: coord.objects); return }
        let kept = (try? await env.llm.rerank(coord.candidates, for: ask)) ?? []
        highlighted = kept.isEmpty ? nil : Set(kept)
        rebuildOverlays(from: coord.objects)
    }

    private func rebuildOverlays(from objects: [ScanCoordinator.SceneObject]) {
        var resolved = 0
        overlays = objects.enumerated().compactMap { idx, obj in
            // Box-less objects (mock engine) fan out down the frame so previews still work.
            let box = obj.anchored ? obj.box
                : BoundingBox(x: 0.15 + 0.4 * Double(idx % 2), y: 0.2 + 0.12 * Double(idx), w: 0.3, h: 0.08)
            switch obj.status {
            case .resolved(let c, let stage):
                resolved += 1
                return HUDOverlay(id: obj.id, box: box, kind: .resolved(c, stage), texts: obj.texts)
            case .ambiguous(let cands):
                return HUDOverlay(id: obj.id, box: box, kind: .ambiguous(cands), texts: obj.texts)
            case .tracking, .resolving, .unresolved:
                // Outline only once an object has proven it's really there.
                guard obj.anchored, obj.framesSeen >= 4 else { return nil }
                return HUDOverlay(id: obj.id, box: box, kind: .pending, texts: obj.texts)
            }
        }
        trackedCount = objects.count
        resolvedCount = resolved
    }
}

/// Camera layer. On a real device it presents whichever engine the composition root chose
/// — VisionKit's `DataScannerViewController` or the AVCapture preview behind
/// `VisionFrameScanEngine`; in the Simulator or on the host — no camera — it falls back
/// to a neutral gradient so the HUD stays previewable.
struct CameraLayer: View {
    var engine: ScanEngine?

    var body: some View {
        #if canImport(VisionKit) && os(iOS)
        if #available(iOS 18.0, *) {
            if let vk = engine as? VisionKitScanEngine, DataScannerViewController.isSupported {
                DataScannerView(engine: vk)
            } else if let vf = engine as? VisionFrameScanEngine {
                CapturePreviewView(engine: vf)
            } else {
                placeholder
            }
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

/// The AVCapture preview for `VisionFrameScanEngine`. Same lifecycle: ask, then start.
@available(iOS 18.0, *)
struct CapturePreviewView: UIViewRepresentable {
    let engine: VisionFrameScanEngine

    func makeUIView(context: Context) -> PreviewHostView {
        let view = PreviewHostView()
        view.attach(engine.makePreviewLayer())
        return view
    }

    func updateUIView(_ view: PreviewHostView, context: Context) {
        guard !context.coordinator.started else { return }
        context.coordinator.started = true
        let engine = self.engine
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { return }
            Task { await engine.start() }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var started = false }
}

final class PreviewHostView: UIView {
    private var previewLayer: CALayer?
    func attach(_ layer: CALayer) {
        previewLayer = layer
        self.layer.addSublayer(layer)
        layer.frame = bounds
    }
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
}
#endif
