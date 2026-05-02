import AVFoundation
import CoreImage
import Darwin
import Foundation
import Network

let logPath = "/tmp/ExpressoCameraBridge.log"
freopen(logPath, "w", stdout)
freopen(logPath, "a", stderr)
setbuf(stdout, nil)

let endpoint = ProcessInfo.processInfo.environment["EXPRESSO_FRAME_ENDPOINT"].flatMap(URL.init(string:))
let controlEndpoint = ProcessInfo.processInfo.environment["EXPRESSO_CONTROL_ENDPOINT"].flatMap(URL.init(string:))
let streamPort = UInt16(ProcessInfo.processInfo.environment["EXPRESSO_STREAM_PORT"] ?? "3988") ?? 3988
let streamPath = ProcessInfo.processInfo.environment["EXPRESSO_STREAM_PATH"] ?? "/stream.mjpg"
let maxFps = Double(ProcessInfo.processInfo.environment["EXPRESSO_CAMERA_FPS"] ?? "30") ?? 30.0
let uploadWidth = Double(ProcessInfo.processInfo.environment["EXPRESSO_UPLOAD_WIDTH"] ?? "1280") ?? 1280.0
let uploadHeight = Double(ProcessInfo.processInfo.environment["EXPRESSO_UPLOAD_HEIGHT"] ?? "720") ?? 720.0
let jpegQuality = Double(ProcessInfo.processInfo.environment["EXPRESSO_JPEG_QUALITY"] ?? "0.70") ?? 0.70
let preferredCameraName = ProcessInfo.processInfo.environment["EXPRESSO_CAMERA_NAME"]?.trimmingCharacters(in: .whitespacesAndNewlines)

func log(_ message: String) {
    print("[\(Date())] \(message)")
}

extension Data {
    mutating func appendUtf8(_ string: String) {
        append(Data(string.utf8))
    }
}

final class MjpegServer {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "ExpressoCameraBridge.mjpeg")
    private var clients: [UUID: NWConnection] = [:]
    private let boundary = "expressoframe"
    private let path: String

    init(port: UInt16, path: String) throws {
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            throw NSError(domain: "ExpressoCameraBridge", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid stream port \(port)"])
        }
        self.listener = try NWListener(using: .tcp, on: nwPort)
        self.path = path.hasPrefix("/") ? path : "/" + path
    }

    func start() {
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.stateUpdateHandler = { state in
            log("MJPEG server state: \(state)")
        }
        listener.start(queue: queue)
    }

    func publish(_ jpeg: Data) {
        queue.async {
            guard !self.clients.isEmpty else {
                return
            }
            var part = Data()
            part.appendUtf8("--\(self.boundary)\r\n")
            part.appendUtf8("Content-Type: image/jpeg\r\n")
            part.appendUtf8("Content-Length: \(jpeg.count)\r\n\r\n")
            part.append(jpeg)
            part.appendUtf8("\r\n")

            for (id, connection) in self.clients {
                connection.send(content: part, completion: .contentProcessed { [weak self] error in
                    if let error {
                        log("MJPEG client send failed: \(error.localizedDescription)")
                        self?.removeClient(id)
                    }
                })
            }
        }
    }

    private func accept(_ connection: NWConnection) {
        let id = UUID()
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                self?.removeClient(id)
            default:
                break
            }
        }
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] data, _, _, error in
            guard let self else {
                connection.cancel()
                return
            }
            if let error {
                log("MJPEG request read failed: \(error.localizedDescription)")
                connection.cancel()
                return
            }
            let request = data.map { String(decoding: $0, as: UTF8.self) } ?? ""
            if request.hasPrefix("GET \(self.path) ") {
                self.sendStreamHeaders(connection, id: id)
            } else if request.hasPrefix("GET /health ") {
                self.sendPlainText(connection, status: "200 OK", body: "ok\n")
            } else {
                self.sendPlainText(
                    connection,
                    status: "404 Not Found",
                    body: "Expresso camera stream is at \(self.path)\n"
                )
            }
        }
    }

    private func sendStreamHeaders(_ connection: NWConnection, id: UUID) {
        let headers = """
        HTTP/1.1 200 OK\r
        Cache-Control: no-store, no-cache, must-revalidate\r
        Pragma: no-cache\r
        Connection: close\r
        Content-Type: multipart/x-mixed-replace; boundary=\(boundary)\r
        \r

        """
        connection.send(content: Data(headers.utf8), completion: .contentProcessed { [weak self] error in
            if let error {
                log("MJPEG header send failed: \(error.localizedDescription)")
                connection.cancel()
                return
            }
            self?.clients[id] = connection
            log("MJPEG client connected. Clients: \(self?.clients.count ?? 0)")
        })
    }

    private func sendPlainText(_ connection: NWConnection, status: String, body: String) {
        let payload = """
        HTTP/1.1 \(status)\r
        Content-Type: text/plain; charset=utf-8\r
        Content-Length: \(body.utf8.count)\r
        Connection: close\r
        \r
        \(body)
        """
        connection.send(content: Data(payload.utf8), completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    private func removeClient(_ id: UUID) {
        queue.async {
            if let connection = self.clients.removeValue(forKey: id) {
                connection.cancel()
                log("MJPEG client disconnected. Clients: \(self.clients.count)")
            }
        }
    }
}

func requestCameraAccess() -> Bool {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
        return true
    case .notDetermined:
        let semaphore = DispatchSemaphore(value: 0)
        var granted = false
        AVCaptureDevice.requestAccess(for: .video) { ok in
            granted = ok
            semaphore.signal()
        }
        semaphore.wait()
        return granted
    default:
        return false
    }
}

let mjpegServer: MjpegServer? = {
    do {
        return try MjpegServer(port: streamPort, path: streamPath)
    } catch {
        log("Cannot start MJPEG server on port \(streamPort): \(error)")
        return nil
    }
}()

func chooseDevice() -> AVCaptureDevice? {
    let discovery = AVCaptureDevice.DiscoverySession(
        deviceTypes: [.builtInWideAngleCamera, .external],
        mediaType: .video,
        position: .unspecified
    )
    let devices = discovery.devices
    log("Video devices: \(devices.map { "\($0.localizedName) [\($0.deviceType.rawValue)]" }.joined(separator: ", "))")

    if let preferredCameraName, !preferredCameraName.isEmpty {
        if let preferred = devices.first(where: {
            $0.localizedName.localizedCaseInsensitiveContains(preferredCameraName)
                || $0.uniqueID.localizedCaseInsensitiveContains(preferredCameraName)
        }) {
            log("Using EXPRESSO_CAMERA_NAME override: \(preferredCameraName)")
            return preferred
        }
        log("EXPRESSO_CAMERA_NAME override did not match any device: \(preferredCameraName)")
    }

    let builtIns = devices.filter { $0.deviceType == .builtInWideAngleCamera }
    if let namedBuiltIn = builtIns.first(where: {
        $0.localizedName.localizedCaseInsensitiveContains("MacBook")
            || $0.localizedName.localizedCaseInsensitiveContains("FaceTime")
            || $0.localizedName.localizedCaseInsensitiveContains("Built-in")
    }) {
        return namedBuiltIn
    }
    if let builtIn = builtIns.first {
        return builtIn
    }

    let nonContinuity = devices.filter {
        !$0.localizedName.localizedCaseInsensitiveContains("iPhone")
            && !$0.localizedName.localizedCaseInsensitiveContains("Continuity")
    }
    if let namedNonContinuity = nonContinuity.first(where: {
        $0.localizedName.localizedCaseInsensitiveContains("MacBook")
            || $0.localizedName.localizedCaseInsensitiveContains("FaceTime")
            || $0.localizedName.localizedCaseInsensitiveContains("Built-in")
    }) {
        return namedNonContinuity
    }
    return nonContinuity.first ?? AVCaptureDevice.default(for: .video) ?? devices.first
}

func configureDevice(_ device: AVCaptureDevice) throws {
    try device.lockForConfiguration()
    defer { device.unlockForConfiguration() }

    let targetFps = max(1.0, min(maxFps, 60.0))
    let fpsScale = CMTimeScale(max(1, Int32(targetFps.rounded())))
    let frameDuration = CMTime(value: 1, timescale: fpsScale)

    let formatCandidates = device.formats.filter { format in
        let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let supportsFps = format.videoSupportedFrameRateRanges.contains { range in
            range.maxFrameRate >= targetFps && range.minFrameRate <= targetFps
        }
        return supportsFps &&
            Int(dimensions.width) >= Int(uploadWidth) &&
            Int(dimensions.height) >= Int(uploadHeight)
    }

    if let format = formatCandidates.min(by: { lhs, rhs in
        let left = CMVideoFormatDescriptionGetDimensions(lhs.formatDescription)
        let right = CMVideoFormatDescriptionGetDimensions(rhs.formatDescription)
        let leftPixels = Int(left.width) * Int(left.height)
        let rightPixels = Int(right.width) * Int(right.height)
        if leftPixels != rightPixels {
            return leftPixels < rightPixels
        }
        let leftMaxFps = lhs.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
        let rightMaxFps = rhs.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
        return leftMaxFps > rightMaxFps
    }) {
        device.activeFormat = format
        let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let ranges = format.videoSupportedFrameRateRanges
            .map { "\(String(format: "%.0f", $0.minFrameRate))-\(String(format: "%.0f", $0.maxFrameRate))" }
            .joined(separator: ",")
        log("Active format: \(dimensions.width)x\(dimensions.height), fps ranges: \(ranges)")
    } else {
        log("No explicit \(Int(targetFps)) FPS format matched \(Int(uploadWidth))x\(Int(uploadHeight)); using current active format.")
    }

    device.activeVideoMinFrameDuration = frameDuration
    device.activeVideoMaxFrameDuration = frameDuration
}

final class BridgeDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let context = CIContext()
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private var lastSent = Date.distantPast
    private var inFlightCount = 0
    private let maxInFlight = 6
    private var frameCount = 0
    private let stateQueue = DispatchQueue(label: "ExpressoCameraBridge.delegate.state")

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let now = Date()
        if now.timeIntervalSince(lastSent) < 0.90 / maxFps {
            return
        }
        let shouldPost = endpoint != nil
        if shouldPost && !reserveUploadSlot() {
            return
        }
        lastSent = now
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            if shouldPost {
                releaseUploadSlot()
            }
            return
        }
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let sx = uploadWidth / max(1.0, image.extent.width)
        let sy = uploadHeight / max(1.0, image.extent.height)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        guard let jpeg = context.jpegRepresentation(
            of: scaled,
            colorSpace: colorSpace,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: jpegQuality]
        ) else {
            if shouldPost {
                releaseUploadSlot()
            }
            return
        }
        frameCount += 1
        mjpegServer?.publish(jpeg)
        if let endpoint {
            postFrame(jpeg, frameCount: frameCount, endpoint: endpoint)
        } else if frameCount == 1 || frameCount % 300 == 0 {
            log("Served frame \(frameCount), bytes \(jpeg.count)")
        }
    }

    private func reserveUploadSlot() -> Bool {
        stateQueue.sync {
            if inFlightCount >= maxInFlight {
                return false
            }
            inFlightCount += 1
            return true
        }
    }

    private func releaseUploadSlot() {
        stateQueue.sync {
            inFlightCount = max(0, inFlightCount - 1)
        }
    }

    private func postFrame(_ jpeg: Data, frameCount: Int, endpoint: URL) {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
        request.setValue("ExpressoCameraBridge", forHTTPHeaderField: "User-Agent")
        URLSession.shared.uploadTask(with: request, from: jpeg) { _, response, error in
            defer { self.releaseUploadSlot() }
            if let error {
                log("Upload failed: \(error.localizedDescription)")
                return
            }
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            if frameCount == 1 || frameCount % 30 == 0 || status >= 300 {
                log("Uploaded frame \(frameCount), status \(status), bytes \(jpeg.count)")
            }
        }.resume()
    }
}

log("Expresso Camera Bridge starting")
log("Stream URL: http://127.0.0.1:\(streamPort)\(streamPath)")
if let endpoint {
    log("Optional POST endpoint: \(endpoint.absoluteString)")
} else {
    log("Optional POST endpoint disabled.")
}
if let controlEndpoint {
    log("Optional control endpoint: \(controlEndpoint.absoluteString)")
} else {
    log("Optional control endpoint disabled.")
}
log("Target FPS: \(maxFps)")
log("Upload size: \(Int(uploadWidth))x\(Int(uploadHeight)), JPEG quality: \(jpegQuality)")

if endpoint == nil && mjpegServer == nil {
    log("No camera output is configured.")
    exit(1)
}

mjpegServer?.start()

guard requestCameraAccess() else {
    log("Camera access was not granted. Enable this app in System Settings > Privacy & Security > Camera, then relaunch.")
    exit(2)
}

guard let device = chooseDevice() else {
    log("No video device found.")
    exit(1)
}

log("Selected device: \(device.localizedName)")

let session = AVCaptureSession()
let output = AVCaptureVideoDataOutput()
let delegate = BridgeDelegate()
let queue = DispatchQueue(label: "ExpressoCameraBridge.frames")
let sessionQueue = DispatchQueue(label: "ExpressoCameraBridge.session")
var captureRequested = true

session.beginConfiguration()
session.sessionPreset = .hd1280x720

do {
    try configureDevice(device)
    let input = try AVCaptureDeviceInput(device: device)
    guard session.canAddInput(input) else {
        log("Cannot add camera input.")
        exit(1)
    }
    session.addInput(input)
} catch {
    log("Cannot create camera input: \(error)")
    exit(1)
}

output.alwaysDiscardsLateVideoFrames = true
output.videoSettings = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
]
output.setSampleBufferDelegate(delegate, queue: queue)
guard session.canAddOutput(output) else {
    log("Cannot add video data output.")
    exit(1)
}
session.addOutput(output)
session.commitConfiguration()

func setCaptureRequested(_ enabled: Bool) {
    guard captureRequested != enabled else {
        return
    }
    captureRequested = enabled
    sessionQueue.async {
        if enabled {
            if !session.isRunning {
                session.startRunning()
                log("Capture resumed by server control.")
            }
        } else {
            if session.isRunning {
                session.stopRunning()
                log("Capture paused by server control.")
            }
        }
    }
}

func pollControl() {
    guard let controlEndpoint else {
        return
    }
    URLSession.shared.dataTask(with: controlEndpoint) { data, _, error in
        if let error {
            log("Control poll failed: \(error.localizedDescription)")
            return
        }
        guard
            let data,
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let enabled = json["enabled"] as? Bool
        else {
            log("Control poll returned invalid JSON.")
            return
        }
        DispatchQueue.main.async {
            setCaptureRequested(enabled)
        }
    }.resume()
}

sessionQueue.async {
    session.startRunning()
    log("Capture running. Keep this app open while calibrating.")
}
if controlEndpoint != nil {
    Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
        pollControl()
    }
    pollControl()
}
RunLoop.main.run()
