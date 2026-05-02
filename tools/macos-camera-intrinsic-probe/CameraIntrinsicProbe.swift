import AVFoundation
import CoreMedia
import Darwin
import Foundation

let logPath = "/tmp/CameraIntrinsicProbe.log"
freopen(logPath, "w", stdout)
freopen(logPath, "a", stderr)
setbuf(stdout, nil)
print("CameraIntrinsicProbe log: \(logPath)")
print("Started: \(Date())")
print("")

func fourCC(_ value: FourCharCode) -> String {
    let chars: [UInt8] = [
        UInt8((value >> 24) & 0xff),
        UInt8((value >> 16) & 0xff),
        UInt8((value >> 8) & 0xff),
        UInt8(value & 0xff),
    ]
    return String(bytes: chars, encoding: .macOSRoman) ?? "\(value)"
}

func describeCFValue(_ value: Any) -> String {
    if let data = value as? Data {
        return "Data(\(data.count) bytes)"
    }
    if let dict = value as? NSDictionary {
        if dict.count <= 12 {
            let pairs = dict.allKeys
                .map { key in "\(key): \(describeCFValue(dict[key]!))" }
                .sorted()
                .joined(separator: ", ")
            return "Dictionary(\(pairs))"
        }
        return "Dictionary(keys: \(dict.allKeys))"
    }
    if let array = value as? NSArray {
        return "Array(count: \(array.count))"
    }
    return String(describing: value)
}

func decodeMatrix3x3(_ value: Any?) -> [[Float]]? {
    guard let data = value as? Data, data.count >= MemoryLayout<Float>.size * 9 else {
        return nil
    }
    return data.withUnsafeBytes { raw in
        let floats = raw.bindMemory(to: Float.self)
        if data.count >= MemoryLayout<Float>.size * 12 {
            let c0 = [floats[0], floats[1], floats[2]]
            let c1 = [floats[4], floats[5], floats[6]]
            let c2 = [floats[8], floats[9], floats[10]]
            return [
                [c0[0], c1[0], c2[0]],
                [c0[1], c1[1], c2[1]],
                [c0[2], c1[2], c2[2]],
            ]
        }
        return [
            [floats[0], floats[1], floats[2]],
            [floats[3], floats[4], floats[5]],
            [floats[6], floats[7], floats[8]],
        ]
    }
}

func printDeviceSummary(_ devices: [AVCaptureDevice]) {
    print("Video devices: \(devices.count)")
    for (index, device) in devices.enumerated() {
        print("")
        print("[\(index)] \(device.localizedName)")
        print("  uniqueID: \(device.uniqueID)")
        print("  modelID: \(device.modelID)")
        print("  manufacturer: \(device.manufacturer)")
        print("  position: \(device.position.rawValue)")
        print("  activeFormat: \(formatSummary(device.activeFormat))")
        print("  formats: \(device.formats.count)")
        for format in device.formats {
            let description = format.formatDescription
            let dimensions = CMVideoFormatDescriptionGetDimensions(description)
            let subtype = CMFormatDescriptionGetMediaSubType(description)
            let extensions = CMFormatDescriptionGetExtensions(description) as NSDictionary?
            var interesting: [String] = []
            if let extensions {
                for key in extensions.allKeys {
                    let name = String(describing: key)
                    if name.localizedCaseInsensitiveContains("camera")
                        || name.localizedCaseInsensitiveContains("intrinsic")
                        || name.localizedCaseInsensitiveContains("calibration") {
                        let value = extensions[key]!
                        interesting.append("\(name)=\(describeCFValue(value))")
                    }
                }
            }
            let suffix = interesting.isEmpty ? "" : " | \(interesting.joined(separator: "; "))"
            print("    \(dimensions.width)x\(dimensions.height) \(fourCC(subtype))\(suffix)")
        }
    }
}

func formatSummary(_ format: AVCaptureDevice.Format) -> String {
    let description = format.formatDescription
    let dimensions = CMVideoFormatDescriptionGetDimensions(description)
    let subtype = CMFormatDescriptionGetMediaSubType(description)
    return "\(dimensions.width)x\(dimensions.height) \(fourCC(subtype))"
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

final class SampleDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let maxSamples: Int
    private let done: DispatchSemaphore
    private var sampleCount = 0
    private var intrinsicCount = 0
    private var firstMatrix: [[Float]]?
    private var firstAttachmentKeys: [String] = []
    private var firstInterestingAttachments: [(String, String)] = []
    private var firstOriginalMatrix: [[Float]]?
    private var firstReferenceDimensions: String?
    private var firstSelectedMetadata: [(String, String)] = []
    private var firstPixelBufferSummary: String?
    private var firstFormatSummary: String?

    init(maxSamples: Int, done: DispatchSemaphore) {
        self.maxSamples = maxSamples
        self.done = done
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        sampleCount += 1

        if sampleCount == 1 {
            if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                firstPixelBufferSummary = "\(CVPixelBufferGetWidth(imageBuffer))x\(CVPixelBufferGetHeight(imageBuffer))"
            }
            if let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) {
                let dimensions = CMVideoFormatDescriptionGetDimensions(formatDescription)
                let aperture = CMVideoFormatDescriptionGetCleanAperture(formatDescription, originIsAtTopLeft: true)
                firstFormatSummary = "\(dimensions.width)x\(dimensions.height), cleanAperture=\(aperture)"
            }
            if let attachments = CMCopyDictionaryOfAttachments(
                allocator: kCFAllocatorDefault,
                target: sampleBuffer,
                attachmentMode: kCMAttachmentMode_ShouldPropagate
            ) as NSDictionary? {
                firstAttachmentKeys = attachments.allKeys.map { String(describing: $0) }.sorted()
                for key in attachments.allKeys {
                    let name = String(describing: key)
                    if name.localizedCaseInsensitiveContains("intrinsic")
                        || name.localizedCaseInsensitiveContains("matrix")
                        || name.localizedCaseInsensitiveContains("zoom")
                        || name.localizedCaseInsensitiveContains("metadata") {
                        let value = attachments[key]!
                        firstInterestingAttachments.append((name, describeCFValue(value)))
                        if name == "OriginalCameraIntrinsicMatrix" {
                            firstOriginalMatrix = decodeMatrix3x3(value)
                        }
                        if name == "OriginalCameraIntrinsicMatrixReferenceDimensions" {
                            firstReferenceDimensions = describeCFValue(value)
                        }
                        if name == "MetadataDictionary", let metadata = value as? NSDictionary {
                            let wanted = [
                                "PinholeCameraFocalLength",
                                "FocalLength",
                                "FocalLenIn35mmFilm",
                                "RawSensorWidth",
                                "RawSensorHeight",
                                "RawCropRect",
                                "SensorReadoutRect",
                                "SensorCropRect",
                                "TotalSensorCropRect",
                                "HighQualitySensorReadoutRect",
                                "SensorRawValidBufferRect",
                                "TotalScalingFromPhysicalSensor",
                                "IntermediateTapTotalScalingFromPhysicalSensor",
                                "SecondaryScalerTotalScalingFromPhysicalSensor",
                                "ZoomFactor",
                                "TotalZoomFactor",
                                "UIZoomFactor",
                                "OpticalCenter",
                                "StaticOpticalCenter",
                                "DistortionCenterInBuffer",
                                "DistortionOpticalCenter",
                                "DistortionOpticalCenterV2",
                                "CurrentFrameRate",
                            ]
                            for key in wanted {
                                if let item = metadata[key] {
                                    firstSelectedMetadata.append((key, describeCFValue(item)))
                                }
                            }
                        }
                    }
                }
            }
        }

        let attachment = CMGetAttachment(
            sampleBuffer,
            key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
            attachmentModeOut: nil
        )
        if attachment != nil {
            intrinsicCount += 1
            if firstMatrix == nil {
                firstMatrix = decodeMatrix3x3(attachment)
            }
        }

        if sampleCount >= maxSamples {
            done.signal()
        }
    }

    func printResult() {
        print("")
        print("Captured samples: \(sampleCount)")
        if let firstPixelBufferSummary {
            print("First sample pixel buffer: \(firstPixelBufferSummary)")
        }
        if let firstFormatSummary {
            print("First sample format description: \(firstFormatSummary)")
        }
        print("Samples with kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix: \(intrinsicCount)")
        if !firstAttachmentKeys.isEmpty {
            print("First sample attachment keys:")
            for key in firstAttachmentKeys {
                print("  \(key)")
            }
        }
        if !firstInterestingAttachments.isEmpty {
            print("First sample interesting attachment values:")
            for (key, value) in firstInterestingAttachments.sorted(by: { $0.0 < $1.0 }) {
                print("  \(key): \(value)")
            }
        }
        if let firstMatrix {
            print("Intrinsic matrix from first intrinsic-bearing sample:")
            for row in firstMatrix {
                print("  \(row)")
            }
        } else {
            print("Intrinsic matrix: not present in captured sample buffers")
        }
        if let firstOriginalMatrix {
            print("OriginalCameraIntrinsicMatrix:")
            for row in firstOriginalMatrix {
                print("  \(row)")
            }
        } else {
            print("OriginalCameraIntrinsicMatrix: not decoded")
        }
        if let firstReferenceDimensions {
            print("OriginalCameraIntrinsicMatrixReferenceDimensions: \(firstReferenceDimensions)")
        }
        if !firstSelectedMetadata.isEmpty {
            print("Selected MetadataDictionary fields:")
            for (key, value) in firstSelectedMetadata.sorted(by: { $0.0 < $1.0 }) {
                print("  \(key): \(value)")
            }
        }
    }
}

func chooseDevice(from devices: [AVCaptureDevice]) -> AVCaptureDevice? {
    if let named = devices.first(where: {
        $0.localizedName.localizedCaseInsensitiveContains("FaceTime")
            || $0.localizedName.localizedCaseInsensitiveContains("Built-in")
            || $0.localizedName.localizedCaseInsensitiveContains("MacBook")
    }) {
        return named
    }
    return AVCaptureDevice.default(for: .video) ?? devices.first
}

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera, .external],
    mediaType: .video,
    position: .unspecified
)
let devices = discovery.devices
printDeviceSummary(devices)

guard requestCameraAccess() else {
    print("")
    print("Camera access was not granted. Grant camera permission to this probe app/terminal and run again.")
    exit(2)
}

guard let device = chooseDevice(from: devices) else {
    print("No video capture device found.")
    exit(1)
}

print("")
print("Selected device: \(device.localizedName)")
print("Selected active format: \(formatSummary(device.activeFormat))")

let session = AVCaptureSession()
session.beginConfiguration()
session.sessionPreset = .high

do {
    let input = try AVCaptureDeviceInput(device: device)
    guard session.canAddInput(input) else {
        print("Cannot add camera input.")
        exit(1)
    }
    session.addInput(input)
} catch {
    print("Cannot create camera input: \(error)")
    exit(1)
}

let output = AVCaptureVideoDataOutput()
output.alwaysDiscardsLateVideoFrames = true
output.videoSettings = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
]
let queue = DispatchQueue(label: "CameraIntrinsicProbe.samples")
let done = DispatchSemaphore(value: 0)
let delegate = SampleDelegate(maxSamples: 60, done: done)
output.setSampleBufferDelegate(delegate, queue: queue)
guard session.canAddOutput(output) else {
    print("Cannot add video data output.")
    exit(1)
}
session.addOutput(output)
session.commitConfiguration()

session.startRunning()
_ = done.wait(timeout: .now() + 8)
session.stopRunning()
delegate.printResult()
