import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'dart:math' as math;
import 'dart:typed_data'; // Wajib untuk manipulasi byte

class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  CameraController? controller;
  late FlutterVision vision;

  List<CameraDescription> cameras = [];
  int selectedCameraIndex = 0;
  CameraImage? latestImage;

  bool isLoaded = false;
  bool isDetecting = false;
  bool isBusy = false;

  List<Map<String, dynamic>> yoloResults = [];

  // === SMOOTHING & STABILIZATION ===
  final Map<String, List<double>> _prevDisplayBoxes = {};
  final Map<String, List<double>> _confidenceHistory = {};
  final int _confidenceHistorySize = 5;
  final Map<String, int> _classFrameCount = {};

  Map<String, dynamic>? _prevBestDetection;
  int _framesSinceLastDetection = 0;

  // === CONFIGURABLE PARAMETERS ===
  // Threshold untuk Int8 biasanya perlu disesuaikan, kita mulai moderat
  double modelConfThreshold = 0.25;
  double modelIouThreshold = 0.45;
  double displayConfThreshold = 0.40;
  
  double boxSmoothingAlpha = 0.6;
  
  final int _minDetectionFrames = 2;
  final int _maxFramesWithoutDetection = 5;
  
  bool forceFrontCamera = true;
  bool debugMode = true;
  bool bypassFilters = false;

  @override
  void initState() {
    super.initState();
    init();
  }

  Future<void> init() async {
    try {
      cameras = await availableCameras();
      vision = FlutterVision();
      await loadYoloModel();
      await initializeCamera();
      setState(() => isLoaded = true);
    } catch (e) {
      debugPrint("Error init: $e");
    }
  }

  @override
  void dispose() {
    controller?.dispose();
    vision.closeYoloModel();
    super.dispose();
  }

Future<void> loadYoloModel() async {
    await vision.loadYoloModel(
      labels: "assets/labels.txt",
      modelPath: "assets/best_float16.tflite", // BALIK KE FLOAT16
      quantization: false, // WAJIB FALSE UNTUK FLOAT
      modelVersion: "yolov8",
      numThreads: 4,
      useGpu: false,
    );
    debugPrint("✓ Model loaded successfully (Mode: YOLOv8 Float16)");
  }

  Future<void> initializeCamera() async {
    if (controller != null) {
      await controller!.dispose();
    }

    if (forceFrontCamera) {
      final frontIndex = cameras
          .indexWhere((c) => c.lensDirection == CameraLensDirection.front);
      if (frontIndex != -1) selectedCameraIndex = frontIndex;
    }

    controller = CameraController(
      cameras[selectedCameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await controller!.initialize();
    debugPrint("📸 Camera initialized: ${controller!.value.previewSize}");
    setState(() {});
  }

  Future<void> switchCamera() async {
    if (cameras.length < 2) return;

    bool wasDetecting = isDetecting;
    if (wasDetecting) await stopDetection();

    selectedCameraIndex = (selectedCameraIndex + 1) % cameras.length;
    await initializeCamera();

    if (wasDetecting) {
      await Future.delayed(const Duration(milliseconds: 500));
      startDetection();
    }
  }

  Future<void> startDetection() async {
    if (controller == null || !controller!.value.isInitialized) return;
    if (controller!.value.isStreamingImages) return;

    isDetecting = true;
    setState(() {});

    await controller!.startImageStream((image) async {
      latestImage = image;
      if (isBusy || !isDetecting) return;

      isBusy = true;
      try {
        await runYolo(image);
      } catch (e) {
        debugPrint("Error detection: $e");
      } finally {
        isBusy = false;
      }
    });
  }

  Future<void> stopDetection() async {
    isDetecting = false;
    setState(() {});
    if (controller != null && controller!.value.isStreamingImages) {
      await controller!.stopImageStream();
    }
    yoloResults.clear();
    _prevDisplayBoxes.clear();
    _confidenceHistory.clear();
    _classFrameCount.clear();
  }

  // FUNGSI PEMBERSIH GAMBAR (WAJIB ADA)
  List<Uint8List> _processCameraImage(CameraImage image) {
    final planes = image.planes;
    final width = image.width;
    final height = image.height;

    Uint8List yBytes;
    final yPlane = planes[0];
    
    if (yPlane.bytesPerRow == width) {
      yBytes = yPlane.bytes;
    } else {
      int totalBytes = width * height;
      yBytes = Uint8List(totalBytes);
      
      for (int i = 0; i < height; i++) {
        int srcOffset = i * yPlane.bytesPerRow;
        int dstOffset = i * width;
        yBytes.setRange(
          dstOffset, 
          dstOffset + width, 
          yPlane.bytes.getRange(srcOffset, srcOffset + width)
        );
      }
    }

    return [
      yBytes,
      planes[1].bytes,
      planes[2].bytes,
    ];
  }

  double _toDouble(dynamic value) {
    if (value is double) return value;
    if (value is int) return value.toDouble();
    return 0.0;
  }

  double _calculateIoU(List<dynamic> box1, List<dynamic> box2) {
    final xA = math.max(box1[0], box2[0]);
    final yA = math.max(box1[1], box2[1]);
    final xB = math.min(box1[2], box2[2]);
    final yB = math.min(box1[3], box2[3]);

    final interArea = math.max(0.0, xB - xA) * math.max(0.0, yB - yA);
    final box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    final box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    return interArea / (box1Area + box2Area - interArea + 1e-6);
  }

  Future<void> runYolo(CameraImage image) async {
    // 1. Bersihkan gambar dari padding
    final cleanBytesList = _processCameraImage(image);

    // 2. Jalankan Model (Int8)
    final result = await vision.yoloOnFrame(
      bytesList: cleanBytesList,
      imageHeight: image.height,
      imageWidth: image.width,
      iouThreshold: modelIouThreshold,
      confThreshold: modelConfThreshold,
      classThreshold: modelConfThreshold,
    );

    // 3. Handle Hasil Kosong
    if (result.isEmpty) {
      _framesSinceLastDetection++;
      if (_framesSinceLastDetection < _maxFramesWithoutDetection &&
          _prevBestDetection != null) {
        if (mounted) setState(() => yoloResults = [_prevBestDetection!]);
        return;
      }
      if (mounted) setState(() => yoloResults = []);
      return;
    }

    _framesSinceLastDetection = 0;
    List<Map<String, dynamic>> validDetections = [];

    // 4. Filtering & Smoothing
    for (var r in result) {
      final box = r["box"];
      final tag = r["tag"];
      final rawConf = _toDouble(box[4]);

      if (bypassFilters) {
        validDetections.add(r);
        continue;
      }

      // Filter ukuran (buang yang terlalu kecil/besar aneh)
      double w = (box[2] - box[0]).abs();
      double h = (box[3] - box[1]).abs();
      double imgArea = (image.width * image.height).toDouble();
      double ratio = (w * h) / imgArea;

      if (ratio < 0.01 || ratio > 0.95) continue;

      // Smoothing Confidence
      _confidenceHistory[tag] = _confidenceHistory[tag] ?? [];
      _confidenceHistory[tag]!.add(rawConf);
      if (_confidenceHistory[tag]!.length > _confidenceHistorySize) {
        _confidenceHistory[tag]!.removeAt(0);
      }

      double smoothedConf = _confidenceHistory[tag]!.reduce((a, b) => a + b) /
          _confidenceHistory[tag]!.length;

      if (smoothedConf < displayConfThreshold) continue;

      // Stability check
      _classFrameCount[tag] = (_classFrameCount[tag] ?? 0) + 1;
      if (_classFrameCount[tag]! < _minDetectionFrames) continue;

      var finalBox = List<double>.from(box);
      finalBox[4] = smoothedConf;

      validDetections.add({
        "tag": tag,
        "box": finalBox,
        "rawConf": rawConf,
      });
    }

    _classFrameCount
        .removeWhere((key, value) => !validDetections.any((d) => d['tag'] == key));

    if (validDetections.isEmpty) {
      setState(() => yoloResults = []);
      return;
    }

    // 5. NMS Manual (Buang kotak duplikat)
    validDetections.sort((a, b) => b['box'][4].compareTo(a['box'][4]));
    List<Map<String, dynamic>> nmsResults = [];
    
    for (var det in validDetections) {
      bool overlap = false;
      for (var saved in nmsResults) {
        if (_calculateIoU(det['box'], saved['box']) > 0.5) {
          overlap = true;
          break;
        }
      }
      if (!overlap) nmsResults.add(det);
    }

    if (nmsResults.isNotEmpty) {
      _prevBestDetection = nmsResults.first;
    }

    if (mounted) {
      setState(() {
        yoloResults = bypassFilters ? validDetections : nmsResults;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!isLoaded || controller == null || !controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Deteksi BISINDO"),
        backgroundColor: Colors.teal,
        actions: [
          IconButton(
            icon: const Icon(Icons.cameraswitch),
            onPressed: switchCamera,
          ),
          IconButton(
            icon: Icon(debugMode ? Icons.bug_report : Icons.bug_report_outlined),
            onPressed: () => setState(() => debugMode = !debugMode),
          ),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final previewSize = controller!.value.previewSize!;
          final isPortrait =
              MediaQuery.of(context).orientation == Orientation.portrait;

          double sensorW, sensorH;
          if (isPortrait) {
            sensorW = previewSize.height;
            sensorH = previewSize.width;
          } else {
            sensorW = previewSize.width;
            sensorH = previewSize.height;
          }

          double scale = 1.0;
          // Scale logic: Cover (Full screen)
          scale = constraints.maxWidth / sensorW;
          if (sensorH * scale < constraints.maxHeight) {
            scale = constraints.maxHeight / sensorH;
          }

          return Stack(
            children: [
              // KAMERA LAYER
              Center(
                child: Transform.scale(
                  scale: scale,
                  child: AspectRatio(
                    aspectRatio: isPortrait
                        ? 1 / controller!.value.aspectRatio
                        : controller!.value.aspectRatio,
                    child: CameraPreview(controller!),
                  ),
                ),
              ),

              // BOX LAYER
              Positioned.fill(
                child: Stack(
                  children: _buildBoxes(constraints, sensorW, sensorH, scale),
                ),
              ),

              // DEBUG LAYER
              if (debugMode) _buildDebugPanel(),

              // BUTTON LAYER
              Positioned(
                bottom: 30, left: 0, right: 0,
                child: Center(
                  child: FloatingActionButton.extended(
                    onPressed: isDetecting ? stopDetection : startDetection,
                    icon: Icon(isDetecting ? Icons.stop : Icons.play_arrow),
                    label: Text(isDetecting ? "STOP" : "MULAI"),
                    backgroundColor: isDetecting ? Colors.red : Colors.teal,
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  List<Widget> _buildBoxes(
      BoxConstraints constraints, double sensorW, double sensorH, double scale) {
    if (yoloResults.isEmpty) return [];

    double previewW = sensorW * scale;
    double previewH = sensorH * scale;
    
    // Center the preview area
    double offsetX = (constraints.maxWidth - previewW) / 2;
    double offsetY = (constraints.maxHeight - previewH) / 2;

    return yoloResults.map((det) {
      final box = det['box'];
      
      // Transformasi koordinat
      double x1 = box[0] * scale + offsetX;
      double y1 = box[1] * scale + offsetY;
      double x2 = box[2] * scale + offsetX;
      double y2 = box[3] * scale + offsetY;

      // Mirroring untuk kamera depan
      if (cameras[selectedCameraIndex].lensDirection ==
          CameraLensDirection.front) {
        double screenCX = constraints.maxWidth / 2;
        x1 = screenCX + (screenCX - x1);
        x2 = screenCX + (screenCX - x2);
        // Swap x1 & x2
        double temp = x1; x1 = x2; x2 = temp;
      }

      // Smoothing posisi box
      final tag = det['tag'];
      final smoothKey = tag;
      if (_prevDisplayBoxes.containsKey(smoothKey)) {
        final prev = _prevDisplayBoxes[smoothKey]!;
        x1 = prev[0] + (x1 - prev[0]) * boxSmoothingAlpha;
        y1 = prev[1] + (y1 - prev[1]) * boxSmoothingAlpha;
        x2 = prev[2] + (x2 - prev[2]) * boxSmoothingAlpha;
        y2 = prev[3] + (y2 - prev[3]) * boxSmoothingAlpha;
      }
      _prevDisplayBoxes[smoothKey] = [x1, y1, x2, y2];

      double w = (x2 - x1).abs();
      double h = (y2 - y1).abs();
      double conf = box[4];

      Color color = conf > 0.6 ? Colors.green : (conf > 0.4 ? Colors.orange : Colors.red);

      return Positioned(
        left: x1, top: y1, width: w, height: h,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(color: color, width: 3),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Align(
            alignment: Alignment.topLeft,
            child: Container(
              color: color,
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              child: Text(
                "$tag ${(conf * 100).toStringAsFixed(0)}%",
                style: const TextStyle(
                    color: Colors.white, fontWeight: FontWeight.bold, fontSize: 12),
              ),
            ),
          ),
        ),
      );
    }).toList();
  }

  Widget _buildDebugPanel() {
    return Positioned(
      bottom: 100, left: 20, right: 20,
      child: Card(
        color: Colors.black87,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text("⚙️ DEBUG (Int8) | Deteksi: ${yoloResults.length}", 
                  style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              Row(
                children: [
                  const Text("Conf:", style: TextStyle(color: Colors.white)),
                  Expanded(
                    child: Slider(
                      value: displayConfThreshold,
                      min: 0.1, max: 0.9,
                      activeColor: Colors.green,
                      onChanged: (v) => setState(() => displayConfThreshold = v),
                    ),
                  ),
                  Text(displayConfThreshold.toStringAsFixed(2), 
                      style: const TextStyle(color: Colors.white)),
                ],
              ),
              Row(
                children: [
                  const Text("IOU:", style: TextStyle(color: Colors.white)),
                  Expanded(
                    child: Slider(
                      value: modelIouThreshold,
                      min: 0.1, max: 0.9,
                      activeColor: Colors.orange,
                      onChanged: (v) => setState(() => modelIouThreshold = v),
                    ),
                  ),
                  Text(modelIouThreshold.toStringAsFixed(2), 
                      style: const TextStyle(color: Colors.white)),
                ],
              ),
              CheckboxListTile(
                title: const Text("Bypass Filters", style: TextStyle(color: Colors.white)),
                value: bypassFilters,
                checkColor: Colors.black,
                activeColor: Colors.white,
                onChanged: (v) => setState(() => bypassFilters = v!),
              )
            ],
          ),
        ),
      ),
    );
  }
}