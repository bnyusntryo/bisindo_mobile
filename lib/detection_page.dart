import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'dart:math' as math;

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

  bool isLoaded = false;
  bool isDetecting = false;
  bool isBusy = false;

  List<Map<String, dynamic>> yoloResults = [];

  // === SMOOTHING & STABILIZATION ===
  final Map<String, List<double>> _prevDisplayBoxes = {};
  final Map<String, List<double>> _confidenceHistory = {};
  final int _confidenceHistorySize = 3;
  final Map<String, int> _classFrameCount = {};

  Map<String, dynamic>? _prevBestDetection;
  int _framesSinceLastDetection = 0;

  // === CONFIGURABLE PARAMETERS ===
  double modelConfThreshold = 0.20;
  double modelIouThreshold = 0.40;
  double modelClassThreshold = 0.20;
  double displayConfThreshold = 0.25;
  
  double boxSmoothingAlpha = 0.7;
  
  final int _minDetectionFrames = 1;
  final int _maxFramesWithoutDetection = 3;
  
  bool forceFrontCamera = true;
  bool debugMode = true;
  bool bypassFilters = false;

  // Tracking untuk debug
  int _totalFrames = 0;
  int _detectionCount = 0;

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
      debugPrint("❌ Error init: $e");
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
      modelPath: "assets/best_float16.tflite",
      quantization: false,
      modelVersion: "yolov8",
      numThreads: 4,
      useGpu: false,
    );
    debugPrint("✓ Model loaded successfully (YOLOv8 Float16)");
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
      ResolutionPreset.high,
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
    _totalFrames = 0;
    _detectionCount = 0;
    setState(() {});

    await controller!.startImageStream((image) async {
      if (isBusy || !isDetecting) return;

      isBusy = true;
      _totalFrames++;
      
      try {
        await runYolo(image);
      } catch (e) {
        debugPrint("❌ Error detection: $e");
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
    
    debugPrint("📊 Session Stats: $_detectionCount detections in $_totalFrames frames");
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
    final result = await vision.yoloOnFrame(
      bytesList: image.planes.map((p) => p.bytes).toList(),
      imageHeight: image.height,
      imageWidth: image.width,
      iouThreshold: modelIouThreshold,
      confThreshold: modelConfThreshold,
      classThreshold: modelClassThreshold,
    );

    // Handle empty result
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
    _detectionCount++;
    
    // Debug log setiap deteksi
    if (_detectionCount % 10 == 0) {
      debugPrint("🎯 Detection #$_detectionCount: ${result.length} objects");
    }

    List<Map<String, dynamic>> validDetections = [];

    for (var r in result) {
      final box = r["box"];
      final tag = r["tag"];
      final rawConf = _toDouble(box[4]);

      // BYPASS MODE: Tampilkan semua!
      if (bypassFilters) {
        validDetections.add(r);
        continue;
      }

      // Filter 1: Ukuran box (buang yang terlalu kecil/besar)
      double w = (box[2] - box[0]).abs();
      double h = (box[3] - box[1]).abs();
      double imgArea = (image.width * image.height).toDouble();
      double ratio = (w * h) / imgArea;

      if (ratio < 0.01 || ratio > 0.95) continue;

      // Filter 2: Smoothing Confidence (rata-rata 3 frame terakhir)
      _confidenceHistory[tag] = _confidenceHistory[tag] ?? [];
      _confidenceHistory[tag]!.add(rawConf);
      if (_confidenceHistory[tag]!.length > _confidenceHistorySize) {
        _confidenceHistory[tag]!.removeAt(0);
      }

      double smoothedConf = _confidenceHistory[tag]!.reduce((a, b) => a + b) /
          _confidenceHistory[tag]!.length;

      if (smoothedConf < displayConfThreshold) continue;

      // Filter 3: Stability check (harus muncul minimal N frame)
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

    // Clean up frame count untuk class yang tidak terdeteksi
    _classFrameCount.removeWhere(
        (key, value) => !validDetections.any((d) => d['tag'] == key));

    if (validDetections.isEmpty) {
      if (mounted) setState(() => yoloResults = []);
      return;
    }

    // NMS: Buang box yang overlap
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

    // Simpan deteksi terbaik
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
      return const Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 20),
              Text("Loading model..."),
            ],
          ),
        ),
      );
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

          // Full screen logic
          double scaleX = constraints.maxWidth / sensorW;
          double scaleY = constraints.maxHeight / sensorH;
          double finalScale = math.max(scaleX, scaleY);

          double scaledPreviewW = sensorW * finalScale;
          double scaledPreviewH = sensorH * finalScale;

          double offsetX = (constraints.maxWidth - scaledPreviewW) / 2;
          double offsetY = (constraints.maxHeight - scaledPreviewH) / 2;

          return Stack(
            fit: StackFit.expand,
            children: [
              // LAYER 1: KAMERA
              Center(
                child: OverflowBox(
                  maxWidth: scaledPreviewW,
                  maxHeight: scaledPreviewH,
                  minWidth: scaledPreviewW,
                  minHeight: scaledPreviewH,
                  child: SizedBox(
                    width: scaledPreviewW,
                    height: scaledPreviewH,
                    child: CameraPreview(controller!),
                  ),
                ),
              ),

              // LAYER 2: BOX DETEKSI
              Positioned.fill(
                child: Stack(
                  children: _buildBoxes(
                      constraints, sensorW, sensorH, finalScale, offsetX, offsetY),
                ),
              ),

              // LAYER 3: STATUS INDICATOR (DIUBAH)
              _buildStatusIndicator(),

              // LAYER 4: DEBUG PANEL
              if (debugMode) _buildDebugPanel(),

              // LAYER 5: TOMBOL START
              Positioned(
                bottom: 30,
                left: 0,
                right: 0,
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
      BoxConstraints constraints,
      double sensorW,
      double sensorH,
      double scale,
      double offsetX,
      double offsetY) {
    if (yoloResults.isEmpty) return [];

    final isFrontCamera = cameras[selectedCameraIndex].lensDirection == 
        CameraLensDirection.front;

    return yoloResults.map((det) {
      final box = det['box'];
      final tag = det['tag'];
      final conf = box[4];

      // 1. Hitung Koordinat Normal (Tanpa peduli arah kamera dulu)
      double x1 = box[0] * scale + offsetX;
      double y1 = box[1] * scale + offsetY;
      double x2 = box[2] * scale + offsetX;
      double y2 = box[3] * scale + offsetY;

      // 2. LOGIC FIX: FLIP Y (ATAS-BAWAH) SAJA!
      if (isFrontCamera) {
        double oldY1 = y1; double oldY2 = y2;
        // Balik Y relatif terhadap tinggi layar
        y1 = constraints.maxHeight - oldY2;
        y2 = constraints.maxHeight - oldY1;
      }

      // 3. Safety check urutan koordinat
      double left = math.min(x1, x2);
      double right = math.max(x1, x2);
      double top = math.min(y1, y2);
      double bottom = math.max(y1, y2);

      // Smoothing posisi box
      final smoothKey = tag;
      if (_prevDisplayBoxes.containsKey(smoothKey)) {
        final prev = _prevDisplayBoxes[smoothKey]!;
        left = prev[0] + (left - prev[0]) * boxSmoothingAlpha;
        top = prev[1] + (top - prev[1]) * boxSmoothingAlpha;
        right = prev[2] + (right - prev[2]) * boxSmoothingAlpha;
        bottom = prev[3] + (bottom - prev[3]) * boxSmoothingAlpha;
      }
      _prevDisplayBoxes[smoothKey] = [left, top, right, bottom];

      double w = (right - left).abs();
      double h = (bottom - top).abs();

      Color color = conf > 0.6
          ? Colors.green
          : (conf > 0.4 ? Colors.orange : Colors.red);

      return Positioned(
        left: left,
        top: top,
        width: w,
        height: h,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(color: color, width: 3),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Align(
            alignment: Alignment.topLeft,
            child: Container(
              color: color,
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
              child: Text(
                "$tag ${(conf * 100).toStringAsFixed(0)}%",
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
            ),
          ),
        ),
      );
    }).toList();
  }

  Widget _buildStatusIndicator() {
    // Hanya ambil satu hasil deteksi untuk ditampilkan di status bar
    final detectedTags = yoloResults.map((r) => r['tag']).join(', ');

    return Positioned(
      top: 40, // Pindah ke atas (di bawah AppBar)
      left: 10, // Pindah ke kiri
      // Hapus 'right' agar tidak memanjang full screen
      child: AnimatedOpacity(
        opacity: yoloResults.isNotEmpty ? 1.0 : 0.0,
        duration: const Duration(milliseconds: 300),
        child: Container(
          // Kecilkan padding
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6), 
          decoration: BoxDecoration(
            color: Colors.green.withOpacity(0.9),
            borderRadius: BorderRadius.circular(8), // Radius lebih kecil
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.3),
                blurRadius: 5, // Shadow lebih kecil
              ),
            ],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.check_circle, color: Colors.white, size: 16), // Icon lebih kecil
              const SizedBox(width: 6),
              Text(
                detectedTags.isEmpty
                    ? "Mencari..."
                    : "Terdeteksi: $detectedTags",
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 12, // Font lebih kecil
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDebugPanel() {
    return Positioned(
      bottom: 100,
      left: 20,
      right: 20,
      child: Card(
        color: Colors.black87,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "⚙️ DEBUG MODE",
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                "Deteksi: ${yoloResults.length} | Frame: $_totalFrames | Total: $_detectionCount",
                style: const TextStyle(color: Colors.white70, fontSize: 12),
              ),
              const Divider(color: Colors.white30),
              Row(
                children: [
                  const Text("Threshold:",
                      style: TextStyle(color: Colors.white, fontSize: 12)),
                  Expanded(
                    child: Slider(
                      value: displayConfThreshold,
                      min: 0.1,
                      max: 0.9,
                      divisions: 16,
                      activeColor: Colors.green,
                      label: "${(displayConfThreshold * 100).toStringAsFixed(0)}%",
                      onChanged: (v) =>
                          setState(() => displayConfThreshold = v),
                    ),
                  ),
                  Text(
                    "${(displayConfThreshold * 100).toStringAsFixed(0)}%",
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
              CheckboxListTile(
                title: const Text(
                  "Bypass Filters (Show All)",
                  style: TextStyle(color: Colors.white, fontSize: 12),
                ),
                subtitle: const Text(
                  "Tampilkan semua deteksi tanpa filter",
                  style: TextStyle(color: Colors.white60, fontSize: 10),
                ),
                value: bypassFilters,
                checkColor: Colors.black,
                activeColor: Colors.white,
                contentPadding: EdgeInsets.zero,
                dense: true,
                onChanged: (v) => setState(() => bypassFilters = v!),
              ),
            ],
          ),
        ),
      ),
    );
  }
}