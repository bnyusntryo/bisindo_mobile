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

  // Strict Mode: TRUE = Huruf Saja, FALSE = Kata Saja
  bool _isAlphabetMode = true; 

  // Stabilization Variables
  final Map<String, List<double>> _prevDisplayBoxes = {};
  final Map<String, List<double>> _confidenceHistory = {};
  final int _confidenceHistorySize = 5; 
  final Map<String, int> _classFrameCount = {}; // Counter Frame
  
  Map<String, dynamic>? _prevBestDetection;
  int _framesSinceLastDetection = 0;

  // === CONFIG ===
  // Threshold model rendah biar semua kandidat masuk dulu
  double modelConfThreshold = 0.15; 
  double modelIouThreshold = 0.40;
  double modelClassThreshold = 0.15;
  
  // Threshold Tampilan (Slider)
  double displayConfThreshold = 0.40; 

  double boxSmoothingAlpha = 0.7;
  // BUG FIX: Turunin jadi 1 biar instan, atau biarin 2 tapi logic cleanup dibenerin
  final int _minDetectionFrames = 2; 
  final int _maxFramesWithoutDetection = 3;

  bool forceFrontCamera = true;
  bool debugMode = false; // Bisa dimatiin kalau udah fix
  bool bypassFilters = false;

  int _totalFrames = 0;
  int _detectionCount = 0;

  static const Color primaryTeal = Color(0xFF004D40);

  @override
  void initState() {
    super.initState();
    init();
  }

  @override
  void dispose() {
    controller?.dispose();
    vision.closeYoloModel();
    super.dispose();
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

  Future<void> loadYoloModel() async {
    await vision.loadYoloModel(
      labels: "assets/labels.txt",
      modelPath: "assets/best_float16.tflite",
      quantization: false,
      modelVersion: "yolov8",
      numThreads: 4,
      useGpu: false,
    );
  }

  Future<void> initializeCamera() async {
    if (controller != null) await controller!.dispose();

    if (forceFrontCamera) {
      final frontIndex = cameras.indexWhere((c) => c.lensDirection == CameraLensDirection.front);
      if (frontIndex != -1) selectedCameraIndex = frontIndex;
    }

    controller = CameraController(
      cameras[selectedCameraIndex],
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await controller!.initialize();
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
    _prevBestDetection = null;
    debugPrint("Stats: $_detectionCount detections");
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

    // --- HANDLE KOSONG & MEMORI ---
    if (result.isEmpty) {
      _framesSinceLastDetection++;
      if (_framesSinceLastDetection < _maxFramesWithoutDetection && _prevBestDetection != null) {
        // Cek validitas memori
        final memTag = _prevBestDetection!['tag'].toString().trim();
        final isMemWord = memTag.length > 1;
        bool isMemoryValid = (_isAlphabetMode && !isMemWord) || (!_isAlphabetMode && isMemWord);

        if (isMemoryValid) {
          if (mounted) setState(() => yoloResults = [_prevBestDetection!]);
          return;
        } else {
          _prevBestDetection = null;
        }
      }
      if (mounted) setState(() => yoloResults = []);
      return;
    }

    _framesSinceLastDetection = 0;
    _detectionCount++;

    // --- PREPARE DATA ---
    List<Map<String, dynamic>> validDetections = [];
    
    // [FIX] Catat semua tag yang MENTAH-MENTAH ada di frame ini
    // Ini dipake buat cleaning counter nanti, biar gak salah hapus
    Set<String> rawTagsInThisFrame = {};

    for (var r in result) {
      final rawTag = r["tag"].toString();
      // Jimat Trim: Bersihin spasi/enter
      final tag = rawTag.trim().replaceAll(RegExp(r'\s+'), ""); 
      rawTagsInThisFrame.add(tag);

      final box = r["box"];
      double rawConf = _toDouble(box[4]);

      if (bypassFilters) {
        validDetections.add(r);
        continue;
      }

      // === STRICT FILTER ===
      bool isWord = tag.length > 1; 
      if (_isAlphabetMode && isWord) continue; // Mode Huruf, usir Kata
      if (!_isAlphabetMode && !isWord) continue; // Mode Kata, usir Huruf

      // Filter Threshold
      if (rawConf < displayConfThreshold) continue;

      // Filter Ukuran
      double w = (box[2] - box[0]).abs();
      double h = (box[3] - box[1]).abs();
      double imgArea = (image.width * image.height).toDouble();
      double ratio = (w * h) / imgArea;
      if (ratio < 0.01 || ratio > 0.90) continue;

      // Smoothing Confidence
      _confidenceHistory[tag] = _confidenceHistory[tag] ?? [];
      _confidenceHistory[tag]!.add(rawConf);
      if (_confidenceHistory[tag]!.length > _confidenceHistorySize) {
        _confidenceHistory[tag]!.removeAt(0);
      }
      double smoothedConf = _confidenceHistory[tag]!.reduce((a, b) => a + b) / _confidenceHistory[tag]!.length;
      if (smoothedConf < displayConfThreshold) continue;

      // === STABILITY CHECK (FIXED) ===
      // Kita naikin counternya
      _classFrameCount[tag] = (_classFrameCount[tag] ?? 0) + 1;
      
      // Kalau belum cukup umur (frame count < min), JANGAN DIMASUKIN ke validDetections
      // TAPI... jangan dihapus juga dari memori counternya.
      if (_classFrameCount[tag]! < _minDetectionFrames) {
        continue; 
      }

      var finalBox = List<double>.from(box);
      finalBox[4] = rawConf;

      validDetections.add({
        "tag": tag,
        "box": finalBox,
        "rawConf": rawConf,
      });
    }

    // [FIX] LOGIC PEMBERSIH COUNTER YANG BENAR
    // Hapus counter HANYA JIKA tag tersebut BENAR-BENAR HILANG dari pandangan kamera (raw result)
    // Bukan karena difilter sama logic kita.
    _classFrameCount.removeWhere((key, value) => !rawTagsInThisFrame.contains(key));

    validDetections.sort((a, b) => b['box'][4].compareTo(a['box'][4]));
    
    // NMS
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

    if (nmsResults.isNotEmpty) _prevBestDetection = nmsResults.first;

    if (mounted) {
      setState(() => yoloResults = bypassFilters ? validDetections : nmsResults);
    }
  }

  // --- UI ---

  void _showBisindoLibrary() {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: const EdgeInsets.all(20),
        child: Container(
          decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(20)),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: primaryTeal,
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.menu_book_rounded, color: Colors.white, size: 28),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text("ABJAD BISINDO", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.white)),
                    ),
                    IconButton(icon: const Icon(Icons.close_rounded, color: Colors.white), onPressed: () => Navigator.pop(context)),
                  ],
                ),
              ),
              Flexible(
                child: InteractiveViewer(
                  minScale: 0.5, maxScale: 4.0,
                  child: Container(
                    padding: const EdgeInsets.all(20),
                    child: Image.asset('assets/abjad.jfif', fit: BoxFit.contain),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLoadingScreen(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(color: primaryTeal, strokeWidth: 4),
            const SizedBox(height: 24),
            Text("Memuat Model...", style: TextStyle(color: primaryTeal, fontWeight: FontWeight.w600, fontSize: 18)),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildBoxes(BoxConstraints constraints, double sensorW, double sensorH, double scale, double offsetX, double offsetY) {
    if (yoloResults.isEmpty) return [];
    final isFrontCamera = cameras[selectedCameraIndex].lensDirection == CameraLensDirection.front;
    const double borderRadius = 6.0;

    return yoloResults.map((det) {
      final box = det['box'];
      final tag = det['tag'];
      final conf = box[4];

      double x1 = box[0] * scale + offsetX;
      double y1 = box[1] * scale + offsetY;
      double x2 = box[2] * scale + offsetX;
      double y2 = box[3] * scale + offsetY;

      if (isFrontCamera) {
        double oldY1 = y1; double oldY2 = y2;
        y1 = constraints.maxHeight - oldY2;
        y2 = constraints.maxHeight - oldY1;
      }

      double left = math.min(x1, x2);
      double right = math.max(x1, x2);
      double top = math.min(y1, y2);
      double bottom = math.max(y1, y2);

      final smoothKey = tag;
      if (_prevDisplayBoxes.containsKey(smoothKey)) {
        final prev = _prevDisplayBoxes[smoothKey]!;
        left = prev[0] + (left - prev[0]) * boxSmoothingAlpha;
        top = prev[1] + (top - prev[1]) * boxSmoothingAlpha;
        right = prev[2] + (right - prev[2]) * boxSmoothingAlpha;
        bottom = prev[3] + (bottom - prev[3]) * boxSmoothingAlpha;
      }
      _prevDisplayBoxes[smoothKey] = [left, top, right, bottom];

      Color borderColor = conf >= 0.60 ? Colors.greenAccent.shade400 : Colors.yellow;

      return Positioned(
        left: left, top: top, width: (right - left).abs(), height: (bottom - top).abs(),
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(color: borderColor, width: 2.5),
            borderRadius: BorderRadius.circular(borderRadius),
          ),
          child: Align(
            alignment: Alignment.topLeft,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: primaryTeal,
                borderRadius: const BorderRadius.only(topLeft: Radius.circular(borderRadius - 1), bottomRight: Radius.circular(3)),
              ),
              child: Text(
                "$tag ${(conf * 100).toStringAsFixed(0)}%",
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 13),
              ),
            ),
          ),
        ),
      );
    }).toList();
  }

  Widget _buildStatusIndicator() {
    final detectedTags = yoloResults.map((r) => r['tag']).join(', ');
    final displayText = detectedTags.isEmpty ? (isDetecting ? "Mencari..." : "Siap") : detectedTags.toUpperCase();
    final color = detectedTags.isEmpty ? Colors.grey.shade700 : Colors.green.shade600;

    String modeLabel = _isAlphabetMode ? "MODE: ABJAD" : "MODE: KATA";
    Color modeColor = _isAlphabetMode ? Colors.orange.shade800 : primaryTeal;

    return Positioned(
      bottom: 100, left: 20, right: 20,
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            margin: const EdgeInsets.only(bottom: 8),
            decoration: BoxDecoration(color: modeColor.withAlpha(200), borderRadius: BorderRadius.circular(15)),
            child: Text(modeLabel, style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold)),
          ),
          Center(
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeOut,
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              decoration: BoxDecoration(color: color.withAlpha(230), borderRadius: BorderRadius.circular(30)),
              child: Text(displayText, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDebugPanel() {
    return Positioned(
      bottom: 200, left: 20, right: 20,
      child: Card(
        color: Colors.black54,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text("Debug: Frames $_totalFrames | Det: $_detectionCount", style: const TextStyle(color: Colors.white)),
              Row(
                children: [
                  const Text("Th:", style: TextStyle(color: Colors.white)),
                  Expanded(
                    child: Slider(
                      value: displayConfThreshold, min: 0.01, max: 0.9,
                      activeColor: primaryTeal,
                      onChanged: (v) => setState(() => displayConfThreshold = v),
                    ),
                  ),
                  Text(displayConfThreshold.toStringAsFixed(2), style: const TextStyle(color: Colors.white)),
                ],
              ),
              CheckboxListTile(
                title: const Text("Bypass Filters", style: TextStyle(color: Colors.white)),
                value: bypassFilters,
                checkColor: primaryTeal,
                activeColor: Colors.white,
                contentPadding: EdgeInsets.zero,
                onChanged: (v) => setState(() => bypassFilters = v!),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!isLoaded || controller == null || !controller!.value.isInitialized) {
      return _buildLoadingScreen(context);
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text("BISINDO Detector"),
        actions: [
          IconButton(
            icon: Icon(
              _isAlphabetMode ? Icons.abc : Icons.chat_bubble_outline,
              color: _isAlphabetMode ? Colors.amber : primaryTeal, 
              size: 30,
            ),
            tooltip: "Ganti Mode",
            onPressed: () {
              setState(() {
                _isAlphabetMode = !_isAlphabetMode;
                yoloResults.clear();
                _prevDisplayBoxes.clear();
                _confidenceHistory.clear();
                _classFrameCount.clear();
                _prevBestDetection = null;
              });
              ScaffoldMessenger.of(context).hideCurrentSnackBar();
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(_isAlphabetMode ? "Mode ABJAD (Hanya Huruf)" : "Mode KATA (Hanya Kata)"),
                  duration: const Duration(seconds: 1),
                  backgroundColor: primaryTeal,
                ),
              );
            },
          ),
          IconButton(icon: const Icon(Icons.menu_book_rounded), onPressed: _showBisindoLibrary, color: primaryTeal),
          IconButton(icon: const Icon(Icons.cameraswitch_outlined), onPressed: switchCamera, color: primaryTeal),
          IconButton(icon: Icon(debugMode ? Icons.bug_report : Icons.bug_report_outlined), onPressed: () => setState(() => debugMode = !debugMode), color: primaryTeal),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final size = controller!.value.previewSize!;
          final isPortrait = MediaQuery.of(context).orientation == Orientation.portrait;
          double sensorW = isPortrait ? size.height : size.width;
          double sensorH = isPortrait ? size.width : size.height;
          double scaleX = constraints.maxWidth / sensorW;
          double scaleY = constraints.maxHeight / sensorH;
          double finalScale = math.max(scaleX, scaleY);
          double w = sensorW * finalScale;
          double h = sensorH * finalScale;
          double ox = (constraints.maxWidth - w) / 2;
          double oy = (constraints.maxHeight - h) / 2;

          return Stack(
            children: [
              Center(child: OverflowBox(minWidth: w, maxWidth: w, minHeight: h, maxHeight: h, child: CameraPreview(controller!))),
              Positioned.fill(child: Stack(children: _buildBoxes(constraints, sensorW, sensorH, finalScale, ox, oy))),
              if (debugMode) _buildDebugPanel(),
              _buildStatusIndicator(),
              Positioned(
                bottom: 30, left: 0, right: 0,
                child: Center(
                  child: FloatingActionButton(
                    heroTag: "fab_action",
                    onPressed: isDetecting ? stopDetection : startDetection,
                    backgroundColor: isDetecting ? Colors.red.shade700 : primaryTeal,
                    foregroundColor: Colors.white,
                    child: Icon(isDetecting ? Icons.stop : Icons.play_arrow_rounded, size: 32),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}