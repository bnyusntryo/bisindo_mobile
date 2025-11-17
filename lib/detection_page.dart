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

  CameraImage? latestImage;

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
  double modelConfThreshold = 0.15;
  double modelIouThreshold = 0.4;
  double modelClassThreshold = 0.15;

  double displayConfThreshold = 0.40;
  double minBoxAreaRatio = 0.001;
  double maxBoxAreaRatio = 0.95;

  double _confGapThreshold = 0.15;
  double boxSmoothingAlpha = 0.6;

  final int _minDetectionFrames = 1;
  final int _maxFramesWithoutDetection = 5;

  int _minSameClassFrames = 2;

  String? _prevClassName;
  int _sameClassCount = 0;

  final Map<String, Map<String, int>> _confusionMatrix = {};

  bool forceFrontCamera = true;
  bool autoMirror = true;
  bool debugMode = true;
  bool bypassFilters = true;

  @override
  void initState() {
    super.initState();
    init();
  }

  Future<void> init() async {
    cameras = await availableCameras();
    vision = FlutterVision();

    await loadYoloModel();
    await initializeCamera();

    setState(() => isLoaded = true);
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
    debugPrint("✓ Model loaded successfully");
  }

  Future<void> initializeCamera() async {
    if (controller != null) await controller!.dispose();

    if (forceFrontCamera) {
      final frontCameraIndex = cameras.indexWhere(
            (camera) => camera.lensDirection == CameraLensDirection.front,
      );
      if (frontCameraIndex != -1) {
        selectedCameraIndex = frontCameraIndex;
      } else {
        debugPrint("⚠️ Front camera tidak ditemukan");
      }
    }

    controller = CameraController(
      cameras[selectedCameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await controller!.initialize();

    debugPrint("📸 Camera initialized");
    debugPrint("   Lens: ${cameras[selectedCameraIndex].lensDirection}");
    debugPrint("   Preview size: ${controller!.value.previewSize}");

    setState(() {});
  }

  Future<void> switchCamera() async {
    if (cameras.length < 2) return;

    bool wasDetecting = isDetecting;
    if (wasDetecting) await stopDetection();

    selectedCameraIndex = (selectedCameraIndex + 1) % cameras.length;
    await initializeCamera();

    _prevDisplayBoxes.clear();
    _confidenceHistory.clear();
    _classFrameCount.clear();
    _prevBestDetection = null;

    if (wasDetecting) {
      await Future.delayed(const Duration(milliseconds: 300));
      startDetection();
    }
  }

  Future<void> startDetection() async {
    if (controller == null || !controller!.value.isInitialized) return;

    debugPrint("\n🚀 === STARTING DETECTION ===");
    debugPrint("   Camera initialized: ${controller!.value.isInitialized}");
    debugPrint("   Already streaming: ${controller!.value.isStreamingImages}");

    isDetecting = true;
    setState(() {});

    if (controller!.value.isStreamingImages) {
      debugPrint("⚠️ Already streaming, skipping startImageStream");
      return;
    }

    debugPrint("📸 Starting image stream...");
    int frameCount = 0;

    await controller!.startImageStream((image) async {
      frameCount++;
      if (frameCount % 30 == 0) {
        debugPrint("📊 Processed $frameCount frames");
      }

      latestImage = image;

      if (isBusy || !isDetecting) {
        if (frameCount % 30 == 0 && isBusy) {
          debugPrint("⏳ Busy processing previous frame...");
        }
        return;
      }

      isBusy = true;

      try {
        await runYolo(image);
      } catch (e, stackTrace) {
        debugPrint("❌ Detection error in frame $frameCount: $e");
        debugPrint("📍 Stack trace: $stackTrace");
      } finally {
        isBusy = false;
      }
    });

    debugPrint("✅ Image stream started");
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

  double _toDouble(dynamic value) {
    if (value is double) return value;
    if (value is int) return value.toDouble();
    if (value is num) return value.toDouble();
    return 0.0;
  }

  double _calculateIoU(List<dynamic> box1, List<dynamic> box2) {
    final x1_1 = _toDouble(box1[0]);
    final y1_1 = _toDouble(box1[1]);
    final x2_1 = _toDouble(box1[2]);
    final y2_1 = _toDouble(box1[3]);

    final x1_2 = _toDouble(box2[0]);
    final y1_2 = _toDouble(box2[1]);
    final x2_2 = _toDouble(box2[2]);
    final y2_2 = _toDouble(box2[3]);

    final xLeft = math.max(x1_1, x1_2);
    final yTop = math.max(y1_1, y1_2);
    final xRight = math.min(x2_1, x2_2);
    final yBottom = math.min(y2_1, y2_2);

    if (xRight < xLeft || yBottom < yTop) {
      return 0.0; 
    }

    final intersectionArea = (xRight - xLeft) * (yBottom - yTop);
    final box1Area = (x2_1 - x1_1) * (y2_1 - y1_1);
    final box2Area = (x2_2 - x1_2) * (y2_2 - y1_2);
    final unionArea = box1Area + box2Area - intersectionArea;

    return intersectionArea / unionArea;
  }

  Future<void> runYolo(CameraImage image) async {
    late List<Map<String, dynamic>> result;

    try {
      result = await vision.yoloOnFrame(
        bytesList: image.planes.map((p) => p.bytes).toList(),
        imageHeight: image.height,
        imageWidth: image.width,
        iouThreshold: modelIouThreshold,
        confThreshold: modelConfThreshold,
        classThreshold: modelClassThreshold,
      );
    } catch (e, stackTrace) {
      debugPrint("❌ YOLO INFERENCE ERROR: $e");
      debugPrint("📍 Stack trace: $stackTrace");
      rethrow;
    }

    if (result.isEmpty) {
      _framesSinceLastDetection++;

      if (_framesSinceLastDetection < _maxFramesWithoutDetection &&
          _prevBestDetection != null) {
        var prevBox = List<double>.from(_prevBestDetection!['box']);
        prevBox[4] = prevBox[4] * 0.9;

        if (prevBox[4] > displayConfThreshold * 0.7) {
          setState(() {
            yoloResults = [_prevBestDetection!];
          });
          return;
        }
      }

      if (mounted) {
        setState(() {
          yoloResults = [];
        });
      }
      return;
    }

    _framesSinceLastDetection = 0;

    final imgW = image.width.toDouble();
    final imgH = image.height.toDouble();
    final imgArea = imgW * imgH;

    List<Map<String, dynamic>> validDetections = [];

    for (var r in result) {
      try {
        final box = r["box"];
        if (box == null || box.length < 5) {
          continue;
        }

        final tag = r["tag"] ?? "unknown";
        final rawConf = _toDouble(box[4]);

        final centerX = _toDouble(box[0]);
        final centerY = _toDouble(box[1]);
        final width = _toDouble(box[2]);
        final height = _toDouble(box[3]);

        final x1 = (centerX - width / 2).clamp(0, imgW);
        final y1 = (centerY - height / 2).clamp(0, imgH);
        final x2 = (centerX + width / 2).clamp(0, imgW);
        final y2 = (centerY + height / 2).clamp(0, imgH);

        if (bypassFilters) {
          validDetections.add({
            "tag": tag,
            "box": [x1, y1, x2, y2, rawConf],
            "rawConf": rawConf,
          });
          continue;
        }

        final boxW = (x2 - x1).abs();
        final boxH = (y2 - y1).abs();

        if (boxW < 10 || boxH < 10) continue;

        final boxArea = boxW * boxH;
        final areaRatio = boxArea / imgArea;

        if (areaRatio < minBoxAreaRatio || areaRatio > maxBoxAreaRatio) continue;

        _confidenceHistory[tag] = _confidenceHistory[tag] ?? [];
        _confidenceHistory[tag]!.add(rawConf);

        if (_confidenceHistory[tag]!.length > _confidenceHistorySize) {
          _confidenceHistory[tag]!.removeAt(0);
        }

        final smoothedConf = _confidenceHistory[tag]!.reduce((a, b) => a + b) /
            _confidenceHistory[tag]!.length;

        if (smoothedConf < displayConfThreshold) continue;

        _classFrameCount[tag] = (_classFrameCount[tag] ?? 0) + 1;

        if (_classFrameCount[tag]! < _minDetectionFrames) continue;

        validDetections.add({
          "tag": tag,
          "box": [x1, y1, x2, y2, smoothedConf],
          "rawConf": rawConf,
        });

      } catch (e, stackTrace) {
        debugPrint("   ❌ ERROR processing detection: $e");
        debugPrint("   📍 Stack: $stackTrace");
        continue;
      }
    }

    _classFrameCount.removeWhere((key, value) {
      bool found = validDetections.any((d) => d['tag'] == key);
      return !found;
    });

    if (validDetections.isEmpty) {
      if (_prevBestDetection != null &&
          _framesSinceLastDetection < _maxFramesWithoutDetection) {
        setState(() {
          yoloResults = [_prevBestDetection!];
        });
      } else {
        setState(() {
          yoloResults = [];
        });
      }
      return;
    }

    validDetections.sort((a, b) {
      final confA = a["box"][4].toDouble();
      final confB = b["box"][4].toDouble();
      return confB.compareTo(confA);
    });

    List<Map<String, dynamic>> nmsFiltered = [];
    for (var det in validDetections) {
      bool keep = true;
      for (var kept in nmsFiltered) {
        if (_calculateIoU(det['box'], kept['box']) > 0.5) {
          keep = false;
          break;
        }
      }
      if (keep) {
        nmsFiltered.add(det);
      }
    }

    if (nmsFiltered.isEmpty) {
      setState(() {
        yoloResults = [];
      });
      return;
    }

    final bestDetection = nmsFiltered.first;
    final bestClassName = bestDetection["tag"];
    final bestConf = bestDetection["box"][4];

    bool instantAccept = false;

    if (_prevBestDetection != null) {
      double prevConf = _toDouble(_prevBestDetection!['rawConf']);
      if (bestConf - prevConf > _confGapThreshold) {
        instantAccept = true;
      }
    }

    if (_prevClassName == bestClassName) {
      _sameClassCount++;
    } else {
      if (instantAccept) {
        _sameClassCount = 1;
      } else {
        _sameClassCount = 1;
      }
    }

    String displayClassName;
    if (_sameClassCount >= _minSameClassFrames || _prevClassName == null || instantAccept) {
      displayClassName = bestClassName;
      _prevClassName = bestClassName;

      if (_prevClassName != null && _prevClassName != bestClassName) {
        _confusionMatrix[_prevClassName!] = _confusionMatrix[_prevClassName!] ?? {};
        _confusionMatrix[_prevClassName!]![bestClassName] =
            (_confusionMatrix[_prevClassName!]![bestClassName] ?? 0) + 1;
      }
    } else {
      displayClassName = _prevClassName!;
    }

    final stableDetection = {
      "tag": displayClassName,
      "box": bestDetection["box"],
      "rawConf": bestDetection["rawConf"],
      "candidateClass": (displayClassName != bestClassName) ? bestClassName : null,
      "stability": _sameClassCount,
      "instantAccept": instantAccept,
    };

    _prevBestDetection = stableDetection;

    if (!mounted) return;

    setState(() {
      yoloResults = [stableDetection];
    });
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
              Text("Loading BISINDO Detector..."),
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
            tooltip: "Switch Camera",
          ),
          IconButton(
            icon: Icon(debugMode ? Icons.bug_report : Icons.bug_report_outlined),
            onPressed: () {
              setState(() {
                debugMode = !debugMode;
              });
            },
            tooltip: "Toggle Debug",
          ),
          IconButton(
            icon: Icon(bypassFilters ? Icons.filter_alt_off : Icons.filter_alt),
            onPressed: () {
              setState(() {
                bypassFilters = !bypassFilters;
              });
              debugPrint("🔧 Bypass filters: $bypassFilters");
            },
            tooltip: bypassFilters ? "Filters OFF" : "Filters ON",
          ),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final screenW = constraints.maxWidth;
          final screenH = constraints.maxHeight;

          final previewSize = controller!.value.previewSize!;
          final previewW = previewSize.height.toDouble();
          final previewH = previewSize.width.toDouble();

          double ratioW = screenW / previewW;
          double ratioH = screenH / previewH;
          double scale = math.min(ratioW, ratioH);

          double renderW = previewW * scale;

          double offsetX = (screenW - renderW) / 2;
          double offsetY = (screenH - (previewH * scale)) / 2;

          return Stack(
            children: [
              Center(
                child: FittedBox(
                  fit: BoxFit.contain,
                  child: SizedBox(
                    width: previewW,
                    height: previewH,
                    child: Transform(
                      alignment: Alignment.center,
                      transform: Matrix4.rotationY(
                        (autoMirror &&
                            cameras[selectedCameraIndex].lensDirection ==
                                CameraLensDirection.front)
                            ? math.pi
                            : 0,
                      ),
                      child: CameraPreview(controller!),
                    ),
                  ),
                ),
              ),
              ..._buildBoxes(
                constraints,
                previewW,
                previewH,
                scale,
                offsetX,
                offsetY,
              ),
              if (debugMode) _buildDebugInfo(),
              if (debugMode) _buildDebugControls(),
              Positioned(
                bottom: 30,
                left: 0,
                right: 0,
                child: Center(
                  child: ElevatedButton.icon(
                    onPressed: isDetecting ? stopDetection : startDetection,
                    icon: Icon(isDetecting ? Icons.stop : Icons.play_arrow),
                    label: Text(
                      isDetecting ? "Stop Deteksi" : "Mulai Deteksi",
                      style: const TextStyle(fontSize: 16),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: isDetecting ? Colors.red : Colors.teal,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 32,
                        vertical: 16,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildDebugInfo() {
    return Positioned(
      top: 10,
      left: 10,
      child: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.black87,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "🐛 Debug Mode",
              style: TextStyle(
                color: Colors.yellow,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              "Streaming: ${controller?.value.isStreamingImages ?? false}",
              style: TextStyle(
                color: (controller?.value.isStreamingImages ?? false) ? Colors.greenAccent : Colors.redAccent,
                fontSize: 12,
              ),
            ),
            Text(
              "Busy: $isBusy",
              style: TextStyle(
                color: isBusy ? Colors.orangeAccent : Colors.white,
                fontSize: 12,
              ),
            ),
            Text(
              "Detections: ${yoloResults.length}",
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
            Text(
              "Bypass: ${bypassFilters ? 'ON' : 'OFF'}",
              style: TextStyle(
                color: bypassFilters ? Colors.greenAccent : Colors.redAccent,
                fontSize: 12,
              ),
            ),
            Text(
              "Conf: ${displayConfThreshold.toStringAsFixed(2)}",
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
            Text(
              "Image: ${latestImage != null ? '${latestImage!.width}x${latestImage!.height}' : 'none'}",
              style: const TextStyle(color: Colors.white, fontSize: 10),
            ),
            if (yoloResults.isNotEmpty)
              Text(
                "Active: ${yoloResults.first['tag']} (${(yoloResults.first['box'][4] * 100).toStringAsFixed(0)}%)",
                style: TextStyle(
                  color: yoloResults.first['instantAccept'] == true ? Colors.greenAccent : Colors.yellowAccent,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
            if (yoloResults.isNotEmpty &&
                yoloResults.first['candidateClass'] != null &&
                yoloResults.first['instantAccept'] != true) ...[
              Text(
                "→ ${yoloResults.first['candidateClass']}? ($_sameClassCount/$_minSameClassFrames)",
                style: const TextStyle(color: Colors.orangeAccent, fontSize: 12),
              ),
              if (_prevBestDetection != null)
                Text(
                  "Gap: ${((yoloResults.first['rawConf'] - _toDouble(_prevBestDetection!['rawConf'])) * 100).toStringAsFixed(0)}%",
                  style: const TextStyle(color: Colors.yellowAccent, fontSize: 10),
                ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildDebugControls() {
    return Positioned(
      bottom: 120,
      left: 20,
      right: 20,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.black87,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.yellow, width: 2),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              "🔧 Debug Controls",
              style: TextStyle(
                color: Colors.yellow,
                fontWeight: FontWeight.bold,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              "Confidence: ${displayConfThreshold.toStringAsFixed(2)}",
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
            Slider(
              value: displayConfThreshold,
              min: 0.1,
              max: 0.9,
              divisions: 80,
              activeColor: Colors.greenAccent,
              inactiveColor: Colors.grey,
              onChanged: (v) => setState(() => displayConfThreshold = v),
              label: displayConfThreshold.toStringAsFixed(2),
            ),
            const SizedBox(height: 8),
            Text(
              "Stability: $_minSameClassFrames",
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
            Slider(
              value: _minSameClassFrames.toDouble(),
              min: 1,
              max: 5,
              divisions: 4,
              activeColor: Colors.purpleAccent,
              inactiveColor: Colors.grey,
              onChanged: (v) => setState(() => _minSameClassFrames = v.round()),
              label: _minSameClassFrames.toString(),
            ),
            const SizedBox(height: 8),
            Text(
              "Model Conf: ${modelConfThreshold.toStringAsFixed(2)}",
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
            Slider(
              value: modelConfThreshold,
              min: 0.05,
              max: 0.5,
              divisions: 45,
              activeColor: Colors.cyanAccent,
              inactiveColor: Colors.grey,
              onChanged: (v) => setState(() => modelConfThreshold = v),
              label: modelConfThreshold.toStringAsFixed(2),
            ),
            const SizedBox(height: 8),
            Text(
              "Conf Gap: ${(_confGapThreshold * 100).toStringAsFixed(0)}%",
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
            Slider(
              value: _confGapThreshold,
              min: 0.05,
              max: 0.30,
              divisions: 25,
              activeColor: Colors.orangeAccent,
              inactiveColor: Colors.grey,
              onChanged: (v) => setState(() => _confGapThreshold = v),
              label: "${(_confGapThreshold * 100).toStringAsFixed(0)}%",
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: () async {
                if (latestImage == null) {
                  debugPrint("⚠️ No image for testing");
                  return;
                }
                debugPrint("\n🧪 === MANUAL TEST ===");
                try {
                  await runYolo(latestImage!);
                  debugPrint("✅ Test completed");
                } catch (e, stackTrace) {
                  debugPrint("❌ Test failed: $e");
                  debugPrint("📍 Stack: $stackTrace");
                }
              },
              icon: const Icon(Icons.science, size: 16),
              label: const Text("Test Now"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              ),
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildBoxes(
      BoxConstraints constraints,
      double previewW,
      double previewH,
      double scale,
      double offsetX,
      double offsetY,
      ) {
    if (yoloResults.isEmpty || latestImage == null) return [];

    bool isFront = cameras[selectedCameraIndex].lensDirection ==
        CameraLensDirection.front;

    List<Widget> widgets = [];

    for (var det in yoloResults) {
      final box = det["box"];
      final tag = det["tag"] ?? "?";
      final candidateClass = det["candidateClass"];
      final stability = det["stability"] ?? 0;
      final conf = _toDouble(box[4]);

      double x1 = _toDouble(box[0]);
      double y1 = _toDouble(box[1]);
      double x2 = _toDouble(box[2]);
      double y2 = _toDouble(box[3]);

      double left = x1 * scale + offsetX;
      double top = y1 * scale + offsetY;
      double w = (x2 - x1) * scale;
      double h = (y2 - y1) * scale;

      if (isFront) {
        left = constraints.maxWidth - left - w;
      }

      final smoothKey = "$tag-${conf.toStringAsFixed(2)}";
      final prev = _prevDisplayBoxes[smoothKey];

      if (prev != null && prev.length == 4) {
        left = prev[0] + (left - prev[0]) * boxSmoothingAlpha;
        top = prev[1] + (top - prev[1]) * boxSmoothingAlpha;
        w = prev[2] + (w - prev[2]) * boxSmoothingAlpha;
        h = prev[3] + (h - prev[3]) * boxSmoothingAlpha;
      }

      _prevDisplayBoxes[smoothKey] = [left, top, w, h];

      left = left.clamp(0.0, constraints.maxWidth - 1.0);
      top = top.clamp(0.0, constraints.maxHeight - 1.0);
      w = w.clamp(1.0, constraints.maxWidth - left);
      h = h.clamp(1.0, constraints.maxHeight - top);

      Color boxColor;
      bool isInstant = det['instantAccept'] == true;
      bool isStable = (_sameClassCount >= _minSameClassFrames);

      if (isInstant && conf > 0.60) {
        boxColor = Colors.greenAccent;
      } else if (isInstant && conf > 0.40) {
        boxColor = Colors.lightGreenAccent;
      } else if (conf > 0.70 && isStable) {
        boxColor = Colors.green;
      } else if (conf > 0.50 && isStable) {
        boxColor = Colors.yellow;
      } else if (isInstant) {
        boxColor = Colors.yellowAccent;
      } else if (stability < _minSameClassFrames) {
        boxColor = Colors.orangeAccent;
      } else {
        boxColor = Colors.redAccent;
      }

      String labelText = "$tag ${(conf * 100).toStringAsFixed(1)}%";
      if (debugMode &&
          candidateClass != null &&
          candidateClass != tag &&
          det['instantAccept'] != true) {
        labelText += "\n→ $candidateClass?";
      }

      widgets.add(
        Positioned(
          left: left,
          top: top,
          width: w,
          height: h,
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(color: boxColor, width: 3),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Stack(
              children: [
                Align(
                  alignment: Alignment.topLeft,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: const BoxDecoration(
                      color: Colors.black87,
                      borderRadius: BorderRadius.only(
                        topLeft: Radius.circular(10),
                        bottomRight: Radius.circular(10),
                      ),
                    ),
                    child: Text(
                      labelText,
                      style: TextStyle(
                        color: boxColor,
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
                if (debugMode && stability < _minSameClassFrames)
                  Align(
                    alignment: Alignment.bottomRight,
                    child: Container(
                      padding: const EdgeInsets.all(4),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        "$stability/$_minSameClassFrames",
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 10,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      );
    }
    return widgets;
  }
}