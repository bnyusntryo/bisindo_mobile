import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'dart:math' as math;
import 'package:flutter/services.dart';

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

  // === SIMPLE TEMPORAL STABILIZER ===
  // String? _lastStableTag;
  // int _stableFrameCount = 0;
  // final int _requiredStableFrames = 3; // need 3 consecutive frames to accept

  // === CONFIGURABLE PARAMETERS ===
  double modelConfThreshold = 0.30; // model filter (used when reading results)
  double modelIouThreshold = 0.45;
  double modelClassThreshold = 0.30;

  double displayConfThreshold = 0.50;

  bool forceFrontCamera = true;
  bool autoMirror = true;
  bool debugMode = true;
  bool bypassFilters = false;

  final List<String> singleLetters = const [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
  ];
  // TAMBAHKAN DAFTAR INI
  final List<String> bisindoWords = const [
    'Aku','Apa','Ayah','Baik','Bantu','Bermain','Dia','Jangan','Kakak','Kamu','Kapan','Keren','Kerja','Maaf','Marah','Minum','Rumah','Sabar','Sedih','Senang','Suka'
  ];

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
    try {
      await vision.loadYoloModel(
        labels: "assets/labels.txt",
        modelPath: "assets/best_float32_V2.tflite",
        quantization: false,
        modelVersion: "yolov8",
        numThreads: 4,
        useGpu: false,
      );
      print("✓ Model loaded successfully");
    } catch (e) {
      print("❌ loadYoloModel error: $e");
      rethrow;
    }
  }

  Future<void> initializeCamera() async {
    if (controller != null) await controller!.dispose();

    if (forceFrontCamera) {
      final frontCameraIndex = cameras.indexWhere(
            (camera) => camera.lensDirection == CameraLensDirection.front,
      );
      if (frontCameraIndex != -1) {
        selectedCameraIndex = frontCameraIndex;
      }
    }

    controller = CameraController(
      cameras[selectedCameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await controller!.initialize();

    // Lock capture orientation to portrait up (optional)
    try {
      await controller!.lockCaptureOrientation(DeviceOrientation.portraitUp);
    } catch (e) {
      // ignore if not supported
    }

    print("📸 Camera initialized");
    print("   Preview size: ${controller!.value.previewSize}");

    setState(() {});
  }

  Future<void> switchCamera() async {
    if (cameras.length < 2) return;

    bool wasDetecting = isDetecting;
    if (wasDetecting) await stopDetection();

    selectedCameraIndex = (selectedCameraIndex + 1) % cameras.length;
    await initializeCamera();

    // reset state
    yoloResults.clear();

    if (wasDetecting) {
      await Future.delayed(const Duration(milliseconds: 300));
      startDetection();
    }
  }

  Future<void> startDetection() async {
    if (controller == null || !controller!.value.isInitialized) return;

    print("\n🚀 STARTING DETECTION");
    isDetecting = true;
    setState(() {});

    if (controller!.value.isStreamingImages) {
      print("⚠️ Already streaming, skipping startImageStream");
      return;
    }

    await controller!.startImageStream((image) async {
      latestImage = image;

      if (isBusy || !isDetecting) return;
      isBusy = true;

      try {
        await runYolo(image);
      } catch (e, st) {
        print("❌ runYolo exception: $e");
        print(st);
      } finally {
        isBusy = false;
      }
    });

    print("✅ Image stream started");
  }

  Future<void> stopDetection() async {
    isDetecting = false;
    setState(() {});

    if (controller != null && controller!.value.isStreamingImages) {
      await controller!.stopImageStream();
    }

    yoloResults.clear();
  }

  double _toDouble(dynamic v) {
    if (v is double) return v;
    if (v is int) return v.toDouble();
    if (v is num) return v.toDouble();
    return 0.0;
  }

  Future<void> runYolo(CameraImage image) async {
    // Run model
    final result = await vision.yoloOnFrame(
      bytesList: image.planes.map((p) => p.bytes).toList(),
      imageHeight: image.height,
      imageWidth: image.width,
      iouThreshold: modelIouThreshold,
      confThreshold: modelConfThreshold,
      classThreshold: modelClassThreshold,
    );

    if (debugMode) {
      print("===== YOLO RAW (${result.length}) =====");
      for (var r in result) {
        print("TAG: ${r['tag']} | BOX: ${r['box']}");
      }
    }

    if (result.isEmpty) {
      // Tidak ada deteksi, bersihkan layar
      if (yoloResults.isNotEmpty) {
        setState(() => yoloResults = []);
      }
      return;
    }

    // === FILTER BARU: HANYA PAKAI SLIDER ===
    List<Map<String, dynamic>> validDetections = [];

    for (var r in result) {
      final tag = (r['tag'] ?? '').toString().trim();
      final box = r['box'];
      final conf = _toDouble(box[4]);

      // Tentukan tipe kelas
      bool isWord = bisindoWords.contains(tag);
      bool isLetter = singleLetters.contains(tag);

      // 1. Jika bukan kata atau huruf yang dikenal, buang
      if (!isWord && !isLetter) continue;

      // 2. HANYA filter berdasarkan slider
      if (conf < displayConfThreshold && !bypassFilters) {
        if (debugMode) print("  ⚠️ $tag: conf ${(conf * 100).toStringAsFixed(1)}% DIBAWAH threshold ($displayConfThreshold)");
        continue;
      }

      // 3. Lolos! Tambahkan ke daftar
      validDetections.add({
        'tag': tag,
        'box': box, // Simpan box mentah
        'conf': conf, // Simpan conf yang sudah di-parse
      });
    }
    // === AKHIR FILTER ===


    if (validDetections.isEmpty) {
      // Tidak ada yang lolos filter, bersihkan layar
      if (yoloResults.isNotEmpty) {
        setState(() => yoloResults = []);
      }
      return;
    }

    // Pilih confidence tertinggi DARI YANG SUDAH LOLOS FILTER
    validDetections.sort((a, b) => b['conf'].compareTo(a['conf']));
    final best = validDetections.first;
    final bestTag = best['tag'];

    if (debugMode) {
      print("Best filtered: $bestTag (${(best['conf'] * 100).toStringAsFixed(1)}%)");
    }

    // === MODE INSTAN: LANGSUNG TAMPILKAN ===
    if (!mounted) return;
    setState(() {
      yoloResults = [best]; // 'best' sudah berisi 'tag' dan 'box'
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
        title: const Text("🤟 BISINDO Detector"),
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
        ],
      ),
      body: Stack(
        children: [
          // Camera Preview - Full Screen
          Positioned.fill(
            child: FittedBox(
              fit: BoxFit.cover,
              child: SizedBox(
                width: controller!.value.previewSize!.height,
                height: controller!.value.previewSize!.width,
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

          // Gradient overlay untuk readability
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            height: 300,
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [
                    Colors.black.withOpacity(0.8),
                    Colors.black.withOpacity(0.5),
                    Colors.transparent,
                  ],
                ),
              ),
            ),
          ),

          // Classification Results Display (Choice B - fancy box)
          if (yoloResults.isNotEmpty) _buildClassificationDisplay(),

          // Start/Stop Button
          Positioned(
            bottom: 30,
            left: 0,
            right: 0,
            child: Center(
              child: ElevatedButton.icon(
                onPressed: isDetecting ? stopDetection : startDetection,
                icon: Icon(isDetecting ? Icons.stop : Icons.play_arrow),
                label: Text(
                  isDetecting ? "Stop" : "Start",
                  style: const TextStyle(fontSize: 18),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: isDetecting ? Colors.red : Colors.teal,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 40,
                    vertical: 20,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  elevation: 8,
                ),
              ),
            ),
          ),

          // Debug Controls (Optional)
          if (debugMode) _buildSimpleDebugPanel(),
        ],
      ),
    );
  }

  Widget _buildSimpleDebugPanel() {
    return Positioned(
      top: 10,
      left: 10,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.black87,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              "Debug Info",
              style: TextStyle(
                color: Colors.greenAccent,
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              "FPS: ${controller?.value.isStreamingImages == true ? '30' : '0'}",
              style: const TextStyle(color: Colors.white, fontSize: 10),
            ),
            Text(
              "Detections: ${yoloResults.length}",
              style: const TextStyle(color: Colors.white, fontSize: 10),
            ),
            Text(
              "Model Conf: ${modelConfThreshold.toStringAsFixed(2)}",
              style: const TextStyle(color: Colors.white, fontSize: 10),
            ),
            Text(
              "Display Conf: ${displayConfThreshold.toStringAsFixed(2)}",
              style: const TextStyle(color: Colors.white, fontSize: 10),
            ),
            const SizedBox(height: 8),
            // Quick adjustment buttons
            Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.remove, color: Colors.white, size: 16),
                  onPressed: () {
                    setState(() {
                      displayConfThreshold = (displayConfThreshold - 0.1).clamp(0.2, 0.9);
                    });
                  },
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(minWidth: 24, minHeight: 24),
                ),
                Text(
                  "${(displayConfThreshold * 100).toStringAsFixed(0)}%",
                  style: const TextStyle(color: Colors.white, fontSize: 10),
                ),
                IconButton(
                  icon: const Icon(Icons.add, color: Colors.white, size: 16),
                  onPressed: () {
                    setState(() {
                      displayConfThreshold = (displayConfThreshold + 0.1).clamp(0.2, 0.9);
                    });
                  },
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(minWidth: 24, minHeight: 24),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildClassificationDisplay() {
    if (yoloResults.isEmpty) return const SizedBox.shrink();

    // Ambil data dari yoloResults
    final Map<String, dynamic> best = yoloResults.first;
    final String tag = (best['tag'] ?? '?').toString();
    // Ambil confidence dari 'box' (sesuai format flutter_vision) atau 'conf' jika ada
    final double confidence = best.containsKey('conf')
        ? _toDouble(best['conf'])
        : _toDouble(best['box'][4]);

    // Tentukan tipe
    final bool isWord = bisindoWords.contains(tag);
    final bool isLetter = singleLetters.contains(tag);

    // Tentukan Tampilan
    final Color primaryColor = isWord ? Colors.blue : (isLetter ? Colors.green : Colors.purple);
    final IconData icon = isWord ? Icons.gesture : (isLetter ? Icons.abc : Icons.help_outline);
    final String typeLabel = isWord ? "KATA" : (isLetter ? "HURUF" : "UNKNOWN");

    // Safety clamp untuk confidence
    final double confClamped = confidence.clamp(0.0, 1.0);

    return Positioned(
      bottom: 120,
      left: 20,
      right: 20,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.3), blurRadius: 10, offset: const Offset(0, 5))],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Type Badge
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: primaryColor.withOpacity(0.2),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: primaryColor, width: 2),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(icon, color: primaryColor, size: 16),
                  const SizedBox(width: 4),
                  Text(
                    typeLabel,
                    style: TextStyle(color: primaryColor, fontSize: 12, fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 12),

            // Main Classification Result
            Text(
              tag,
              style: const TextStyle(fontSize: 48, fontWeight: FontWeight.bold, color: Colors.black87),
            ),

            const SizedBox(height: 8),

            // Confidence Bar
            Column(
              children: [
                Text(
                  "Confidence: ${(confClamped * 100).toStringAsFixed(0)}%",
                  style: TextStyle(color: Colors.grey[600], fontSize: 14),
                ),
                const SizedBox(height: 8),
                Container(
                  height: 8,
                  decoration: BoxDecoration(color: Colors.grey[200], borderRadius: BorderRadius.circular(4)),
                  child: FractionallySizedBox(
                    widthFactor: confClamped,
                    alignment: Alignment.centerLeft,
                    child: Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: confClamped > 0.7
                              ? [Colors.green, Colors.greenAccent]
                              : confClamped > 0.5
                              ? [Colors.yellow, Colors.yellowAccent]
                              : [Colors.orange, Colors.orangeAccent],
                        ),
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  // Test helper
  void _createTestDetection(String className) {
    if (latestImage == null) return;

    setState(() {
      yoloResults = [{
        "tag": className,
        "box": [
          latestImage!.width * 0.25,
          latestImage!.height * 0.25,
          latestImage!.width * 0.75,
          latestImage!.height * 0.75,
          0.95,
        ],
      }];
    });

    print("📍 Test detection created for: $className");
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("Testing: $className"), backgroundColor: Colors.green, duration: const Duration(seconds: 1)),
    );
  }
}