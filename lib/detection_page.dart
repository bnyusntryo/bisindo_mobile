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

class _DetectionPageState extends State<DetectionPage> with TickerProviderStateMixin {
  CameraController? controller;

  // === SINGLE MODEL SYSTEM ===
  late FlutterVision vision;

  late AnimationController _pulseController;
  late AnimationController _shimmerController;

  List<CameraDescription> cameras = [];
  int selectedCameraIndex = 0;

  bool isLoaded = false;
  bool isDetecting = false;
  bool isBusy = false;
  bool debugMode = false;

  // Real-time detection state
  String? _currentDetection;
  double _currentConfidence = 0.0;

  // Statistics
  int _frameCount = 0;
  int _detectionCount = 0;

  // Device info
  String _deviceInfo = "";
  int _androidVersion = 0;

  // === ADAPTIVE SETTINGS ===
  late double baseConf;
  late double letterConf;
  late ResolutionPreset _resolution;
  late ImageFormatGroup _imageFormat;
  late int _numThreads;

  // Class definitions - Only letters
  final List<String> letters = const [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
  ];

  @override
  void initState() {
    super.initState();
    _initializeAnimations();
    _detectDeviceAndSetup();
    init();
  }

  void _initializeAnimations() {
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    )..repeat(reverse: true);

    _shimmerController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat();
  }

  /// === DEVICE DETECTION & ADAPTIVE SETUP ===
  void _detectDeviceAndSetup() {
    try {
      _androidVersion = _estimateAndroidVersion();
    } catch (e) {
      _androidVersion = 11;
    }

    _deviceInfo = "Android $_androidVersion";

    // === ADAPTIVE CONFIGURATION ===
    if (_androidVersion >= 14) {
      // NEW DEVICES (Android 14+)
      debugPrint("🔧 Detected MODERN device (Android $_androidVersion+)");

      baseConf = 0.15;
      letterConf = 0.22;    // Balanced threshold

      _resolution = ResolutionPreset.medium;
      _imageFormat = ImageFormatGroup.yuv420;
      _numThreads = 3;     // Lower threads to avoid throttling

    } else {
      // OLD DEVICES (Android 11-13)
      debugPrint("🔧 Detected LEGACY device (Android $_androidVersion)");

      baseConf = 0.12;
      letterConf = 0.18;    // Easier threshold

      _resolution = ResolutionPreset.medium;
      _imageFormat = ImageFormatGroup.yuv420;
      _numThreads = 4;     // Can use more threads safely
    }

    debugPrint("📊 Thresholds:");
    debugPrint("   Letter threshold: $letterConf");
    debugPrint("   Threads: $_numThreads");
  }

  int _estimateAndroidVersion() {
    final now = DateTime.now();
    if (now.year >= 2024) return 14;
    if (now.year >= 2023) return 13;
    if (now.year >= 2022) return 12;
    return 11;
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _shimmerController.dispose();
    controller?.dispose();

    try {
      vision.closeYoloModel();
    } catch (e) {
      debugPrint("Model disposal warning: $e");
    }

    super.dispose();
  }

  Future<void> init() async {
    try {
      cameras = await availableCameras();

      if (cameras.isEmpty) {
        debugPrint("❌ No cameras available");
        return;
      }

      vision = FlutterVision();

      await loadYoloModel();
      await initializeCamera();

      if (mounted) {
        setState(() => isLoaded = true);
      }
    } catch (e) {
      debugPrint("❌ Initialization error: $e");
      if (mounted) {
        _showErrorDialog("Initialization failed: $e");
      }
    }
  }

  Future<void> loadYoloModel() async {
    try {
      debugPrint("📥 Loading Abjad Model...");
      await vision.loadYoloModel(
        labels: "assets/labels_abjad.txt",
        modelPath: "assets/model_abjad.tflite",
        quantization: false,
        modelVersion: "yolov8",
        numThreads: _numThreads,
        useGpu: false,
      );
      debugPrint("✓ Abjad Model loaded");
      debugPrint("✓ SINGLE MODEL READY");
      debugPrint("   Device: $_deviceInfo");
      debugPrint("   Threads: $_numThreads");
      debugPrint("   Letter threshold: $letterConf");
    } catch (e) {
      debugPrint("❌ Model load error: $e");
      throw Exception("Failed to load model: $e");
    }
  }

  Future<void> initializeCamera() async {
    try {
      if (controller != null) {
        await controller!.dispose();
      }

      final frontIndex = cameras.indexWhere(
            (cam) => cam.lensDirection == CameraLensDirection.front,
      );
      selectedCameraIndex = frontIndex != -1 ? frontIndex : 0;

      controller = CameraController(
        cameras[selectedCameraIndex],
        _resolution,
        enableAudio: false,
        imageFormatGroup: _imageFormat,
      );

      await controller!.initialize();

      if (_androidVersion >= 14) {
        try {
          await controller!.setFocusMode(FocusMode.auto);
          debugPrint("✓ Focus mode set");
        } catch (e) {
          debugPrint("⚠️ Could not set focus mode: $e");
        }
      }

      try {
        await controller!.lockCaptureOrientation(DeviceOrientation.portraitUp);
      } catch (e) {
        debugPrint("⚠️ Could not lock orientation: $e");
      }

      if (mounted) {
        setState(() {});
      }
    } catch (e) {
      debugPrint("❌ Camera initialization error: $e");
      throw Exception("Failed to initialize camera: $e");
    }
  }

  Future<void> switchCamera() async {
    if (cameras.length < 2) {
      debugPrint("⚠️ Only one camera available");
      return;
    }

    final wasDetecting = isDetecting;
    if (wasDetecting) await stopDetection();

    selectedCameraIndex = (selectedCameraIndex + 1) % cameras.length;
    await initializeCamera();

    _resetStats();

    if (wasDetecting) {
      await Future.delayed(const Duration(milliseconds: 150));
      startDetection();
    }
  }

  void _resetStats() {
    _currentDetection = null;
    _currentConfidence = 0.0;
    _frameCount = 0;
    _detectionCount = 0;
  }

  Future<void> startDetection() async {
    if (controller == null || !controller!.value.isInitialized) {
      debugPrint("⚠️ Camera not ready");
      return;
    }

    if (controller!.value.isStreamingImages) {
      debugPrint("⚠️ Already streaming");
      return;
    }

    setState(() => isDetecting = true);

    try {
      await controller!.startImageStream((image) async {
        if (!isDetecting || isBusy) return;

        _frameCount++;

        // === ADAPTIVE FRAME SKIPPING ===
        if (_androidVersion >= 14) {
          if (_frameCount % 2 != 0) return; // Process every 2nd frame
        }

        isBusy = true;

        try {
          await _processFrame(image);
        } catch (e) {
          debugPrint("Frame processing error: $e");
        } finally {
          isBusy = false;
        }
      });
    } catch (e) {
      debugPrint("❌ Failed to start detection: $e");
      setState(() => isDetecting = false);
    }
  }

  Future<void> stopDetection() async {
    if (!isDetecting) return;

    setState(() => isDetecting = false);

    try {
      if (controller != null && controller!.value.isStreamingImages) {
        await controller!.stopImageStream();
      }
    } catch (e) {
      debugPrint("⚠️ Error stopping stream: $e");
    }

    _resetStats();
  }

  Future<void> _processFrame(CameraImage image) async {
    final bytesListData = image.planes.map((p) => p.bytes).toList();

    try {
      final results = await vision.yoloOnFrame(
        bytesList: bytesListData,
        imageHeight: image.height,
        imageWidth: image.width,
        iouThreshold: 0.30,
        confThreshold: baseConf,
        classThreshold: 0.20,
      );

      final bestDetection = _selectBestDetection(results, image);

      if (!mounted) return;

      if (bestDetection == null) {
        if (_currentDetection != null) {
          setState(() {
            _currentDetection = null;
            _currentConfidence = 0.0;
          });
        }
        return;
      }

      _detectionCount++;

      setState(() {
        _currentDetection = bestDetection['tag'];
        _currentConfidence = bestDetection['confidence'];
      });
    } catch (e) {
      debugPrint("Error in processing: $e");
    }
  }

  Map<String, dynamic>? _selectBestDetection(
      List<dynamic> results,
      CameraImage image,
      ) {
    final imgArea = image.width.toDouble() * image.height.toDouble();
    final candidates = _processCandidates(results, imgArea);

    if (candidates.isEmpty) return null;

    // Sort by confidence, then by area
    candidates.sort((a, b) {
      // 1. Higher confidence wins
      final confDiff = b['confidence'].compareTo(a['confidence']);
      if (confDiff.abs() > 0.05) return confDiff;

      // 2. Prefer ideal area (10-50% of screen)
      final aArea = a['areaRatio'] as double;
      final bArea = b['areaRatio'] as double;

      const idealMin = 0.10;
      const idealMax = 0.50;

      final aInRange = aArea >= idealMin && aArea <= idealMax;
      final bInRange = bArea >= idealMin && bArea <= idealMax;

      if (aInRange != bInRange) {
        return aInRange ? -1 : 1;
      }

      // 3. Within range, prefer larger
      return bArea.compareTo(aArea);
    });

    return candidates.first;
  }

  List<Map<String, dynamic>> _processCandidates(
      List<dynamic> results,
      double imgArea,
      ) {
    final List<Map<String, dynamic>> candidates = [];

    for (var result in results) {
      final tag = (result['tag'] ?? '').toString().trim();
      if (tag.isEmpty) continue;

      // Only accept letters
      if (!letters.contains(tag)) continue;

      final box = result['box'];
      if (box == null || box.length < 5) continue;

      final confidence = _toDouble(box[4]);
      final boxWidth = _toDouble(box[2]);
      final boxHeight = _toDouble(box[3]);
      final boxArea = boxWidth * boxHeight;
      final areaRatio = boxArea / imgArea;

      double requiredConf = letterConf;

      // === ADAPTIVE FILTERING ===
      if (areaRatio > 0.60) {
        requiredConf += 0.10; // Larger boxes need higher confidence
      } else if (areaRatio < 0.05) {
        requiredConf += 0.05; // Smaller boxes need higher confidence
      }

      // Slightly lower threshold on newer devices
      if (_androidVersion >= 14) {
        requiredConf -= 0.03;
      }

      // Apply filters
      if (confidence < requiredConf) continue;
      if (boxWidth < 30 || boxHeight < 30) continue;
      if (areaRatio < 0.015 || areaRatio > 0.75) continue;

      // Aspect ratio filter
      final aspectRatio = boxWidth / boxHeight;
      if (aspectRatio < 0.3 || aspectRatio > 3.0) continue;

      candidates.add({
        'tag': tag,
        'confidence': confidence,
        'area': boxArea,
        'areaRatio': areaRatio,
        'boxWidth': boxWidth,
        'boxHeight': boxHeight,
        'aspectRatio': aspectRatio,
      });
    }

    return candidates;
  }

  double _toDouble(dynamic value) {
    if (value is double) return value;
    if (value is int) return value.toDouble();
    if (value is num) return value.toDouble();
    return 0.0;
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!isLoaded || controller == null || !controller!.value.isInitialized) {
      return _buildLoadingScreen();
    }

    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(child: _buildCameraPreview()),
          Positioned.fill(child: _buildModernOverlay()),
          _buildModernHeader(),
          if (isDetecting && _currentDetection != null) _buildModernDetectionCard(),
          _buildModernControlPanel(),
          if (debugMode) _buildDebugInfo(),
        ],
      ),
    );
  }

  Widget _buildLoadingScreen() {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF667eea),
              Color(0xFF764ba2),
              Color(0xFFf093fb),
            ],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  shape: BoxShape.circle,
                ),
                child: const CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 3,
                ),
              ),
              const SizedBox(height: 32),
              const Text(
                "Loading BISINDO",
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.5,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                "Preparing AI Model...",
                style: TextStyle(
                  color: Colors.white.withOpacity(0.8),
                  fontSize: 14,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCameraPreview() {
    return FittedBox(
      fit: BoxFit.cover,
      child: SizedBox(
        width: controller!.value.previewSize!.height,
        height: controller!.value.previewSize!.width,
        child: Transform(
          alignment: Alignment.center,
          transform: Matrix4.rotationY(
            cameras[selectedCameraIndex].lensDirection == CameraLensDirection.front ? math.pi : 0,
          ),
          child: CameraPreview(controller!),
        ),
      ),
    );
  }

  Widget _buildModernOverlay() {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Colors.black.withOpacity(0.6),
            Colors.transparent,
            Colors.transparent,
            Colors.black.withOpacity(0.8),
          ],
          stops: const [0.0, 0.2, 0.65, 1.0],
        ),
      ),
    );
  }

  Widget _buildModernHeader() {
    return Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Colors.purple.shade400,
                      Colors.blue.shade400,
                    ],
                  ),
                  borderRadius: BorderRadius.circular(30),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.purple.withOpacity(0.3),
                      blurRadius: 20,
                      offset: const Offset(0, 8),
                    ),
                  ],
                ),
                child: Row(
                  children: [
                    const Text("✨", style: TextStyle(fontSize: 24)),
                    const SizedBox(width: 10),
                    const Text(
                      "BISINDO",
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w900,
                        color: Colors.white,
                        letterSpacing: 1.2,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.3),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Text(
                        "ABJAD",
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              Row(
                children: [
                  _buildModernIconButton(
                    icon: Icons.flip_camera_ios_rounded,
                    onPressed: switchCamera,
                  ),
                  const SizedBox(width: 10),
                  _buildModernIconButton(
                    icon: debugMode ? Icons.visibility_off : Icons.visibility,
                    onPressed: () => setState(() => debugMode = !debugMode),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildModernIconButton({
    required IconData icon,
    required VoidCallback onPressed,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.2),
        shape: BoxShape.circle,
        border: Border.all(
          color: Colors.white.withOpacity(0.3),
          width: 2,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.2),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: IconButton(
        icon: Icon(icon, color: Colors.white),
        onPressed: onPressed,
      ),
    );
  }

  Widget _buildModernDetectionCard() {
    final tag = _currentDetection!;
    final gradient = [const Color(0xFF48c6ef), const Color(0xFF6f86d6)];

    return Positioned(
      left: 20,
      right: 20,
      top: MediaQuery.of(context).size.height * 0.48,
      child: AnimatedOpacity(
        opacity: 1.0,
        duration: const Duration(milliseconds: 200),
        child: AnimatedBuilder(
          animation: _shimmerController,
          builder: (context, child) {
            return Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(32),
                gradient: LinearGradient(
                  colors: gradient,
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                boxShadow: [
                  BoxShadow(
                    color: gradient[0].withOpacity(0.5),
                    blurRadius: 30,
                    spreadRadius: 5,
                    offset: const Offset(0, 15),
                  ),
                ],
              ),
              child: Container(
                margin: const EdgeInsets.all(3),
                padding: const EdgeInsets.all(32),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(30),
                ),
                child: Column(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(colors: gradient),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: const [
                          Icon(Icons.fingerprint, color: Colors.white, size: 18),
                          SizedBox(width: 8),
                          Text(
                            "LETTER",
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                              fontWeight: FontWeight.bold,
                              letterSpacing: 2,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 24),
                    AnimatedSwitcher(
                      duration: const Duration(milliseconds: 300),
                      transitionBuilder: (child, animation) {
                        return ScaleTransition(
                          scale: Tween<double>(begin: 0.8, end: 1.0).animate(animation),
                          child: FadeTransition(opacity: animation, child: child),
                        );
                      },
                      child: Text(
                        tag,
                        key: ValueKey(tag),
                        style: TextStyle(
                          fontSize: 96,
                          fontWeight: FontWeight.w900,
                          foreground: Paint()
                            ..shader = LinearGradient(
                              colors: gradient,
                            ).createShader(const Rect.fromLTWH(0, 0, 200, 100)),
                          letterSpacing: 2,
                          height: 1.0,
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                    _buildModernConfidenceBar(gradient),
                  ],
                ),
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildModernConfidenceBar(List<Color> gradient) {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              "ACCURACY",
              style: TextStyle(
                color: Colors.grey,
                fontSize: 11,
                fontWeight: FontWeight.w600,
                letterSpacing: 1.5,
              ),
            ),
            Text(
              "${(_currentConfidence * 100).toStringAsFixed(0)}%",
              style: TextStyle(
                color: gradient[0],
                fontSize: 16,
                fontWeight: FontWeight.w900,
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        Container(
          height: 8,
          decoration: BoxDecoration(
            color: Colors.grey.shade200,
            borderRadius: BorderRadius.circular(10),
          ),
          child: AnimatedFractionallySizedBox(
            duration: const Duration(milliseconds: 400),
            curve: Curves.easeOut,
            widthFactor: _currentConfidence.clamp(0.0, 1.0),
            alignment: Alignment.centerLeft,
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(colors: gradient),
                borderRadius: BorderRadius.circular(10),
                boxShadow: [
                  BoxShadow(
                    color: gradient[0].withOpacity(0.4),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildModernControlPanel() {
    return Positioned(
      bottom: 40,
      left: 0,
      right: 0,
      child: Column(
        children: [
          if (isDetecting)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF11998e), Color(0xFF38ef7d)],
                ),
                borderRadius: BorderRadius.circular(25),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF11998e).withOpacity(0.4),
                    blurRadius: 15,
                    offset: const Offset(0, 5),
                  ),
                ],
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  AnimatedBuilder(
                    animation: _pulseController,
                    builder: (context, child) {
                      return Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          color: Colors.white,
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                              color: Colors.white.withOpacity(0.8),
                              blurRadius: 8 * _pulseController.value,
                              spreadRadius: 2 * _pulseController.value,
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                  const SizedBox(width: 12),
                  const Text(
                    "ABJAD DETECTION ACTIVE",
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1.5,
                    ),
                  ),
                ],
              ),
            ),
          const SizedBox(height: 20),
          GestureDetector(
            onTap: isDetecting ? stopDetection : startDetection,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: LinearGradient(
                  colors: isDetecting
                      ? [const Color(0xFFeb3349), const Color(0xFFf45c43)]
                      : [const Color(0xFF667eea), const Color(0xFF764ba2)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                boxShadow: [
                  BoxShadow(
                    color: isDetecting
                        ? const Color(0xFFeb3349).withOpacity(0.5)
                        : const Color(0xFF667eea).withOpacity(0.5),
                    blurRadius: 25,
                    spreadRadius: 2,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: Icon(
                isDetecting ? Icons.stop_rounded : Icons.play_arrow_rounded,
                color: Colors.white,
                size: 42,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDebugInfo() {
    return Positioned(
      bottom: 140,
      left: 16,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.85),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: Colors.blue.withOpacity(0.6),
            width: 2,
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "🔍 DEBUG INFO",
              style: TextStyle(
                color: Colors.blueAccent,
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
            ),
            const SizedBox(height: 8),
            _buildDebugRow("Device", _deviceInfo),
            _buildDebugRow("Detection", _currentDetection ?? 'None'),
            _buildDebugRow("Confidence", "${(_currentConfidence * 100).toStringAsFixed(1)}%"),
            const Divider(color: Colors.grey, height: 12, thickness: 0.5),
            _buildDebugRow("Frames", "$_frameCount"),
            _buildDebugRow("Detections", "$_detectionCount"),
            _buildDebugRow("Threads", "$_numThreads"),
            _buildDebugRow("Threshold", "${(letterConf * 100).toInt()}%"),
          ],
        ),
      ),
    );
  }

  Widget _buildDebugRow(String label, String value, {Color? color}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 3),
      child: Row(
        children: [
          Text(
            "$label: ",
            style: const TextStyle(color: Colors.grey, fontSize: 9),
          ),
          Text(
            value,
            style: TextStyle(
              color: color ?? Colors.white,
              fontSize: 9,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}