// Refactored DetectionPage.dart
// Clean, stable, correct scaling, correct YOLO mapping, front-camera mirroring fixed.
// Works with FlutterVision + YOLO (cx,cy,w,h) format.

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

    setState(() {
      isLoaded = true;
    });
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
      modelPath: "assets/best_float16.tflite", // or fp32 model
      quantization: false,
      modelVersion: "yolov8",
      numThreads: 4,
      useGpu: false,
    );
  }

  Future<void> initializeCamera() async {
    if (controller != null) await controller!.dispose();

    controller = CameraController(
      cameras[selectedCameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await controller!.initialize();
    print("🔥 PREVIEW SIZE: ${controller!.value.previewSize}");
    print("🔥 ASPECT RATIO: ${controller!.value.aspectRatio}");
    setState(() {});
  }

  Future<void> switchCamera() async {
    if (cameras.length < 2) return;

    bool wasDetecting = isDetecting;
    if (wasDetecting) await stopDetection();

    selectedCameraIndex = (selectedCameraIndex + 1) % cameras.length;
    await initializeCamera();

    if (wasDetecting) {
      await Future.delayed(const Duration(milliseconds: 300));
      startDetection();
    }
  }

  Future<void> startDetection() async {
    if (controller == null || !controller!.value.isInitialized) return;

    isDetecting = true;
    setState(() {});

    if (controller!.value.isStreamingImages) return;

    await controller!.startImageStream((image) async {
      latestImage = image;

      if (isBusy || !isDetecting) return;

      isBusy = true;
      await runYolo(image);
      isBusy = false;
    });
  }

  Future<void> stopDetection() async {
    isDetecting = false;
    setState(() {});

    if (controller != null && controller!.value.isStreamingImages) {
      await controller!.stopImageStream();
    }
  }

  Future<void> runYolo(CameraImage image) async {
    final result = await vision.yoloOnFrame(
      bytesList: image.planes.map((p) => p.bytes).toList(),
      imageHeight: image.height,
      imageWidth: image.width,
      iouThreshold: 0.4,
      confThreshold: 0.3,
      classThreshold: 0.6,
    );

    if (!mounted) return;

    setState(() {
      yoloResults = result;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!isLoaded || controller == null || !controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Deteksi BISINDO"),
        actions: [
          IconButton(
            icon: const Icon(Icons.cameraswitch),
            onPressed: switchCamera,
          )
        ],
      ),

      body: LayoutBuilder(
        builder: (context, constraints) {
          // Use actual preview size from controller (previewSize exists after initialize)
          final previewSize = controller!.value.previewSize;
          if (previewSize == null) {
            return const SizedBox();
          }

          // Note: previewSize's width/height are in the sensor orientation (landscape),
          // so for portrait mode we swap them when building the preview box.
          double iw = previewSize.height.toDouble(); // displayed width in portrait
          double ih = previewSize.width.toDouble();  // displayed height in portrait

          // scale agar kamera muat di layar
          double ratioW = constraints.maxWidth / iw;
          double ratioH = constraints.maxHeight / ih;
          double scale = math.min(ratioW, ratioH);

          double previewW = iw * scale;
          double previewH = ih * scale;

          return Stack(
            children: [
              //------------------------- CAMERA PREVIEW -------------------------//
              Center(
                child: FittedBox(
                  fit: BoxFit.contain,
                  child: SizedBox(
                    width: iw,
                    height: ih,
                    child: Transform(
                      alignment: Alignment.center,
                      transform: Matrix4.rotationY(
                        cameras[selectedCameraIndex].lensDirection == CameraLensDirection.front
                            ? math.pi
                            : 0,
                      ),
                      child: CameraPreview(controller!),
                    ),
                  ),
                ),
              ),

              //------------------------- OVERLAY -------------------------//
              ...buildBoxes(Size(previewW, previewH), constraints),

              //------------------------- BUTTON -------------------------//
              Positioned(
                bottom: 30,
                left: 0,
                right: 0,
                child: Center(
                  child: ElevatedButton(
                    onPressed: isDetecting ? stopDetection : startDetection,
                    child: Text(isDetecting ? "Stop Deteksi" : "Mulai Deteksi"),
                  ),
                ),
              )
            ],
          );
        },
      ),
    );
  }

  List<Widget> buildBoxes(Size preview, BoxConstraints constraints) {
    if (yoloResults.isEmpty || latestImage == null) return [];

    // Use preview size from controller to compute mapping
    final previewSize = controller!.value.previewSize!;
    double iw = previewSize.height.toDouble();
    double ih = previewSize.width.toDouble();

    double ratioW = constraints.maxWidth / iw;
    double ratioH = constraints.maxHeight / ih;
    double scale = math.min(ratioW, ratioH);

    double previewW = iw * scale;
    double previewH = ih * scale;

    double offsetX = (constraints.maxWidth - previewW) / 2;
    double offsetY = (constraints.maxHeight - previewH) / 2;

    bool isFront = cameras[selectedCameraIndex].lensDirection == CameraLensDirection.front;

    return yoloResults.map((det) {
      final b = det["box"];

      double x1 = b[0] * scale + offsetX;
      double y1 = b[1] * scale + offsetY;
      double x2 = b[2] * scale + offsetX;
      double y2 = b[3] * scale + offsetY;

      double w = x2 - x1;
      double h = y2 - y1;

      if (isFront) x1 = constraints.maxWidth - x1 - w;

      return Positioned(
        left: x1,
        top: y1,
        width: w,
        height: h,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(color: Colors.greenAccent, width: 3),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Align(
            alignment: Alignment.topLeft,
            child: Container(
              padding: const EdgeInsets.all(4),
              color: Colors.black87,
              child: Text(
                "${det['tag']} ${(b[4] * 100).toStringAsFixed(1)}%",
                style: const TextStyle(color: Colors.white, fontSize: 12),
              ),
            ),
          ),
        ),
      );
    }).toList();
  }
}