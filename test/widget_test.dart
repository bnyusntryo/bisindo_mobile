import 'package:flutter_bisindo/main.dart';
import 'package:flutter_test/flutter_test.dart';
// import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:camera_platform_interface/camera_platform_interface.dart';

class MockCameraPlatform extends CameraPlatform {
  @override
  Future<List<CameraDescription>> availableCameras() async {
    return [
      CameraDescription(
        name: 'MockCamera',
        lensDirection: CameraLensDirection.back,
        sensorOrientation: 0,
      ),
    ];
  }
  Future<void> init() async {}

  @override
  Future<void> dispose(int cameraId) async {}

  @override
  Future<int> createCamera(
      CameraDescription cameraDescription,
      ResolutionPreset? resolutionPreset, {
        bool enableAudio = false,
      }) async {
    return 0; // id kamera dummy
  }

  @override
  Stream<CameraInitializedEvent> onCameraInitialized(int cameraId) =>
      const Stream.empty();
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    CameraPlatform.instance = MockCameraPlatform();
  });

  testWidgets('Bisindo widget builds correctly', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());
    await tester.pumpAndSettle();

    expect(find.text('No camera available'), findsNothing);
  });
}
