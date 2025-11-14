import 'dart:math' as math;

/// Sigmoid
double sigmoid(double x) => 1 / (1 + math.exp(-x));

/// Konversi [x, y, w, h] → [x1, y1, x2, y2]
List<double> xywh2xyxy(List<double> box) {
  final x = box[0], y = box[1], w = box[2], h = box[3];
  return [x - w / 2, y - h / 2, x + w / 2, y + h / 2];
}

/// Hitung IOU (untuk NMS)
double iou(List<double> box1, List<double> box2) {
  final xx1 = math.max(box1[0], box2[0]);
  final yy1 = math.max(box1[1], box2[1]);
  final xx2 = math.min(box1[2], box2[2]);
  final yy2 = math.min(box1[3], box2[3]);
  final interArea = math.max(0, xx2 - xx1) * math.max(0, yy2 - yy1);
  final box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
  final box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
  return interArea / (box1Area + box2Area - interArea + 1e-6);
}

/// NMS sederhana
List<int> nms(List<List<double>> boxes, List<double> scores, double iouThres) {
  final idxs = List.generate(scores.length, (i) => i)
    ..sort((a, b) => scores[b].compareTo(scores[a]));
  final selected = <int>[];

  while (idxs.isNotEmpty) {
    final current = idxs.removeAt(0);
    selected.add(current);
    idxs.removeWhere((i) => iou(boxes[current], boxes[i]) > iouThres);
  }
  return selected;
}

/// Decoding YOLOv8 TFLite BISINDO (Final Calibrated Confidence)
List<Map<String, dynamic>> decodeYolo(
    List<List<List<double>>> output,
    int origW,
    int origH, {
      double confThres = 0.25,
      double iouThres = 0.45,
    }) {
  // --- Struktur model kamu: (1, 30, 8400)
  final raw = output[0]; // ambil batch ke-0
  final out = List.generate(8400,
          (i) => List.generate(30, (j) => raw[j][i].toDouble())); // transpose ke (8400, 30)

  final decodedBoxes = <List<double>>[];
  final decodedScores = <double>[];
  final decodedClasses = <int>[];

  final strides = [8, 16, 32];
  final gridSizes = [80, 40, 20];
  final numCells = [6400, 1600, 400]; // total 8400
  int offset = 0;

  for (int h = 0; h < 3; h++) {
    final stride = strides[h];
    final g = gridSizes[h];
    for (int i = 0; i < numCells[h]; i++) {
      final pred = List<double>.from(out[offset + i]);

      // grid posisi
      final xCell = i % g;
      final yCell = (i / g).floor();

      // Kalibrasi seperti Python (pred sigmoid internal, ditingkatkan kontras)
      for (int j = 4; j < pred.length; j++) {
        pred[j] = math.pow(pred[j].clamp(0.0, 1.0), 2.5).toDouble();
      }

      // Decode xywh
      final xy = [(pred[0] * 2 - 0.5 + xCell) * stride,
        (pred[1] * 2 - 0.5 + yCell) * stride];
      final wh = [math.pow(pred[2] * 2, 2) * stride,
        math.pow(pred[3] * 2, 2) * stride];

      final conf = pred[4];
      final cls = pred.sublist(5);
      final clsIdx = cls.indexOf(cls.reduce(math.max));
      final score = conf * cls[clsIdx];

      if (score > confThres) {
        final box = xywh2xyxy([
          xy[0].toDouble(),
          xy[1].toDouble(),
          wh[0].toDouble(),
          wh[1].toDouble(),
        ]);

        // Skala kembali ke ukuran asli frame kamera
        box[0] = (box[0] * origW / 640).toDouble();
        box[1] = (box[1] * origH / 640).toDouble();
        box[2] = (box[2] * origW / 640).toDouble();
        box[3] = (box[3] * origH / 640).toDouble();

        decodedBoxes.add(List<double>.from(box));
        decodedScores.add(score.toDouble());
        decodedClasses.add(clsIdx);
      }
    }
    offset += numCells[h];
  }

  // --- Non-Max Suppression
  final keep = nms(decodedBoxes, decodedScores, iouThres);

  final labels = List.generate(26, (i) => String.fromCharCode(65 + i)); // A–Z
  final results = <Map<String, dynamic>>[];

  for (final i in keep) {
    results.add({
      'box': decodedBoxes[i],
      'score': decodedScores[i],
      'cls': labels[decodedClasses[i] % labels.length],
    });
  }

  print("✅ ${results.length} deteksi ditemukan");
  return results;
}
