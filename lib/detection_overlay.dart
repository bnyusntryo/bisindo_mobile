import 'package:flutter/material.dart';

class DetectionOverlay extends StatelessWidget {
  final List<Map<String, dynamic>> results;
  final double previewW;
  final double previewH;

  const DetectionOverlay({
    super.key,
    required this.results,
    required this.previewW,
    required this.previewH,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: results.map((det) {
        final box = det['box']; // [x1, y1, x2, y2]
        final label = det['cls'];
        final score = det['score'];

        final left = box[0] / previewW * MediaQuery.of(context).size.width;
        final top = box[1] / previewH * MediaQuery.of(context).size.height;
        final width = (box[2] - box[0]) / previewW * MediaQuery.of(context).size.width;
        final height = (box[3] - box[1]) / previewH * MediaQuery.of(context).size.height;

        return Positioned(
          left: left,
          top: top,
          width: width,
          height: height,
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(color: Colors.greenAccent, width: 2),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Align(
              alignment: Alignment.topLeft,
              child: Container(
                color: Colors.black54,
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                child: Text(
                  '$label ${(score * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(color: Colors.white, fontSize: 12),
                ),
              ),
            ),
          ),
        );
      }).toList(),
    );
  }
}
