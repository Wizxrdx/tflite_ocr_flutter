import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../models/text_detection.dart';

/// A widget that displays an image with overlay boxes from text detection.
///
/// Accepts either [imageBytes] or [imageProvider], and a list of [polygons]
/// containing detected text regions.
/// The polygons are overlaid on the image with proper coordinate scaling.
class ImageDisplayWidget extends StatelessWidget {
  final Uint8List? imageBytes;
  final ImageProvider? imageProvider;
  final Size sourceImageSize;
  final List<Polygon> polygons;
  final BoxDecoration? decoration;
  final BoxFit boxFit;

  const ImageDisplayWidget({
    super.key,
    this.imageBytes,
    this.imageProvider,
    required this.sourceImageSize,
    required this.polygons,
    this.decoration,
    this.boxFit = BoxFit.contain,
  }) : assert(imageBytes != null || imageProvider != null,
            'Either imageBytes or imageProvider must be provided');

  /// Create from raw detection output (List<List<List<double>>>)
  factory ImageDisplayWidget.fromRawOutput({
    required Uint8List imageBytes,
    required Size sourceImageSize,
    required List<List<List<double>>> rawPolygons,
    BoxDecoration? decoration,
    BoxFit boxFit = BoxFit.contain,
  }) {
    final result = TextDetectionResult.fromRawOutput(rawPolygons);
    return ImageDisplayWidget(
      imageBytes: imageBytes,
      sourceImageSize: sourceImageSize,
      polygons: result.polygons,
      decoration: decoration,
      boxFit: boxFit,
    );
  }

  @override
  Widget build(BuildContext context) {
    final image = imageBytes != null
        ? Image.memory(
            imageBytes!,
            fit: boxFit,
            alignment: Alignment.center,
          )
        : Image(
            image: imageProvider!,
            fit: boxFit,
            alignment: Alignment.center,
          );

    return Container(
      decoration: decoration ??
          BoxDecoration(
            color: Colors.black12,
            borderRadius: BorderRadius.circular(12),
          ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          return Stack(
            fit: StackFit.expand,
            children: [
              image,
              CustomPaint(
                painter: _BoxOverlayPainter(
                  polygons: polygons,
                  sourceImageSize: sourceImageSize,
                  canvasSize: Size(constraints.maxWidth, constraints.maxHeight),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _BoxOverlayPainter extends CustomPainter {
  final List<Polygon> polygons;
  final Size sourceImageSize;
  final Size canvasSize;

  _BoxOverlayPainter({
    required this.polygons,
    required this.sourceImageSize,
    required this.canvasSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final imageRect = _fitContainRect(canvasSize, sourceImageSize);
    if (imageRect.isEmpty) {
      return;
    }

    canvas.save();
    canvas.clipRect(imageRect);

    final shadowPaint = Paint()
      ..color = const Color(0xFF000000).withOpacity(0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final highlightPaint = Paint()
      ..color = const Color(0xFF00FF00)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    for (final polygon in polygons) {
      if (!polygon.isValid) continue;

      final path = _polygonToPath(polygon, imageRect, sourceImageSize);

      canvas.drawPath(path, shadowPaint);
      canvas.drawPath(path, highlightPaint);
    }

    canvas.restore();
  }

  Path _polygonToPath(
    Polygon polygon,
    Rect imageRect,
    Size imageSize,
  ) {
    final path = Path();

    for (var i = 0; i < polygon.points.length; i++) {
      final point = polygon.points[i];
      final x = imageRect.left + (point.x / imageSize.width) * imageRect.width;
      final y = imageRect.top + (point.y / imageSize.height) * imageRect.height;

      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    path.close();
    return path;
  }

  Rect _fitContainRect(Size canvasSize, Size imageSize) {
    if (canvasSize.width <= 0 || canvasSize.height <= 0) {
      return Rect.zero;
    }

    if (imageSize.width <= 0 || imageSize.height <= 0) {
      return Rect.zero;
    }

    final scale = min(
      canvasSize.width / imageSize.width,
      canvasSize.height / imageSize.height,
    );
    final fittedWidth = imageSize.width * scale;
    final fittedHeight = imageSize.height * scale;
    final left = (canvasSize.width - fittedWidth) / 2;
    final top = (canvasSize.height - fittedHeight) / 2;

    return Rect.fromLTWH(left, top, fittedWidth, fittedHeight);
  }

  @override
  bool shouldRepaint(covariant _BoxOverlayPainter oldDelegate) {
    return oldDelegate.polygons != polygons ||
        oldDelegate.sourceImageSize != sourceImageSize ||
        oldDelegate.canvasSize != canvasSize;
  }
}
