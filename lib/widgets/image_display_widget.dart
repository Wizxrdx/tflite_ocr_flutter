
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../models/text_detection.dart';

/// A widget that displays an image with overlay boxes from text detection.
///
/// Accepts [imageProvider], and a list of [polygons]
/// containing detected text regions.
/// The polygons are overlaid on the image with proper coordinate scaling.
class ImageDisplayWidget extends StatefulWidget {
  final ImageProvider imageProvider;
  final Size? sourceImageSize;
  final List<Polygon> polygons;
  final BoxDecoration? decoration;
  final BoxFit boxFit;
  final Alignment alignment;

  const ImageDisplayWidget({
    super.key,
    required this.imageProvider,
    this.sourceImageSize,
    this.polygons = const [],
    this.decoration,
    this.boxFit = BoxFit.contain,
    this.alignment = Alignment.center,
  });

factory ImageDisplayWidget.fromRawOutput({
  Uint8List? imageBytes,
  ImageProvider? imageProvider,
  Size? sourceImageSize,
  List<List<List<double>>> rawPolygons = const [],
  BoxDecoration? decoration,
  BoxFit boxFit = BoxFit.contain,
  Alignment alignment = Alignment.center,
}) {
  final result = TextDetectionResult.fromRawOutput(rawPolygons);

  final provider = imageBytes != null
      ? MemoryImage(imageBytes)
      : (imageProvider ?? const AssetImage('assets/wizardiusbewebicon.png'));

  // If caller supplied a size, use it. Otherwise:
  // - bytes => leave null so the widget will decode the bytes and set the size
  // - provider => set Size.zero
  final resolvedSize = sourceImageSize ?? (imageBytes != null ? null : Size.zero);

  return ImageDisplayWidget(
    imageProvider: provider,
    sourceImageSize: resolvedSize,
    polygons: result.polygons,
    decoration: decoration,
    boxFit: boxFit,
    alignment: alignment,
  );
}

  @override
  State<ImageDisplayWidget> createState() => _ImageDisplayWidgetState();
}

class _ImageDisplayWidgetState extends State<ImageDisplayWidget> {
  late ImageProvider _provider;
  Size? _resolvedSize;
  ImageStream? _stream;
  ImageStreamListener? _listener;

  @override
  void initState() {
    super.initState();

    _provider = widget.imageProvider;
    _resolvedSize = widget.sourceImageSize;
    if (_resolvedSize == null) _resolveProviderForSize(_provider);
  }

  @override
  void didUpdateWidget(covariant ImageDisplayWidget oldWidget) {
    super.didUpdateWidget(oldWidget);

    if (
      widget.imageProvider != oldWidget.imageProvider ||
      widget.sourceImageSize != oldWidget.sourceImageSize
    ) {
      _stream?.removeListener(_listener!);
      _resolvedSize = widget.sourceImageSize;
      _provider = widget.imageProvider;
      if (_resolvedSize == null) _resolveProviderForSize(_provider);
      setState(() {});
    }
  }

  @override
  void dispose() {
    if (_stream != null && _listener != null) {
      _stream!.removeListener(_listener!);
    }
    super.dispose();
  }

  void _resolveProviderForSize(ImageProvider provider) {
    final stream = provider.resolve(const ImageConfiguration());
    _stream = stream;
    _listener = ImageStreamListener((ImageInfo info, bool _) {
      if (!mounted) return;
      setState(() {
        _resolvedSize = Size(
          info.image.width.toDouble(),
          info.image.height.toDouble(),
        );
      });

      stream.removeListener(_listener!);
    }, onError: (_, __) {
      stream.removeListener(_listener!);
    });
    stream.addListener(_listener!);
  }

  @override
  Widget build(BuildContext context) {
    final image = Image(
      image: _provider,
      fit: widget.boxFit,
      alignment: widget.alignment,
    );
    
    return Container(
      decoration: widget.decoration ??
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
              if (_resolvedSize != null)
                CustomPaint(
                  painter: _BoxOverlayPainter(
                    polygons: widget.polygons,
                    sourceImageSize: _resolvedSize!,
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
      ..color = const Color(0xFF000000).withValues(alpha: 0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final highlightPaint = Paint()
      ..color = const Color(0xFF00FF00)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;

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
