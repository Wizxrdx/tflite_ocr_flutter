
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../models/result.dart';

/// A widget that displays an image with overlay boxes from text detection.
///
/// Accepts [ImageProvider], and a list of [LabeledBox] objects
/// containing detected text regions.
/// The polygons are overlaid on the image with proper coordinate scaling.
class ImageDisplayWidget extends StatefulWidget {
  final ImageProvider imageProvider;
  final Size? sourceImageSize;
  final OCRResult result;
  final BoxDecoration? decoration;
  final BoxFit boxFit;
  final Alignment alignment;

  const ImageDisplayWidget({
    super.key,
    required this.imageProvider,
    this.sourceImageSize,
    this.result = const OCRResult([]),
    this.decoration,
    this.boxFit = BoxFit.contain,
    this.alignment = Alignment.center,
  });

factory ImageDisplayWidget.fromRawOutput({
  required OCRResult result,
  Uint8List? imageBytes,
  ImageProvider? imageProvider,
  Size? sourceImageSize,
  BoxDecoration? decoration,
  BoxFit boxFit = BoxFit.contain,
  Alignment alignment = Alignment.center,
}) {
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
    result: result,
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
                    boxes: widget.result.boxes,
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
  final List<LabeledBox> boxes;
  final Size sourceImageSize;
  final Size canvasSize;

  _BoxOverlayPainter({
    required this.boxes,
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

    for (final box in boxes) {
      final path = _boxToPath(box, imageRect, sourceImageSize);

      canvas.drawPath(path, shadowPaint);
      canvas.drawPath(path, highlightPaint);

      final textSpan = TextSpan(
        text: box.label,
        style: TextStyle(
          color: Color(0xFF00FF00),
          fontSize: 12,
          fontWeight: FontWeight.bold,  
          backgroundColor: Color(0xAA000000),
        ),
      );

      final TextPainter textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      // Calculate the bottom-left X and Y of the bounding box
      final boxBottomLeftX = (imageRect.left + (box.x / sourceImageSize.width) * imageRect.width);
      final boxBottomLeftY = (imageRect.top + (box.y / sourceImageSize.height) * imageRect.height);
      final textY = max(boxBottomLeftY - textPainter.height - 2, 0.0);
      
      textPainter.paint(canvas, Offset(boxBottomLeftX, textY));
    }

    canvas.restore();
  }

  Path _boxToPath(
    LabeledBox box,
    Rect imageRect,
    Size imageSize,
  ) {
    final path = Path();

    for (var i = 0; i < 4; i++) {
      int x, y;
      switch (i) {
        case 0:
          x = box.x;
          y = box.y;
          break;
        case 1:
          x = box.x + box.width;
          y = box.y;
          break;
        case 2:
          x = box.x + box.width;
          y = box.y + box.height;
          break;
        case 3:
          x = box.x;
          y = box.y + box.height;
          break;
        default:
          x = box.x;
          y = box.y;
      }
      x = (imageRect.left + (x / imageSize.width) * imageRect.width).round();
      y = (imageRect.top + (y / imageSize.height) * imageRect.height).round();

      if (i == 0) {
        path.moveTo(x.toDouble(), y.toDouble());
      } else {
        path.lineTo(x.toDouble(), y.toDouble());
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
    return oldDelegate.boxes != boxes ||
        oldDelegate.sourceImageSize != sourceImageSize ||
        oldDelegate.canvasSize != canvasSize;
  }
}
