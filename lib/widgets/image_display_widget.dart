
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
    if (imageRect.isEmpty) return;

    canvas.save();
    canvas.clipRect(imageRect);

    final blockPaint = Paint()
      ..color = const Color(0xD9FFFFFF) // 85% opacity white
      ..style = PaintingStyle.fill;

    for (final box in boxes) {
      // Calculate the physical dimensions of the box on the canvas
      final rectLeft = imageRect.left + (box.x / sourceImageSize.width) * imageRect.width;
      final rectTop = imageRect.top + (box.y / sourceImageSize.height) * imageRect.height;
      final rectWidth = (box.width / sourceImageSize.width) * imageRect.width;
      final rectHeight = (box.height / sourceImageSize.height) * imageRect.height;

      // Create a premium rounded rectangle
      final RRect roundedRect = RRect.fromRectAndRadius(
        Rect.fromLTWH(rectLeft, rectTop, rectWidth, rectHeight),
        const Radius.circular(4.0), // Subtle rounded corners
      );

      // Draw the solid block to hide the original text
      canvas.drawRRect(roundedRect, blockPaint);

      // Draw the recognized text inside the block
      if (box.label != null && box.label!.isNotEmpty) {
        final textSpan = TextSpan(
          text: box.label,
          style: const TextStyle(
            color: Colors.black87, // Dark text on white background
            fontWeight: FontWeight.w600,
            fontFamily: 'Roboto', // Modern sleek font
          ),
        );

        final textPainter = TextPainter(
          text: textSpan,
          textDirection: TextDirection.ltr,
          maxLines: 1,
        );

        // Layout first with infinite constraints to get the text's natural size
        textPainter.layout();

        // Calculate scaling factor so the text fits perfectly inside the box padding
        final scaleX = (rectWidth * 0.9) / textPainter.width;
        final scaleY = (rectHeight * 0.8) / textPainter.height;
        final scale = min(scaleX, scaleY);

        canvas.save();
        
        // Move canvas origin to the center of the bounding box
        canvas.translate(rectLeft + rectWidth / 2, rectTop + rectHeight / 2);
        canvas.scale(scale); // Scale the text perfectly

        // Draw the text exactly centered
        textPainter.paint(
          canvas,
          Offset(-textPainter.width / 2, -textPainter.height / 2),
        );

        canvas.restore();
      }
    }

    canvas.restore();
  }

  Rect _fitContainRect(Size canvasSize, Size imageSize) {
    if (canvasSize.width <= 0 || canvasSize.height <= 0) return Rect.zero;
    if (imageSize.width <= 0 || imageSize.height <= 0) return Rect.zero;

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