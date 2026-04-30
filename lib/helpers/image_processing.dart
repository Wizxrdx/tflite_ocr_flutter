import 'dart:math';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';

img.Image resizeLinearOpenCv(
    img.Image source, int targetWidth, int targetHeight) {
  if (source.width == targetWidth && source.height == targetHeight) {
    return img.Image.from(source);
  }

  const coefBits = 11;
  const coefScale = 1 << coefBits;

  final resized =
      img.Image(width: targetWidth, height: targetHeight, numChannels: 3);
  final scaleX = source.width / targetWidth;
  final scaleY = source.height / targetHeight;

  final xOffsets = List<int>.filled(targetWidth, 0);
  final xAlpha0 = List<int>.filled(targetWidth, 0);
  final xAlpha1 = List<int>.filled(targetWidth, 0);

  for (var dstX = 0; dstX < targetWidth; dstX++) {
    final srcX = (dstX + 0.5) * scaleX - 0.5;
    var x0 = srcX.floor();
    var wx = srcX - x0;

    if (x0 < 0) {
      x0 = 0;
      wx = 0.0;
    }

    if (x0 >= source.width - 1) {
      x0 = source.width - 1;
      wx = 0.0;
    }

    final alpha0 = ((1.0 - wx) * coefScale).round();
    final alpha1 = (wx * coefScale).round();

    xOffsets[dstX] = x0;
    xAlpha0[dstX] = alpha0;
    xAlpha1[dstX] = alpha1;
  }

  final yOffsets = List<int>.filled(targetHeight, 0);
  final yBeta0 = List<int>.filled(targetHeight, 0);
  final yBeta1 = List<int>.filled(targetHeight, 0);

  for (var dstY = 0; dstY < targetHeight; dstY++) {
    final srcY = (dstY + 0.5) * scaleY - 0.5;
    var y0 = srcY.floor();
    var wy = srcY - y0;

    if (y0 < 0) {
      y0 = 0;
      wy = 0.0;
    }

    if (y0 >= source.height - 1) {
      y0 = source.height - 1;
      wy = 0.0;
    }

    final beta0 = ((1.0 - wy) * coefScale).round();
    final beta1 = (wy * coefScale).round();

    yOffsets[dstY] = y0;
    yBeta0[dstY] = beta0;
    yBeta1[dstY] = beta1;
  }

  for (var dstY = 0; dstY < targetHeight; dstY++) {
    final y0 = yOffsets[dstY];
    final y1 = min(y0 + 1, source.height - 1);
    final beta0 = yBeta0[dstY];
    final beta1 = yBeta1[dstY];

    for (var dstX = 0; dstX < targetWidth; dstX++) {
      final x0 = xOffsets[dstX];
      final x1 = min(x0 + 1, source.width - 1);
      final alpha0 = xAlpha0[dstX];
      final alpha1 = xAlpha1[dstX];

      final p00 = source.getPixel(x0, y0);
      final p10 = source.getPixel(x1, y0);
      final p01 = source.getPixel(x0, y1);
      final p11 = source.getPixel(x1, y1);

      final row0R = p00.r.toInt() * alpha0 + p10.r.toInt() * alpha1;
      final row1R = p01.r.toInt() * alpha0 + p11.r.toInt() * alpha1;
      final row0G = p00.g.toInt() * alpha0 + p10.g.toInt() * alpha1;
      final row1G = p01.g.toInt() * alpha0 + p11.g.toInt() * alpha1;
      final row0B = p00.b.toInt() * alpha0 + p10.b.toInt() * alpha1;
      final row1B = p01.b.toInt() * alpha0 + p11.b.toInt() * alpha1;

      final r = ((((beta0 * (row0R >> 4)) >> 16) +
                  ((beta1 * (row1R >> 4)) >> 16) +
                  2) >>
              2)
          .clamp(0, 255);
      final g = ((((beta0 * (row0G >> 4)) >> 16) +
                  ((beta1 * (row1G >> 4)) >> 16) +
                  2) >>
              2)
          .clamp(0, 255);
      final b = ((((beta0 * (row0B >> 4)) >> 16) +
                  ((beta1 * (row1B >> 4)) >> 16) +
                  2) >>
              2)
          .clamp(0, 255);

      resized.setPixelRgb(dstX, dstY, r, g, b);
    }
  }

  return resized;
}

Future<Uint8List> drawBoxesOnImage(
    XFile imageFile, List<List<int>> boxes) async {
  final Uint8List uint8List = await imageFile.readAsBytes();
  return Isolate.run(() => _drawBoxesOnImageBytes(uint8List, boxes));
}

Future<Uint8List> drawPolygonsOnImage(
    XFile imageFile, List<List<List<double>>> polygons) async {
  final Uint8List uint8List = await imageFile.readAsBytes();
  return Isolate.run(() => _drawPolygonsOnImageBytes(uint8List, polygons));
}

Uint8List _drawBoxesOnImageBytes(Uint8List imageBytes, List<List<int>> boxes) {
  final img.Image sourceImage = img.decodeImage(imageBytes)!;
  final shadowColor = img.ColorRgba8(0, 0, 0, 200);
  final highlightColor = img.ColorRgba8(0, 255, 0, 255);

  for (final box in boxes) {
    final x1 = min(max(box[0], 0), sourceImage.width - 1);
    final y1 = min(max(box[1], 0), sourceImage.height - 1);
    final x2 = min(max(box[2], 0), sourceImage.width - 1);
    final y2 = min(max(box[3], 0), sourceImage.height - 1);

    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    img.drawRect(
      sourceImage,
      x1: x1,
      y1: y1,
      x2: x2,
      y2: y2,
      color: shadowColor,
      thickness: 7,
    );

    img.drawRect(
      sourceImage,
      x1: x1,
      y1: y1,
      x2: x2,
      y2: y2,
      color: highlightColor,
      thickness: 3,
    );
  }

  return Uint8List.fromList(img.encodeJpg(sourceImage));
}

Uint8List _drawPolygonsOnImageBytes(
    Uint8List imageBytes, List<List<List<double>>> polygons) {
  final img.Image sourceImage = img.decodeImage(imageBytes)!;
  final shadowColor = img.ColorRgba8(0, 0, 0, 200);
  final highlightColor = img.ColorRgba8(0, 255, 0, 255);

  for (final polygon in polygons) {
    if (polygon.length < 2) {
      continue;
    }

    for (var i = 0; i < polygon.length; i++) {
      final p1 = polygon[i];
      final p2 = polygon[(i + 1) % polygon.length];

      final x1 = min(max(p1[0].round(), 0), sourceImage.width - 1);
      final y1 = min(max(p1[1].round(), 0), sourceImage.height - 1);
      final x2 = min(max(p2[0].round(), 0), sourceImage.width - 1);
      final y2 = min(max(p2[1].round(), 0), sourceImage.height - 1);

      img.drawLine(
        sourceImage,
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        color: shadowColor,
        thickness: 7,
      );

      img.drawLine(
        sourceImage,
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        color: highlightColor,
        thickness: 3,
      );
    }
  }

  return Uint8List.fromList(img.encodeJpg(sourceImage));
}

Future<List<Uint8List>> extractImagesInsideBoundingBoxes(
    XFile imageFile, List<List<int>> boxes) async {
  final Uint8List uint8List = await imageFile.readAsBytes();
  final img.Image sourceImage = img.decodeImage(uint8List)!;
  final extractedImages = <Uint8List>[];

  for (final box in boxes) {
    final x1 = min(max(box[0], 0), sourceImage.width - 1);
    final y1 = min(max(box[1], 0), sourceImage.height - 1);
    final x2 = min(max(box[2], 0), sourceImage.width);
    final y2 = min(max(box[3], 0), sourceImage.height);

    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    final crop = img.copyCrop(
      sourceImage,
      x: x1,
      y: y1,
      width: x2 - x1,
      height: y2 - y1,
    );
    extractedImages.add(Uint8List.fromList(img.encodeJpg(crop)));
  }

  return extractedImages;
}
