import 'dart:typed_data';
import 'dart:math';
import 'dart:io';

import 'package:image/image.dart' as img;

import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_text_extraction/models/result.dart';

class TextDetection {
  static const _modelPath = "assets/craft-text-detector-fp16.tflite";
  late Interpreter _interpreter;
  late Tensor _inputTensor;
  bool _isInitialized = false;

  int get interpreterAddress => _interpreter.address;

  Future<void> init({int? address}) async {
    final options = InterpreterOptions()
    ..threads = max(1, min(4, Platform.numberOfProcessors));

    if (address != null) {
      _interpreter = Interpreter.fromAddress(address);
    } else {
      _interpreter = await Interpreter.fromAsset(_modelPath, options: options);
    }
    _inputTensor = _interpreter.getInputTensors().first;
    _isInitialized = true;
  }

  Future<void> close() async {
    if (!_isInitialized) {
      return;
    }

    _interpreter.close();
    _isInitialized = false;
  }

  void _verifyModelInputShape() {
    final inputShape = _inputTensor.shape;

    if (inputShape.length != 4) {
      throw StateError('Expected 4D input tensor [N,C,H,W], got: $inputShape');
    }

    print('Model input shape: $inputShape');
    final expectedChannels = inputShape[1];

    if (expectedChannels != 3) {
      throw StateError(
          'Unsupported input channel count: $expectedChannels. This pipeline expects 3 channels (RGB).');
    }
  }

  List<dynamic> _transposeNchwToNhwc(List<dynamic> input, List<int> shape) {
    final batch = shape[0];
    final channels = shape[1];
    final height = shape[2];
    final width = shape[3];

    return List.generate(
      batch,
      (n) => List.generate(
        height,
        (y) => List.generate(
          width,
          (x) => List.generate(
            channels,
            (c) => (((input[n] as List)[c] as List)[y] as List)[x] as double,
          ),
        ),
      ),
    );
  }

  List<List<List<double>>> _postprocess(
    List<int> rawScoreShape,
    List<dynamic> rawScoreMap,
    int resizedWidth,
    int resizedHeight,
    int originalWidth,
    int originalHeight,
    double scale
  ) {
    late final List<dynamic> scoresRaw;
    late final List<int> scoreShape;
    if (rawScoreShape[3] == 2) {
      scoresRaw = rawScoreMap;
      scoreShape = rawScoreShape;
    } else if (rawScoreShape[1] == 2) {
      scoresRaw = _transposeNchwToNhwc(rawScoreMap, rawScoreShape);
      scoreShape = [
        rawScoreShape[0],
        rawScoreShape[2],
        rawScoreShape[3],
        rawScoreShape[1],
      ];
    } else {
      throw StateError(
          'Could not find CRAFT score map with 2 channels. Shape: $rawScoreShape');
    }

    final scoreHeight = scoreShape[1];
    final scoreWidth = scoreShape[2];
    final scoreChannels = scoreShape[3];

    if (scoreChannels < 2) {
      throw StateError(
          'Expected a score tensor with at least 2 channels, got: $scoreShape');
    }

    // Extract textmap and linkmap from raw scores
    final textmap = Float32List(scoreHeight * scoreWidth);
    final linkmap = Float32List(scoreHeight * scoreWidth);
    final combinedMap = Uint8List(scoreHeight * scoreWidth);

    double maxTextValue = 0.0;
    double maxLinkValue = 0.0;

    for (var y = 0; y < scoreHeight; y++) {
      for (var x = 0; x < scoreWidth; x++) {
        int i = y * scoreWidth + x; 
        
        textmap[i] = scoresRaw[0][y][x][0] as double;
        linkmap[i] = scoresRaw[0][y][x][1] as double;

        if (textmap[i] > maxTextValue) maxTextValue = textmap[i];
        if (linkmap[i] > maxLinkValue) maxLinkValue = linkmap[i];
      }
    }

    print('CRAFT Max Text Value: $maxTextValue, Max Link Value: $maxLinkValue');

    // Dynamically adjust thresholds based on the max confidence.
    // If the image is shrunk, confidence drops, so we lower the threshold.
    final double actualTextThreshold = maxTextValue * 0.7; // Very strict core
    final double actualLinkThreshold = maxLinkValue * 0.9; // Only keep the absolute strongest links
    final double actualDetectionThreshold = maxTextValue * 0.4;

    for (var i = 0; i < scoreHeight * scoreWidth; i++) {
        int textBit = textmap[i] > actualTextThreshold ? 1 : 0;
        int linkBit = linkmap[i] > actualLinkThreshold ? 1 : 0;
        combinedMap[i] = min(1, textBit + linkBit);
    }

    // Connected components on combined score map
    final componentInfo = _connectedComponentsWithStats(combinedMap, textmap, scoreHeight, scoreWidth);

    final boxes = <List<List<double>>>[];

    for (final component in componentInfo) {
      final size = component['size'] as int;
      final maxTextValue = component['maxTextValue'] as double;
      final bounds = component['bounds'] as Map<String, int>;
      if (maxTextValue < actualDetectionThreshold) {
        continue;
      }

      final componentPixels = component['pixels'] as List<int>;
      final segmap = Uint8List(scoreHeight * scoreWidth);

      // Build component segmap while removing link-only pixels.
      for (final pixel in componentPixels) {
        final x = pixel % scoreWidth;
        final y = pixel ~/ scoreWidth;
        int i = y * scoreWidth + x;

        if (linkmap[i] == 1 && textmap[i] == 0) {
          continue;
        }
        segmap[i] = 255;
      }

      // Dilate the segmentation map
      final dilatedSegmap = _dilateSegmap(segmap, scoreHeight, scoreWidth, size, bounds);

      // Extract bounding box from dilated segmap
      final box =
          _getRotatedBoundingBox(dilatedSegmap, scoreHeight, scoreWidth);
      if (box != null) {
        boxes.add(box);
      }
    }

    // Scale boxes back to original size
    final finalPolygons = _scalePolygons(
      boxes,
      scoreWidth,
      scoreHeight,
      resizedWidth,
      resizedHeight,
      originalWidth,
      originalHeight,
      scale
    );
    return finalPolygons;
  }

  List<Map<String, dynamic>> _connectedComponentsWithStats(
      List<int> combinedScore,
      List<double> textmap,
      int height,
      int width) {
    final visited = List<bool>.filled(height * width, false);
    final componentInfo = <Map<String, dynamic>>[];

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        int i = y * width + x;
        if (visited[i] || combinedScore[i] == 0) {
          continue;
        }

        // BFS to find component
        final queue = <int>[i];
        var queueHead = 0;
        visited[i] = true;

        var minX = x;
        var minY = y;
        var maxX = x;
        var maxY = y;
        var size = 0;
        var maxTextValue = 0.0;
        final pixels = <int>[];

        while (queueHead < queue.length) {
          final current = queue[queueHead++];
          final cx = current % width;
          final cy = current ~/ width;
          size++;
          pixels.add(current);

          // Track max text value for this component (from textmap, not combined)
          maxTextValue = max(maxTextValue, textmap[current]);

          if (cx < minX) minX = cx;
          if (cy < minY) minY = cy;
          if (cx > maxX) maxX = cx;
          if (cy > maxY) maxY = cy;

          // Check 4-connectivity
          for (final neighbor in const [
            [0, -1],
            [-1, 0],
            [1, 0],
            [0, 1],
          ]) {
            final nx = cx + neighbor[0];
            final ny = cy + neighbor[1];

            if (nx < 0 ||
                nx >= width ||
                ny < 0 ||
                ny >= height ||
                visited[ny * width + nx]) {
              continue;
            }

             int ni = ny * width + nx;
            if (combinedScore[ni] == 1) {
              visited[ni] = true;
              queue.add(ni);
            }
          }
        }

        componentInfo.add({
          'size': size,
          'maxTextValue': maxTextValue,
          'bounds': {
            'left': minX,
            'top': minY,
            'right': maxX,
            'bottom': maxY,
            'width': maxX - minX + 1,
            'height': maxY - minY + 1,
          },
          'pixels': pixels,
        });
      }
    }

    return componentInfo;
  }

  Uint8List _dilateSegmap(Uint8List segmap, int height, int width,
      int size, Map<String, int> bounds) {
    final w = bounds['width'] as int;
    final h = bounds['height'] as int;

    // Because the image is shrunk, boxes are already close.
    // Use an extremely small dilation multiplier to prevent overlapping.
    final niter = (sqrt(size * min(w, h) / (w * h)) * 3.0).toInt();

    final sx = max(bounds['left']! - niter, 0);
    final sy = max(bounds['top']! - niter, 0);
    final ex = min(bounds['right']! + niter + 1, width);
    final ey = min(bounds['bottom']! + niter + 1, height);

    final kernelSize = 1 + niter;
    final kernelAnchor = kernelSize ~/ 2;

    // Apply morphological dilation to the region
    final dilated = Uint8List.fromList(segmap);

    for (var yy = sy; yy < ey; yy++) {
      for (var xx = sx; xx < ex; xx++) {
        bool found = false;
        for (var ky = 0; ky < kernelSize && !found; ky++) {
          for (var kx = 0; kx < kernelSize && !found; kx++) {
            final ny = yy - kernelAnchor + ky;
            final nx = xx - kernelAnchor + kx;
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
              int ni = ny * width + nx;
              if (segmap[ni] > 0) {
                found = true;
              }
            }
          }
        }
        if (found) {
          int dilatedIndex = yy * width + xx;
          dilated[dilatedIndex] = 255;
        }
      }
    }

    return dilated;
  }

  double _cross(List<double> o, List<double> a, List<double> b) {
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  }

  List<List<double>> _convexHull(List<List<double>> points) {
    final sorted = List<List<double>>.from(points)
      ..sort((p1, p2) {
        final xCmp = p1[0].compareTo(p2[0]);
        if (xCmp != 0) return xCmp;
        return p1[1].compareTo(p2[1]);
      });

    if (sorted.length <= 1) {
      return sorted;
    }

    final lower = <List<double>>[];
    for (final p in sorted) {
      while (lower.length >= 2 &&
          _cross(lower[lower.length - 2], lower.last, p) <= 0) {
        lower.removeLast();
      }
      lower.add(p);
    }

    final upper = <List<double>>[];
    for (var i = sorted.length - 1; i >= 0; i--) {
      final p = sorted[i];
      while (upper.length >= 2 &&
          _cross(upper[upper.length - 2], upper.last, p) <= 0) {
        upper.removeLast();
      }
      upper.add(p);
    }

    lower.removeLast();
    upper.removeLast();
    return [...lower, ...upper];
  }

  List<List<double>> _minimumAreaRectangle(List<List<double>> points) {
    final hull = _convexHull(points);

    if (hull.isEmpty) {
      return const [];
    }

    if (hull.length == 1) {
      final p = hull.first;
      return [
        [p[0], p[1]],
        [p[0], p[1]],
        [p[0], p[1]],
        [p[0], p[1]],
      ];
    }

    var bestArea = double.infinity;
    var bestTheta = 0.0;
    var bestMinX = 0.0;
    var bestMaxX = 0.0;
    var bestMinY = 0.0;
    var bestMaxY = 0.0;

    for (var i = 0; i < hull.length; i++) {
      final p0 = hull[i];
      final p1 = hull[(i + 1) % hull.length];
      final dx = p1[0] - p0[0];
      final dy = p1[1] - p0[1];

      if (dx == 0.0 && dy == 0.0) {
        continue;
      }

      final theta = atan2(dy, dx);
      final cosT = cos(theta);
      final sinT = sin(theta);

      var minX = double.infinity;
      var maxX = double.negativeInfinity;
      var minY = double.infinity;
      var maxY = double.negativeInfinity;

      for (final p in hull) {
        // Rotate point by -theta.
        final rx = p[0] * cosT + p[1] * sinT;
        final ry = -p[0] * sinT + p[1] * cosT;

        if (rx < minX) minX = rx;
        if (rx > maxX) maxX = rx;
        if (ry < minY) minY = ry;
        if (ry > maxY) maxY = ry;
      }

      final area = (maxX - minX) * (maxY - minY);
      if (area < bestArea) {
        bestArea = area;
        bestTheta = theta;
        bestMinX = minX;
        bestMaxX = maxX;
        bestMinY = minY;
        bestMaxY = maxY;
      }
    }

    final cosT = cos(bestTheta);
    final sinT = sin(bestTheta);

    List<double> toOriginal(double rx, double ry) {
      // Inverse rotation by +theta.
      final x = rx * cosT - ry * sinT;
      final y = rx * sinT + ry * cosT;
      return [x, y];
    }

    return [
      toOriginal(bestMinX, bestMinY),
      toOriginal(bestMaxX, bestMinY),
      toOriginal(bestMaxX, bestMaxY),
      toOriginal(bestMinX, bestMaxY),
    ];
  }

  double _distance(List<double> a, List<double> b) {
    final dx = a[0] - b[0];
    final dy = a[1] - b[1];
    return sqrt(dx * dx + dy * dy);
  }

  List<List<double>> _rollStartAtMinSum(List<List<double>> box) {
    var startIdx = 0;
    var minSum = double.infinity;
    for (var i = 0; i < box.length; i++) {
      final sum = box[i][0] + box[i][1];
      if (sum < minSum) {
        minSum = sum;
        startIdx = i;
      }
    }

    return [
      ...box.sublist(startIdx),
      ...box.sublist(0, startIdx),
    ];
  }

  List<List<double>>? _getRotatedBoundingBox(
      Uint8List segmap, int height, int width) {
    // Find contour points as done in the Python CRAFT utility pipeline.
    final points = <List<double>>[];
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        int i = y * width + x;
        if (segmap[i] > 0) {
          points.add([x.toDouble(), y.toDouble()]);
        }
      }
    }

    if (points.isEmpty) {
      return null;
    }

    var box = _minimumAreaRectangle(points);
    if (box.isEmpty) {
      return null;
    }

    // Match CRAFT's near-square fallback to an axis-aligned rectangle.
    final w = _distance(box[0], box[1]);
    final h = _distance(box[1], box[2]);
    final ratio = max(w, h) / (min(w, h) + 1e-5);
    if ((1.0 - ratio).abs() <= 0.1) {
      var minX = points.first[0];
      var minY = points.first[1];
      var maxX = points.first[0];
      var maxY = points.first[1];

      for (final p in points) {
        if (p[0] < minX) minX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] > maxY) maxY = p[1];
      }

      box = [
        [minX, minY],
        [maxX, minY],
        [maxX, maxY],
        [minX, maxY],
      ];
    }

    return _rollStartAtMinSum(box);
  }

  List<List<List<double>>> _scalePolygons(
      List<List<List<double>>> boxes,
      int scoreWidth,
      int scoreHeight,
      int targetWidth,
      int targetHeight,
      int originalWidth,
      int originalHeight,
      double scale
      ) {
    final scoreToInputRatio = targetWidth / scoreWidth;

    final result = <List<List<double>>>[];

    for (var i = 0; i < boxes.length; i++) {
      final box = boxes[i];
      final scaled = <List<double>>[];

      for (final point in box) {
        final x = (point[0] * scoreToInputRatio) / scale;
        final y = (point[1] * scoreToInputRatio) / scale;
        final clampedX = x.clamp(0.0, originalWidth.toDouble()).toDouble();
        final clampedY = y.clamp(0.0, originalHeight.toDouble()).toDouble();
        scaled.add([clampedX, clampedY]);
      }

      if (scaled.length == 4) {
        result.add(scaled);
      }
    }

    return result;
  }

  List<List<int>> _polygonsToAxisAlignedBoxes(List<List<List<double>>> polygons,
      int originalWidth, int originalHeight) {
    final result = <List<int>>[];

    for (final polygon in polygons) {
      if (polygon.isEmpty) {
        continue;
      }

      var minX = double.maxFinite;
      var minY = double.maxFinite;
      var maxX = double.negativeInfinity;
      var maxY = double.negativeInfinity;

      for (final point in polygon) {
        minX = min(minX, point[0]);
        minY = min(minY, point[1]);
        maxX = max(maxX, point[0]);
        maxY = max(maxY, point[1]);
      }

      if (minX > maxX || minY > maxY) {
        continue;
      }

      final box = [
        max(minX.floor(), 0),
        max(minY.floor(), 0),
        min(maxX.ceil(), originalWidth),
        min(maxY.ceil(), originalHeight),
      ];

      if (box[2] > box[0] && box[3] > box[1]) {
        result.add(box);
      }
    }

    return result;
  }

  Future<Map<String, dynamic>> _runInference(List<dynamic> inputTensor) async {
    // We already have _interpreter initialized in this isolate!
    // Creating a new Interpreter from address causes its internal shape cache to desync.
    final outputTensors = _interpreter.getOutputTensors();

    List<int>? rawScoreShape;
    var scoreTensorIndex = -1;
    for (var i = 0; i < outputTensors.length; i++) {
      final shape = outputTensors[i].shape;
      if (shape.length != 4) {
        throw StateError(
            'Expected 4D output tensor for CRAFT, got shape: $shape');
      }

      if (scoreTensorIndex < 0 && (shape[3] == 2 || shape[1] == 2)) {
        scoreTensorIndex = i;
        rawScoreShape = shape;
      }
    }

    if (scoreTensorIndex < 0 || rawScoreShape == null) {
      final shapes = outputTensors.map((t) => t.shape.toString()).toList();
      throw StateError(
          'Could not find CRAFT score map with 2 channels. Outputs: $shapes');
    }
    
    _interpreter.runInference([inputTensor]);

    final rawScoreMap = List.generate(
      rawScoreShape[0],
      (_) => List.generate(
        rawScoreShape![1],
        (_) => List.generate(
          rawScoreShape![2],
          (_) => List<double>.filled(rawScoreShape![3], 0.0),
        ),
      ),
    );

    outputTensors[scoreTensorIndex].copyTo(rawScoreMap);
    return {
      'rawScoreMap': rawScoreMap,
      'rawScoreShape': rawScoreShape,
    };
  }

  Future<Map<String, dynamic>> _preprocess(img.Image decodedImage) async {
    final targetWidth = _inputTensor.shape[3];
    final targetHeight = _inputTensor.shape[2];

    final originalWidth = decodedImage.width;
    final originalHeight = decodedImage.height;

    final scale = min(targetWidth / originalWidth, targetHeight / originalHeight);
    final newWidth = (originalWidth * scale).toInt();
    final newHeight = (originalHeight * scale).toInt();

    final resizedImage = img.copyResize(decodedImage, width: newWidth, height: newHeight);

    final textmap = Float32List(targetHeight * targetWidth * 3);

    const meanR = 123.675; 
    const meanG = 116.28; 
    const meanB = 103.53; 
    const stdR = 58.395; 
    const stdG = 57.12; 
    const stdB = 57.375;

    for (var y = 0; y < targetHeight; y++) {
      for (var x = 0; x < targetWidth; x++) {
        final area = targetWidth * targetHeight;
        
        // Pad with 0 for normalized input (black padding)
        if (x < newWidth && y < newHeight) {
            final pixel = resizedImage.getPixel(x, y);
            textmap[0 * area + y * targetWidth + x] = (pixel.r - meanR) / stdR;
            textmap[1 * area + y * targetWidth + x] = (pixel.g - meanG) / stdG;
            textmap[2 * area + y * targetWidth + x] = (pixel.b - meanB) / stdB;
        } else {
            textmap[0 * area + y * targetWidth + x] = -meanR / stdR;
            textmap[1 * area + y * targetWidth + x] = -meanG / stdG;
            textmap[2 * area + y * targetWidth + x] = -meanB / stdB;
        }
      }
    }

    return {
      'inputTensor': textmap.reshape([1, 3, targetHeight, targetWidth]),
      'scale': scale,
      'targetHeight': targetHeight,
      'targetWidth': targetWidth,
      'originalWidth': originalWidth,
      'originalHeight': originalHeight,
    };
  }

  Future<List<LabeledBox>> detectBoxes(img.Image decodedImage) async {
    final totalTimer = Stopwatch()..start();
    _verifyModelInputShape();

    final preprocessTimer = Stopwatch()..start();
    final preprocessedData = await _preprocess(
      decodedImage,
    );
    final inputTensor = preprocessedData['inputTensor'] as List<dynamic>;
    final scale = preprocessedData['scale'] as double;
    final targetHeight = preprocessedData['targetHeight'] as int;
    final targetWidth = preprocessedData['targetWidth'] as int;
    final originalWidth = preprocessedData['originalWidth'] as int;
    final originalHeight = preprocessedData['originalHeight'] as int;
    preprocessTimer.stop();

    final inferenceTimer = Stopwatch()..start();
    final inferenceResult = await _runInference(inputTensor);
    inferenceTimer.stop();

    final rawScoreShape = inferenceResult['rawScoreShape'] as List<int>;
    final rawScoreMap = inferenceResult['rawScoreMap'] as List<dynamic>;

    final postprocessTimer = Stopwatch()..start();
    final polygons = _postprocess(
      rawScoreShape,
      rawScoreMap,
      targetWidth,
      targetHeight,
      originalWidth,
      originalHeight,
      scale
    );

    postprocessTimer.stop();
    totalTimer.stop();

    print('======= Text Detection Timing =======');
    print('Preprocessing time: ${preprocessTimer.elapsedMilliseconds} ms');
    print('Inference time: ${inferenceTimer.elapsedMilliseconds} ms');
    print('Postprocessing time: ${postprocessTimer.elapsedMilliseconds} ms');
    print('Total text detection time: ${totalTimer.elapsedMilliseconds} ms');

    final boxes = _polygonsToAxisAlignedBoxes(polygons, originalWidth, originalHeight);
    return boxes.map((box) {
      final x = box[0];
      final y = box[1];
      final width = box[2] - box[0];
      final height = box[3] - box[1];
      return LabeledBox(x, y, width, height);
    }).toList();
  }
}
