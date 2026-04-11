import 'dart:math';
import 'dart:isolate';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:tflite_text_extraction/helpers/image_processing.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';

Map<String, dynamic> _preprocessImageForCraft(
  Uint8List imageBytes,
  int targetWidth,
  int targetHeight,
) {
  final decodedImage = img.decodeImage(imageBytes);
  if (decodedImage == null) {
    throw StateError('Failed to decode input image.');
  }

  final originalWidth = decodedImage.width;
  final originalHeight = decodedImage.height;

  final resizedImage =
      resizeLinearOpenCv(decodedImage, targetWidth, targetHeight);

  const meanR = 123.675; // 0.485 * 255
  const meanG = 116.28; // 0.456 * 255
  const meanB = 103.53; // 0.406 * 255
  const stdR = 58.395; // 0.229 * 255
  const stdG = 57.12; // 0.224 * 255
  const stdB = 57.375; // 0.225 * 255

  final height = resizedImage.height;
  final width = resizedImage.width;
  final channels = List.generate(
    3,
    (_) => List.generate(
      height,
      (_) => List<double>.filled(width, 0.0),
    ),
  );

  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final pixel = resizedImage.getPixel(x, y);
      channels[0][y][x] = (pixel.r - meanR) / stdR;
      channels[1][y][x] = (pixel.g - meanG) / stdG;
      channels[2][y][x] = (pixel.b - meanB) / stdB;
    }
  }

  return {
    'inputTensor': [channels],
    'resizedWidth': width,
    'resizedHeight': height,
    'originalWidth': originalWidth,
    'originalHeight': originalHeight,
  };
}

List _create4DTensorBufferForShape(List<int> shape) {
  return List.generate(
    shape[0],
    (_) => List.generate(
      shape[1],
      (_) => List.generate(
        shape[2],
        (_) => List<double>.filled(shape[3], 0.0),
      ),
    ),
  );
}

Map<String, dynamic> _runCraftPreprocessAndInference(
  Uint8List imageBytes,
  int interpreterAddress,
  int targetWidth,
  int targetHeight,
) {
  final preprocessTimer = Stopwatch()..start();
  final preprocessed =
      _preprocessImageForCraft(imageBytes, targetWidth, targetHeight);
  preprocessTimer.stop();

  final inputTensor = preprocessed['inputTensor'] as List;
  final interpreter = Interpreter.fromAddress(interpreterAddress);
  final outputTensors = interpreter.getOutputTensors();

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

  final inferenceTimer = Stopwatch()..start();
  interpreter.runInference([inputTensor]);
  inferenceTimer.stop();

  final rawScoreMap = _create4DTensorBufferForShape(rawScoreShape);
  outputTensors[scoreTensorIndex].copyTo(rawScoreMap);

  return {
    'rawScoreMap': rawScoreMap,
    'rawScoreShape': rawScoreShape,
    'resizedWidth': preprocessed['resizedWidth'],
    'resizedHeight': preprocessed['resizedHeight'],
    'originalWidth': preprocessed['originalWidth'],
    'originalHeight': preprocessed['originalHeight'],
    'preprocessMs': preprocessTimer.elapsedMicroseconds / 1000.0,
    'inferenceMs': inferenceTimer.elapsedMicroseconds / 1000.0,
  };
}

class TextDetection {
  static const _modelPath = "assets/craft-text-detector-fp16.tflite";
  late Interpreter _interpreter;
  late Tensor _inputTensor;
  IsolateInterpreter? _isolateInterpreter;
  bool _isInitialized = false;

  Future<void> init() async {
    await _loadModel();
  }

  Future<void> _loadModel() async {
    final options = InterpreterOptions()
      ..threads = max(1, min(4, Platform.numberOfProcessors));
    // Load model from assets
    _interpreter = await Interpreter.fromAsset(_modelPath, options: options);
    _inputTensor = _interpreter.getInputTensors().first;
    _isolateInterpreter = await IsolateInterpreter.create(
      address: _interpreter.address,
      debugName: 'TextDetectionIsolate',
    );
    _isInitialized = true;
  }

  Future<void> close() async {
    if (!_isInitialized) {
      return;
    }

    await _isolateInterpreter?.close();
    _interpreter.close();
    _isolateInterpreter = null;
    _isInitialized = false;
  }

  void _verifyModelInputShape() {
    final inputShape = _inputTensor.shape;

    if (inputShape.length != 4) {
      throw StateError('Expected 4D input tensor [N,C,H,W], got: $inputShape');
    }

    final expectedChannels = inputShape[1];

    if (expectedChannels != 3) {
      throw StateError(
          'Unsupported input channel count: $expectedChannels. This pipeline expects 3 channels (RGB).');
    }
  }

  String _formatMs(Stopwatch stopwatch) {
    return (stopwatch.elapsedMicroseconds / 1000.0).toStringAsFixed(1);
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
      List scoresRaw,
      int scoreHeight,
      int scoreWidth,
      int resizedWidth,
      int resizedHeight,
      int originalWidth,
      int originalHeight) {
    // Original CRAFT thresholds (Python reference)
    const double detectionThreshold = 0.7;
    const double textThreshold = 0.4;
    const double linkThreshold = 0.2;
    const int sizeThreshold = 10;

    // Extract textmap and linkmap from raw scores
    final textmap =
        List.generate(scoreHeight, (_) => List<double>.filled(scoreWidth, 0.0));
    final linkmap =
        List.generate(scoreHeight, (_) => List<double>.filled(scoreWidth, 0.0));

    for (var y = 0; y < scoreHeight; y++) {
      for (var x = 0; x < scoreWidth; x++) {
        textmap[y][x] = (scoresRaw[0][y][x][0] as double);
        linkmap[y][x] = (scoresRaw[0][y][x][1] as double);
      }
    }

    // Binarize maps using thresholds
    final textScore =
        List.generate(scoreHeight, (_) => List<int>.filled(scoreWidth, 0));
    final linkScore =
        List.generate(scoreHeight, (_) => List<int>.filled(scoreWidth, 0));

    for (var y = 0; y < scoreHeight; y++) {
      for (var x = 0; x < scoreWidth; x++) {
        textScore[y][x] = textmap[y][x] > textThreshold ? 1 : 0;
        linkScore[y][x] = linkmap[y][x] > linkThreshold ? 1 : 0;
      }
    }

    // CRITICAL: Find components on COMBINED text+link map (Python does this!)
    final combinedScore =
        List.generate(scoreHeight, (_) => List<int>.filled(scoreWidth, 0));
    for (var y = 0; y < scoreHeight; y++) {
      for (var x = 0; x < scoreWidth; x++) {
        combinedScore[y][x] =
            min(1, textScore[y][x] + linkScore[y][x]); // Combine
      }
    }

    // Connected components on combined score map
    final componentInfo = _connectedComponentsWithStats(
        combinedScore, textmap, scoreHeight, scoreWidth);

    final boxes = <List<List<double>>>[];

    for (final component in componentInfo) {
      final size = component['size'] as int;
      final maxTextValue = component['maxTextValue'] as double;
      final bounds = component['bounds'] as Map<String, int>;

      if (size < sizeThreshold) {
        continue;
      }

      if (maxTextValue < detectionThreshold) {
        continue;
      }

      final componentPixels = component['pixels'] as List<int>;
      final segmap =
          List.generate(scoreHeight, (_) => List<int>.filled(scoreWidth, 0));

      // Build component segmap while removing link-only pixels.
      for (final pixel in componentPixels) {
        final x = pixel % scoreWidth;
        final y = pixel ~/ scoreWidth;
        if (linkScore[y][x] == 1 && textScore[y][x] == 1) {
          continue;
        }
        segmap[y][x] = 255;
      }

      // Dilate the segmentation map
      final dilatedSegmap =
          _dilateSegmap(segmap, scoreHeight, scoreWidth, size, bounds);

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
    );

    return finalPolygons;
  }

  List<Map<String, dynamic>> _connectedComponentsWithStats(
      List<List<int>> combinedScore,
      List<List<double>> textmap,
      int height,
      int width) {
    final visited =
        List.generate(height, (_) => List<bool>.filled(width, false));
    final componentInfo = <Map<String, dynamic>>[];

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        if (visited[y][x] || combinedScore[y][x] == 0) {
          continue;
        }

        // BFS to find component
        final queue = <int>[y * width + x];
        var queueHead = 0;
        visited[y][x] = true;

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
          maxTextValue = max(maxTextValue, textmap[cy][cx]);

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
                visited[ny][nx]) {
              continue;
            }

            if (combinedScore[ny][nx] == 1) {
              visited[ny][nx] = true;
              queue.add(ny * width + nx);
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

  List<List<int>> _dilateSegmap(List<List<int>> segmap, int height, int width,
      int size, Map<String, int> bounds) {
    final w = bounds['width'] as int;
    final h = bounds['height'] as int;

    final niter = (sqrt(size * min(w, h) / (w * h)) * 2.0).toInt();

    final sx = max(bounds['left']! - niter, 0);
    final sy = max(bounds['top']! - niter, 0);
    final ex = min(bounds['right']! + niter + 1, width);
    final ey = min(bounds['bottom']! + niter + 1, height);

    // Create structuring element (kernel)
    final kernelSize = 1 + niter;
    final kernelAnchor = kernelSize ~/ 2;

    // Apply morphological dilation to the region
    final dilated = List.generate(height, (i) => List<int>.from(segmap[i]));

    for (var yy = sy; yy < ey; yy++) {
      for (var xx = sx; xx < ex; xx++) {
        // Check if any pixel in kernel neighborhood is set
        bool found = false;
        for (var ky = 0; ky < kernelSize && !found; ky++) {
          for (var kx = 0; kx < kernelSize && !found; kx++) {
            final ny = yy - kernelAnchor + ky;
            final nx = xx - kernelAnchor + kx;
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
              if (segmap[ny][nx] > 0) {
                found = true;
              }
            }
          }
        }
        if (found) {
          dilated[yy][xx] = 255;
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
      List<List<int>> segmap, int height, int width) {
    // Find contour points as done in the Python CRAFT utility pipeline.
    final points = <List<double>>[];
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        if (segmap[y][x] > 0) {
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
      int resizedWidth,
      int resizedHeight,
      int originalWidth,
      int originalHeight) {
    final scaleXResize = resizedWidth / scoreWidth;
    final scaleYResize = resizedHeight / scoreHeight;
    final scaleXOrig = originalWidth / resizedWidth;
    final scaleYOrig = originalHeight / resizedHeight;

    final result = <List<List<double>>>[];

    for (var i = 0; i < boxes.length; i++) {
      final box = boxes[i];
      final scaled = <List<double>>[];

      for (final point in box) {
        final x = point[0] * scaleXResize * scaleXOrig;
        final y = point[1] * scaleYResize * scaleYOrig;
        final clampedX = x.clamp(0.0, originalWidth.toDouble()).toDouble();
        final clampedY = y.clamp(0.0, originalHeight.toDouble()).toDouble();
        scaled.add([clampedX, clampedY]);
      }

      if (scaled.length < 4) {
        continue;
      }

      result.add(scaled);
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

  Future<Map<String, dynamic>> _detectImpl(XFile imageFile) async {
    final totalTimer = Stopwatch()..start();

    _verifyModelInputShape();
    final Uint8List uint8List = await imageFile.readAsBytes();

    final targetWidth = _inputTensor.shape[3];
    final targetHeight = _inputTensor.shape[2];
    final interpreterAddress = _interpreter.address;

    final modelResult = await Isolate.run(
      () => _runCraftPreprocessAndInference(
        uint8List,
        interpreterAddress,
        targetWidth,
        targetHeight,
      ),
    );

    final preprocessMs = modelResult['preprocessMs'] as double;
    final inferenceMs = modelResult['inferenceMs'] as double;
    final rawScoreShape = modelResult['rawScoreShape'] as List<int>;
    final rawScoreMap = modelResult['rawScoreMap'] as List<dynamic>;
    final resizedWidth = modelResult['resizedWidth'] as int;
    final resizedHeight = modelResult['resizedHeight'] as int;
    final originalWidth = modelResult['originalWidth'] as int;
    final originalHeight = modelResult['originalHeight'] as int;

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

    final postprocessTimer = Stopwatch()..start();
    final polygons = _postprocess(
      scoresRaw,
      scoreHeight,
      scoreWidth,
      resizedWidth,
      resizedHeight,
      originalWidth,
      originalHeight,
    );
    postprocessTimer.stop();

    totalTimer.stop();

    print('Detection timing (ms): '
        'preprocess=${preprocessMs.toStringAsFixed(1)}, '
        'inference=${inferenceMs.toStringAsFixed(1)}, '
        'postprocess=${_formatMs(postprocessTimer)}, '
        'total=${_formatMs(totalTimer)}');

    return {
      'polygons': polygons,
      'originalWidth': originalWidth,
      'originalHeight': originalHeight,
    };
  }

  Future<List<List<int>>> detect(XFile imageFile) async {
    final result = await _detectImpl(imageFile);
    final polygons = result['polygons'] as List<List<List<double>>>;
    final originalWidth = result['originalWidth'] as int;
    final originalHeight = result['originalHeight'] as int;
    return _polygonsToAxisAlignedBoxes(polygons, originalWidth, originalHeight);
  }

  Future<List<List<List<double>>>> detectPolygons(XFile imageFile) async {
    final result = await _detectImpl(imageFile);
    return result['polygons'] as List<List<List<double>>>;
  }
}
