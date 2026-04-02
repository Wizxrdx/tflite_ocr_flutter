import 'dart:developer' as dev;
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';

class ImageHelper {
  static const _modelPath =
      "assets/lite-model_east-text-detector_fp16_1.tflite";
  late Interpreter _interpreter;
  late Tensor _inputTensor;
  late Tensor _outputScoreTensor;
  late Tensor _outputGeometryTensor;

  Future<void> init() async {
    await _loadModel();
  }

  Future<void> _loadModel() async {
    final options = InterpreterOptions();
    // Load model from assets
    _interpreter = await Interpreter.fromAsset(_modelPath, options: options);
    _inputTensor = _interpreter.getInputTensors().first;
    _outputScoreTensor = _interpreter.getOutputTensors().first;
    _outputGeometryTensor = _interpreter.getOutputTensors().last;
  }

  void _runInference(img.Image inputImageData) async {
    // Resize image

    // Prepare input
    final imageMatrix = List.generate(
      inputImageData.height,
      (y) => List.generate(
        inputImageData.width,
        (x) {
          final pixel = inputImageData.getPixel(x, y);
          return [pixel.r - 103.939, pixel.g - 116.779, pixel.b - 123.68];
        },
      ),
    );

    final input = [imageMatrix];

    _interpreter.runInference([input]);
    _outputScoreTensor = _interpreter.getOutputTensors().first;
    _outputGeometryTensor = _interpreter.getOutputTensors().last;
  }

  Future<Uint8List> analyzeImage(XFile imageFile) async {
    final Uint8List uint8List = await imageFile.readAsBytes();

    img.Image? image = img.decodeImage(uint8List);
    img.Image? imageInput = img.copyResize(image!,
        width: _inputTensor.shape[1], height: _inputTensor.shape[2]);

    List scoresRaw = List.filled(
        _outputScoreTensor.shape[0],
        List.filled(
            _outputScoreTensor.shape[1],
            List.filled(_outputScoreTensor.shape[2],
                List<double>.filled(_outputScoreTensor.shape[3], 0.0))));
    List geometry = List.filled(
        _outputGeometryTensor.shape[0],
        List.filled(
            _outputGeometryTensor.shape[1],
            List.filled(_outputGeometryTensor.shape[2],
                List<double>.filled(_outputGeometryTensor.shape[3], 0.0))));

    _runInference(imageInput);
    _outputScoreTensor.copyTo(scoresRaw);
    _outputGeometryTensor.copyTo(geometry);

    // scoresRaw = _m2d.transpose(scoresRaw.toList());
    // geometry = _m2d.transpose(geometry.toList());

    var rects = List.empty(growable: true);
    var confidences = List.empty(growable: true);
    var cols = _outputGeometryTensor.shape[1];
    var rows = _outputGeometryTensor.shape[2];

    dev.log(cols.toString());
    dev.log(rows.toString());

    for (var i = 0; i < cols; i++) {
      for (var j = 0; j < rows; j++) {
        var scoresData = scoresRaw[0][i][j][0];
        var xData0 = geometry[0][i][j][0];
        var xData1 = geometry[0][i][j][1];
        var xData2 = geometry[0][i][j][2];
        var xData3 = geometry[0][i][j][3];
        var angle = geometry[0][i][j][4];

        // dev.log(scoresData.toString());

        if (scoresData >= 0.95) {
          // compute the offset factor as our resulting feature maps will
          // be 4x smaller than the input image
          var offsetX = j * 4.0;
          var offsetY = i * 4.0;

          // extract the rotation angle for the prediction and then
          // compute the sin and cosine
          var cosine = cos(angle);
          var sine = sin(angle);

          // use the geometry volume to derive the width and height of
          // the bounding box
          var h = xData0 + xData2;
          var w = xData1 + xData3;

          // compute both the starting and ending (x, y)-coordinates for
          // the text prediction bounding box
          var endX = (offsetX + (cosine * xData1) + (sine * xData2)).toInt();
          var endY = (offsetY - (sine * xData1) + (cosine * xData2)).toInt();
          var startX = (endX - w).toInt();
          var startY = (endY - h).toInt();

          rects.add([startX, startY, endX, endY]);
          confidences.add(scoresData);
        }
      }
    }

    var maxBoxes = 5;

    for (var i = 0; i < rects.length && i <= maxBoxes; i++) {
      img.drawRect(
        imageInput,
        x1: rects[i][0],
        y1: rects[i][1],
        x2: rects[i][2],
        y2: rects[i][3],
        color: img.ColorRgb8(0, 255, 0),
        thickness: 1,
      );

      img.drawString(
        imageInput,
        '${confidences[i]}',
        font: img.arial14,
        x: rects[i][0] + 1,
        y: rects[i][1] + 1,
        color: img.ColorRgb8(255, 0, 0),
      );
    }

    return img.encodeJpg(imageInput);
  }

  // non-maximum suppression
  void nms(List boxes, List scores, double overlapThresh) {

  }
}
