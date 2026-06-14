import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_text_extraction/models/result.dart';

class TextRecognition {
  static const _modelPath = "assets/crnn_kurapan_fused.tflite";
  late Interpreter _interpreter;
  late Tensor _inputTensor;
  bool _isInitialized = false;

  get _inputWidth => _inputTensor.shape[2];
  get _inputHeight => _inputTensor.shape[1];

  Future<void> init() async {
    final options = InterpreterOptions()
      ..threads = max(1, min(4, Platform.numberOfProcessors));

    // Load model from assets
    _interpreter = await Interpreter.fromAsset(_modelPath, options: options);
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

  void _verifyTensorsShape() {
    print("Model input shape: ${_inputTensor.shape}");
    print("Model output shape: ${_interpreter.getOutputTensors().first.shape}");
  }

  Future<void> recognizeText(Uint8List imageBytes, OCRResult result) async {
    final textRecognitionTimer = Stopwatch()..start();
    _verifyTensorsShape();

    if (!_isInitialized) {
      throw StateError('TextRecognition model is not initialized.');
    }

    final preprocessTimer = Stopwatch()..start();
    final preprocessedImages = await _preprocess(imageBytes, result);
    preprocessTimer.stop();

    final inferenceTimer = Stopwatch()..start();
    final inferenceResults = _runInference(preprocessedImages);
    inferenceTimer.stop();

    final postprocessTimer = Stopwatch()..start();
    _postprocess(inferenceResults, result);
    postprocessTimer.stop();
    textRecognitionTimer.stop();

    print('Preprocessing time: ${preprocessTimer.elapsedMilliseconds} ms');
    print('Inference time: ${inferenceTimer.elapsedMilliseconds} ms');
    print('Postprocessing time: ${postprocessTimer.elapsedMilliseconds} ms');
    print('Total text recognition time: ${textRecognitionTimer.elapsedMilliseconds} ms');
  }

  List<List<List<double>>> _runInference(List<List<dynamic>> preprocessedImages) {
    final interpreter = Interpreter.fromAddress(_interpreter.address);

    final result = List<List<List<double>>>.generate(preprocessedImages.length, (i) => List<List<double>>.generate(48, (j) => List<double>.generate(37, (k) => 0.0)));

    for (int i = 0; i < preprocessedImages.length; i++) {
      final inputTensorData = preprocessedImages[i];
      final outputTensorData = List.generate(1, (_) => List.generate(48, (_) => List.generate(37, (_) => 0.0)));
      interpreter.run(inputTensorData, outputTensorData);

      result[i] = outputTensorData[0];
    }

    return result;
  }

  Future<List<List<dynamic>>> _preprocess(Uint8List imageBytes, OCRResult result) async {
    // 1. Decode the original image ONCE before the loop
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      print("Failed to decode image");
      return [];
    }
    
    List<List<dynamic>> processedImages = [];

    for (int im = 0; im < result.boxes.length; im++) {
      final box = result.boxes[im];

      final cropped = img.copyCrop(
        originalImage,
        x: box.x.toInt(),
        y: box.y.toInt(),
        width: box.width.toInt(),
        height: box.height.toInt(),
      );

      final resized = img.copyResize(cropped, width: _inputWidth, height: _inputHeight);
      final grayscale = img.grayscale(resized);

      final boxImage = Float32List(_inputHeight * _inputWidth);

      for (int y = 0; y < _inputHeight; y++) {
        for (int x = 0; x < _inputWidth; x++) {
          final pixel = grayscale.getPixel(x, y);
          final luminance = pixel.r.toDouble(); 
          
          boxImage[(y * _inputWidth + x) as int] = (luminance / 255.0);
        }
      }

      processedImages.add(boxImage.reshape([1, _inputHeight, _inputWidth, 1]));
    }
    return processedImages;
  }

  void _postprocess(List<List<List<double>>> sequences, OCRResult result) {
    const String alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";

    for (int i = 0; i < sequences.length; i++) {
      final box = result.boxes[i];
      final sequence = sequences[i];

      const int blankIndex = 36;
      String decodedText = "";
      int previousIndex = -1;
      // Iterate through each of the 48 timesteps
      for (int t = 0; t < sequence.length; t++) {
        final timestepProbabilities = sequence[t];
        // Find the index with the maximum probability (argmax)
        int maxIndex = 0;
        double maxProb = timestepProbabilities[0];
        
        for (int i = 1; i < timestepProbabilities.length; i++) {
          if (timestepProbabilities[i] > maxProb) {
            maxProb = timestepProbabilities[i];
            maxIndex = i;
          }
        }
        // CTC Rule: Ignore consecutive duplicates and ignore the blank character
        if (maxIndex != blankIndex && maxIndex != previousIndex) {
          decodedText += alphabet[maxIndex]; // Map index to the actual character
        }
        previousIndex = maxIndex;
      }

      box.label = decodedText;
    }
  }
}