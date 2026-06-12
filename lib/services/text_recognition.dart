import 'dart:io';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_text_extraction/models/result.dart';

class TextRecognition {
  static const _modelPath = "assets/crnn_kurapan_fused.tflite";
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
      debugName: 'TextRecognitionIsolate',
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

    print('Model input shape: $inputShape');
  }

  String _formatMs(Stopwatch stopwatch) {
    return (stopwatch.elapsedMicroseconds / 1000.0).toStringAsFixed(1);
  }

  Future<void> recognizeText(Uint8List imageBytes, OCRResult result) async {
    if (!_isInitialized) {
      throw StateError('TextRecognition model is not initialized.');
    }
    // 1. Decode the original image ONCE before the loop
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      print("Failed to decode image");
      return;
    }
    final int inputHeight = 31;
    final int inputWidth = 200;

    for (var box in result.boxes) {
      // 2. Crop the image to the bounding box
      // Note: Make sure your box.x, box.y, box.width, box.height match the image's coordinate scale
      final cropped = img.copyCrop(
        originalImage, 
        x: box.x.toInt(), 
        y: box.y.toInt(), 
        width: box.width.toInt(), 
        height: box.height.toInt(),
      );
      // 3. Resize it to the exact dimensions expected by the model (200x31)
      final resized = img.copyResize(cropped, width: inputWidth, height: inputHeight);
      // 4. Convert to Grayscale
      final grayscale = img.grayscale(resized);
      // 5. Create the input tensor structure: [batch_size, height, width, channels] -> [1, 31, 200, 1]
      // tflite_flutter accepts nested standard Dart Lists as input
      var inputTensorData = List.generate(
        1, 
        (b) => List.generate(
          inputHeight, 
          (y) => List.generate(
            inputWidth, 
            (x) => List.filled(1, 0.0) // 1 channel (grayscale)
          )
        )
      );
      // 6. Iterate through pixels and extract/normalize the values
      for (int y = 0; y < inputHeight; y++) {
        for (int x = 0; x < inputWidth; x++) {
          // Get the pixel at (x, y)
          final pixel = grayscale.getPixel(x, y);
          
          // Extract the luminance value (0-255). 
          // Note: For image package >= 4.0.0 use `pixel.r` or `pixel.r.toDouble()`. 
          // For older versions use `img.getRed(pixel)`.
          final luminance = pixel.r.toDouble(); 
          
          // Normalize the value. 
          // Most models expect 0.0 to 1.0 (luminance / 255.0) 
          // Or -1.0 to 1.0 ((luminance - 127.5) / 127.5)
          // Let's use 0.0 to 1.0 as standard, adjust if your specific model needs -1 to 1.
          inputTensorData[0][y][x][0] = luminance / 255.0; 
          
          // If your model expects Float32 normalized values using mean/std mapping you would do:
          // inputTensorData[0][y][x][0] = (luminance - mean) / std;
        }
      }

      final address = _interpreter.address;
      final modelResult = await Isolate.run(
        () => _runRCNNPreprocessAndInference(
          address,
          inputTensorData
        ),
      );

      final outputData = modelResult['outputData'] as List<List<List<double>>>;
      final recognizedText = _decodeCTC(outputData[0]);
      print('Recognized text: $recognizedText at (${box.x}, ${box.y}, ${box.width}, ${box.height})');
      box.label = recognizedText; // Store the recognized text back in the box for later use
    }
  }

  static Map<String, dynamic> _runRCNNPreprocessAndInference(
    int address,
    List<List<List<List<double>>>> inputTensorData
  ) {
    final preprocessTimer = Stopwatch()..start();
    preprocessTimer.stop();

    final interpreter = Interpreter.fromAddress(address);
    final outputTensorData = List.generate(
      1,
      (i) => List.generate(
        48,
        (j) => List.filled(37, 0.0)
      )
    );

    final inferenceTimer = Stopwatch()..start();
    interpreter.run(inputTensorData, outputTensorData);
    inferenceTimer.stop();

    return {
      'preprocessTimeMs': preprocessTimer.elapsedMicroseconds / 1000.0,
      'inferenceTimeMs': inferenceTimer.elapsedMicroseconds / 1000.0,
      'outputData': outputTensorData,
    };
  }

  static String _decodeCTC(List<List<double>> sequence) {
    // Standard Kurapan alphabet. 
    // If your output is gibberish, your model might be using a different alphabet string.
    const String alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
    
    // The blank character is typically the last index (36). 
    // (In some models, it's at index 0, and the alphabet shifts by 1).
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
    return decodedText;
  }
}