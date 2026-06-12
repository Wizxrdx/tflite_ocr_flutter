import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:tflite_text_extraction/models/result.dart';
import 'package:tflite_text_extraction/services/text_recognition.dart';
import 'package:image/image.dart' as img;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('runs tflite inference on provided test image', () async {
    const inputPath = r'D:\Documents\Flutter Projects\tflite_text_extraction\temp\cropped_1780479479358442_0.jpg';

    expect(File(inputPath).existsSync(), isTrue,
        reason: 'Expected test image at $inputPath');

    final recognitionModel = TextRecognition();
    await recognitionModel.init();
    final image = await img.decodeImageFile(inputPath);
    final imageBytes = await File(inputPath).readAsBytes();

    print('imageBytes.shape: ${imageBytes.length}');

    expect(image, isNotNull, reason: 'Failed to decode image at $inputPath');

    final detectionResult = OCRResult.fromRawOutput([LabeledBox(0, 0, image!.width, image.height)]);

    await recognitionModel.recognizeText(imageBytes, detectionResult);

    for (var boxes in detectionResult.boxes) {
      print('Recognized text: ${boxes.label} at (${boxes.x}, ${boxes.y}, ${boxes.width}, ${boxes.height})');
    }
  });
}