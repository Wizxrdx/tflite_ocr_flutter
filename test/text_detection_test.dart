import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tflite_text_extraction/helpers/image_processing.dart';
import 'package:tflite_text_extraction/models/result.dart';
import 'package:tflite_text_extraction/services/text_detection.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('runs tflite inference on provided test image', () async {
    const inputPath = r'D:\Documents\keras-ocr\resources\test_image.png';

    expect(File(inputPath).existsSync(), isTrue,
        reason: 'Expected test image at $inputPath');

    final detectionModel = TextDetection();
    await detectionModel.init();

    final rawBoxes = await detectionModel.detectBoxes(await File(inputPath).readAsBytes());
    final detectionResult = OCRResult.fromRawOutput(rawBoxes);

    expect(detectionResult.boxes, isNotEmpty);

    final croppedImages =
        await extractImagesInsideBoundingBoxes(XFile(inputPath), detectionResult.boxes);

    expect(croppedImages, isNotEmpty);

    final outputBytes = await drawBoxesOnImage(XFile(inputPath), detectionResult.boxes);

    final tempDirectory = Directory(
      '${Directory.current.path}${Platform.pathSeparator}temp',
    );
    await tempDirectory.create(recursive: true);

    final outputFile = File(
      '${tempDirectory.path}${Platform.pathSeparator}tflite_detect_${DateTime.now().microsecondsSinceEpoch}.jpg',
    );
    await outputFile.writeAsBytes(outputBytes, flush: true);

    expect(await outputFile.exists(), isTrue);
    expect(await outputFile.length(), greaterThan(0));
    print('Saved detected image to ${outputFile.path}');

    for (var i = 0; i < croppedImages.length; i++) {
      final croppedFile = File(
        '${tempDirectory.path}${Platform.pathSeparator}cropped_${DateTime.now().microsecondsSinceEpoch}_$i.jpg',
      );
      await croppedFile.writeAsBytes(croppedImages[i], flush: true);
      expect(await croppedFile.exists(), isTrue);
      expect(await croppedFile.length(), greaterThan(0));
      print('Saved cropped image to ${croppedFile.path}');
    }
  });
}
