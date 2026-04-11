import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tflite_text_extraction/helpers/image_processing.dart';
import 'package:tflite_text_extraction/services/text_detection.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('runs tflite inference on provided test image', () async {
    const inputPath = r'D:\Documents\keras-ocr\resources\test_image.png';

    expect(File(inputPath).existsSync(), isTrue,
        reason: 'Expected test image at $inputPath');

    final helper = TextDetection();
    await helper.init();

    final boxes = await helper.detect(XFile(inputPath));

    expect(boxes, isNotEmpty);

    final croppedImages =
        await extractImagesInsideBoundingBoxes(XFile(inputPath), boxes);

    expect(croppedImages, isNotEmpty);

    final outputBytes = await drawBoxesOnImage(XFile(inputPath), boxes);

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
  });
}
