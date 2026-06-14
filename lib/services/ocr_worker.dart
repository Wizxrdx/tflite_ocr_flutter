import 'dart:isolate';
import 'package:flutter/services.dart';

import 'package:camera/camera.dart';
import 'package:tflite_text_extraction/models/ocr_worker.dart';
import 'package:tflite_text_extraction/models/result.dart';
import 'package:tflite_text_extraction/services/text_detection.dart';
import 'package:tflite_text_extraction/services/text_recognition.dart';

Future<void> workerEntryPoint(List<dynamic> args) async {
  final sendPortToMainThread = args[0] as SendPort;
  final rootToken = args[1] as RootIsolateToken;
  final detectionAddress = args[2] as int;
  final recognitionAddress = args[3] as int;

  BackgroundIsolateBinaryMessenger.ensureInitialized(rootToken);

  final workerReceivePort = ReceivePort();
  sendPortToMainThread.send(workerReceivePort.sendPort);

  final textDetection = TextDetection();
  final textRecognition = TextRecognition();
  await textDetection.init(address: detectionAddress);
  await textRecognition.init(address: recognitionAddress);

  // 3. Listen for incoming image requests
  await for (final message in workerReceivePort) {
    if (message is OCRWorkerRequest) {
      try {
        final xFile = XFile(message.imagePath);
        final imageBytes = await xFile.readAsBytes();
        
        final rawResult = await textDetection.detectBoxes(xFile);
        final result = OCRResult.fromRawOutput(rawResult);
        await textRecognition.recognizeText(imageBytes, result);

        // Send the final result back to the UI
        sendPortToMainThread.send(OCRWorkerResponse(id: message.id, result: result));
      } catch (e) {
        sendPortToMainThread.send(OCRWorkerError(id: message.id, errorMessage: e.toString()));
      }
    }
  }
}
