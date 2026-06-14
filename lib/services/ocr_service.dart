import 'dart:async';
import 'dart:isolate';
import 'package:flutter/services.dart';

import 'package:tflite_text_extraction/models/ocr_worker.dart';
import 'package:tflite_text_extraction/services/ocr_worker.dart' as worker;
import 'package:tflite_text_extraction/services/text_detection.dart';
import 'package:tflite_text_extraction/services/text_recognition.dart';

class OCRService {
  Isolate? _ocrWorker;
  late SendPort _sendPortToWorker;
  late final ReceivePort _receivePortFromWorker = ReceivePort();

  final Map<int, Completer<OCRWorkerResponse>> _pendingRequests = {};
  int _requestIdCounter = 0;
  bool _isInitialized = false;

  Future<void> init() async {
    if (_isInitialized) {
      return;
    }

    final completer = Completer<void>();

    _receivePortFromWorker.listen((message) {
      if (message is SendPort) {
        _setSendPort(message).then((_) => completer.complete());
      } else if (message is OCRWorkerResponse) {
        _pendingRequests.remove(message.id)?.complete(message);
      } else if (message is OCRWorkerError) {
        _pendingRequests.remove(message.id)?.completeError(message.errorMessage);
      } else {
        _pendingRequests.remove(message.id)?.completeError('Unknown message type from worker: $message');
      }
    });

    final detection = TextDetection();
    await detection.init();
    final recognition = TextRecognition();
    await recognition.init();

    final rootToken = RootIsolateToken.instance!;
    _ocrWorker = await Isolate.spawn(worker.workerEntryPoint, [
      _receivePortFromWorker.sendPort,
      rootToken,
      detection.interpreterAddress,
      recognition.interpreterAddress,
    ]);
  }

  Future<void> _setSendPort(SendPort sendPort) async {
    _sendPortToWorker = sendPort;
    _isInitialized = true;
  }

  Future<OCRWorkerResponse> performOCR(Uint8List imageBytes) async {
    if (!_isInitialized) {
      throw StateError('OCRService is not initialized. Call init() first.');
    }

    final requestId = _requestIdCounter++;
    final completer = Completer<OCRWorkerResponse>();
    _pendingRequests[requestId] = completer;

    _sendPortToWorker.send(OCRWorkerRequest(id: requestId, imageBytes: imageBytes));

    return completer.future;
  }

  Future<void> dispose() async {
    _receivePortFromWorker.close();
    _ocrWorker?.kill();
    _isInitialized = false;
  }

}
