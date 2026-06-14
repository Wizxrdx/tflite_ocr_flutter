import 'dart:typed_data';

import 'package:tflite_text_extraction/models/result.dart';

class OCRWorkerRequest {
  final int id;
  final Uint8List imageBytes;
  OCRWorkerRequest({required this.id, required this.imageBytes});
}

class OCRWorkerResponse {
  final int id;
  final OCRResult? result;
  OCRWorkerResponse({required this.id, required this.result});
}

class OCRWorkerError {
  final int id;
  final String errorMessage;
  OCRWorkerError({required this.id, required this.errorMessage});
}
