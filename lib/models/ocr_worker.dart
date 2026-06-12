import 'package:tflite_text_extraction/models/result.dart';

class OCRWorkerRequest {
  final int id;
  final String imagePath;
  OCRWorkerRequest({required this.id, required this.imagePath});
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
