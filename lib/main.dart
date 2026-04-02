import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_text_extraction/app.dart';

enum AppEnvironment { testing, production }

void _logError(String code, String? message) {
  // ignore: avoid_print
  print('Error: $code${message == null ? '' : '\nError Message: $message'}');
}

Future<void> bootstrapApp(AppEnvironment environment) async {
  List<CameraDescription> cameras = <CameraDescription>[];

  try {
    WidgetsFlutterBinding.ensureInitialized();
    cameras = await availableCameras();
  } on CameraException catch (e) {
    _logError(e.code, e.description);
  }
  runApp(MyApp(cameras: cameras, environment: environment));
}

void main() async {
  await bootstrapApp(AppEnvironment.production);
}