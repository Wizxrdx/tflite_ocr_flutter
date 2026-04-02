import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_text_extraction/main.dart';
import 'package:tflite_text_extraction/screens/home_screen.dart';

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  final AppEnvironment environment;

  const MyApp({super.key, required this.cameras, required this.environment});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        debugShowCheckedModeBanner: environment == AppEnvironment.testing,
        theme: ThemeData(
          useMaterial3: true,
        ),
        home: HomeScreen(
          cameras: cameras,
          environment: environment,
        ));
  }
}