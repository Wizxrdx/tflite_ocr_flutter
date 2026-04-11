import 'dart:async';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_text_extraction/helpers/image_processing.dart';
import 'package:tflite_text_extraction/services/text_detection.dart';
import 'package:tflite_text_extraction/widgets/camera_button.dart';
import 'package:tflite_text_extraction/widgets/image_picker_button.dart';
import 'package:tflite_text_extraction/main.dart';

class HomeScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  final AppEnvironment environment;

  const HomeScreen({
    super.key,
    required this.cameras,
    required this.environment,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late ImageProvider _imageProvider;
  late TextDetection _textDetection;
  late Future<void> _imageHelperInit;
  bool _isDetecting = false;

  @override
  void initState() {
    super.initState();

    _textDetection = TextDetection();
    _imageHelperInit = _textDetection.init();
    _imageProvider = const AssetImage('assets/wizardiusbewebicon.png');
  }

  @override
  void dispose() {
    unawaited(_textDetection.close());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor:
            ColorScheme.fromSeed(seedColor: Colors.deepPurple).inversePrimary,
        title: Text(
          widget.environment == AppEnvironment.testing
              ? 'Flutter Demo (Testing)'
              : 'Flutter Demo',
        ),
      ),
      body: Stack(
        children: [
          Center(
            child: IgnorePointer(
              ignoring: _isDetecting,
              child: Column(
                children: <Widget>[
                  Expanded(
                    child: Container(
                      width: double.infinity,
                      margin: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.black12,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Image(
                        image: _imageProvider,
                        fit: BoxFit.contain,
                        alignment: Alignment.center,
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  CameraButton(
                    cameras: widget.cameras,
                    onImageCaptured: _imageProcess,
                  ),
                  const SizedBox(height: 8),
                  ImagePickerButton(
                    onImagePicked: _imageProcess,
                  ),
                  const SizedBox(height: 12),
                ],
              ),
            ),
          ),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 220),
            child: !_isDetecting
                ? const SizedBox.shrink()
                : Container(
                    key: const ValueKey('detect-loader'),
                    color: Colors.black45,
                    child: Center(
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 24,
                          vertical: 20,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: const Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            SizedBox(
                              width: 34,
                              height: 34,
                              child: CircularProgressIndicator(strokeWidth: 3),
                            ),
                            SizedBox(height: 12),
                            Text(
                              'Detecting text...',
                              style: TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Future<void> _imageProcess(XFile imageFile) async {
    if (_isDetecting) {
      return;
    }

    setState(() {
      _isDetecting = true;
    });

    // Ensure the loading overlay is painted before heavy compute starts.
    await WidgetsBinding.instance.endOfFrame;

    try {
      await _imageHelperInit;
      final polygons = await _textDetection.detectPolygons(imageFile);
      final outputFile = await drawPolygonsOnImage(imageFile, polygons);

      if (!mounted) return;
      setState(() {
        _imageProvider = MemoryImage(outputFile);
      });
    } catch (error, stackTrace) {
      print('===== DETECTION ERROR START =====');
      print(error.toString());
      print('----- STACK TRACE -----');
      print(stackTrace.toString());
      print('===== DETECTION ERROR END =====');

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Detection failed: $error')),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isDetecting = false;
        });
      }
    }
  }
}
