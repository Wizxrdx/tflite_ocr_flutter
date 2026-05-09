import 'dart:async';
import 'dart:typed_data';


import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite_text_extraction/services/text_detection.dart';
import 'package:tflite_text_extraction/widgets/camera_button.dart';
import 'package:tflite_text_extraction/widgets/image_display_widget.dart' show ImageDisplayWidget;
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
  late Uint8List? _imageBytes;
  late ImageProvider? _imageProvider;
  late TextDetection _textDetection;
  late Future<void> _imageHelperInit;
  late List<List<List<double>>> _polygons;
  bool _isDetecting = false;
  

  @override
  void initState() {
    super.initState();    

    _textDetection = TextDetection();
    _imageHelperInit = _textDetection.init();
    _imageBytes = null;
    _imageProvider = const AssetImage('assets/wizardiusbewebicon.png');
    _polygons = [];
  }

  @override
  void dispose() {
    unawaited(_textDetection.close());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBody: false,
      floatingActionButton: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        spacing: 16,
        children: [
          ImagePickerButton(
            onImagePicked: _imageProcess,
          ),
          CameraButton(
            cameras: widget.cameras,
            onImageCaptured: _imageProcess,
          ),
        ],
      ),
      body: SafeArea(
        top: false,
        child: Stack(
        children: [
          Center(
            child: IgnorePointer(
              ignoring: _isDetecting,
              child: Column(
                children: <Widget>[
                  Expanded(
                    child: Container(
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: Colors.black12,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: ImageDisplayWidget.fromRawOutput(
                        imageBytes: _imageBytes,
                        imageProvider: _imageProvider,
                        rawPolygons: _polygons,
                        boxFit: BoxFit.contain,
                        alignment: Alignment.center,
                      ),
                    ),
                  ),
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
    ));
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
    final imageBytes = await imageFile.readAsBytes();

    setState(() {
      _imageProvider = null;
      _imageBytes = imageBytes;
      _polygons = [];
    });

    try {
      await _imageHelperInit;
      final polygons = await _textDetection.detectPolygons(imageFile);
      // final outputFile = await drawPolygonsOnImage(imageFile, polygons);

      if (!mounted) return;
      setState(() {
        _polygons = polygons;
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
