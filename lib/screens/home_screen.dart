import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_text_extraction/services/tflite_service.dart';
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
  late Image _imageFile;
  late ImageHelper _imageHelper;
  late Future<void> _imageHelperInit;

  @override
  void initState() {
    super.initState();

    _imageHelper = ImageHelper();
    _imageHelperInit = _imageHelper.init();
    _imageFile = Image.asset('assets/wizardiusbewebicon.png');
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
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _imageFile,
            const SizedBox(height: 12),
            CameraButton(
              cameras: widget.cameras,
              onImageCaptured: _imageProcess,
            ),
            const SizedBox(height: 8),
            ImagePickerButton(
              onImagePicked: _imageProcess,
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _imageProcess(XFile imageFile) async {
    await _imageHelperInit;
    var outputFile = await _imageHelper.analyzeImage(imageFile);

    if (!mounted) return;
    setState(() {
      _imageFile = Image.memory(outputFile);
    });
  }
}
