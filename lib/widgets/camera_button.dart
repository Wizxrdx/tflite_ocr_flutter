import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_text_extraction/screens/camera_screen.dart';

class CameraButton extends StatelessWidget {
	final List<CameraDescription> cameras;
	final ValueChanged<XFile> onImageCaptured;

	const CameraButton({
		super.key,
		required this.cameras,
		required this.onImageCaptured,
	});

	@override
	Widget build(BuildContext context) {
		return FloatingActionButton(
			heroTag: 'camera_button',
      child: const Icon(Icons.camera_alt),
			onPressed: () async {
				try {
					await Navigator.push(
						context,
						MaterialPageRoute(
							builder: (context) => CameraScreen(
								cameras,
								imageFile: onImageCaptured,
							),
						),
					);
				} catch (e, st) {
					print('Camera open error: $e');
					print(st);
					final messenger = ScaffoldMessenger.maybeOf(context);
					if (messenger != null) {
						messenger.showSnackBar(
							SnackBar(content: Text('Camera open error: $e')),
						);
					}
				}
			},
		);
	}
}
