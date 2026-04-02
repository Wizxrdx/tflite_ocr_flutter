import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_text_extraction/services/camera_service.dart';

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
		return ElevatedButton(
			onPressed: () {
				Navigator.push(
					context,
					MaterialPageRoute(
						builder: (context) => CameraScreen(
							cameras,
							imageFile: onImageCaptured,
						),
					),
				);
			},
			child: const Text('Open Camera'),
		);
	}
}
