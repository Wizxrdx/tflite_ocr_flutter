import 'package:camera/camera.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

class ImagePickerButton extends StatelessWidget {
  final ValueChanged<XFile> onImagePicked;

  const ImagePickerButton({
    super.key,
    required this.onImagePicked,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: () async {
        final result = await FilePicker.platform.pickFiles(
          type: FileType.custom,
          allowMultiple: false,
          withData: kIsWeb,
          allowedExtensions: const ['jpg', 'jpeg', 'png'],
        );

        if (result == null || result.files.isEmpty) return;

        final pickedFile = result.files.single;
        final path = pickedFile.path;
        if (path != null && path.isNotEmpty) {
          onImagePicked(XFile(path));
          return;
        }

        final bytes = pickedFile.bytes;
        if (bytes == null || bytes.isEmpty) return;

        onImagePicked(
          XFile.fromData(
            bytes,
            name: pickedFile.name,
            mimeType: _mimeTypeFromExtension(pickedFile.extension),
          ),
        );
      },
      child: const Text('Pick Image'),
    );
  }

  String? _mimeTypeFromExtension(String? extension) {
    switch (extension?.toLowerCase()) {
      case 'jpg':
      case 'jpeg':
        return 'image/jpeg';
      case 'png':
        return 'image/png';
      default:
        return null;
    }
  }
}
