import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:oc_3000/counting_page.dart';


class HomeWidget extends StatefulWidget {
  const HomeWidget({super.key});

  @override
  State<HomeWidget> createState() => _HomeWidgetState();
}

class _HomeWidgetState extends State<HomeWidget> {

  final ImagePicker _picker = ImagePicker();
  late XFile? pickedFile;

  Future<void> _pickFromCamera() async {
    pickedFile = await _picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      _navigateToEditor(pickedFile!.path);
    }
  }

  Future<void> _pickFromGallery() async {
    pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      _navigateToEditor(pickedFile!.path);
    }
  }

  Future<void> _navigateToEditor(String imagePath) async {
    // final File? editedFile = await Navigator.push(
    //   context,
    //   // MaterialPageRoute(
    //   //   builder: (context) => EditPhotoScreen(imageFile: imagePath),
    //   // ),
    //   MaterialPageRoute(builder: (_)=> OnnxTesterScreen(imageFile: imagePath,)),
    // );
    // if (editedFile != null) {
    //   Navigator.push(
    //     context,
    //     MaterialPageRoute(
    //       builder: (context) => OnnxTesterScreen(imageFile: editedFile.path),
    //     ),
    //   );
    // } else {
    //   Navigator.push(
    //     context,
    //     MaterialPageRoute(
    //       builder: (context) => OnnxTesterScreen(imageFile: pickedFile!.path),
    //     ),
    //   );
    // }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => CountingPage(imageFile: pickedFile!.path),
      ),
    );
  }




  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('OC 3000 - Prototype'),

      ),

      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              onPressed: _pickFromCamera,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Capture Image from Camera'),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: _pickFromGallery,
              icon: const Icon(Icons.photo_library),
              label: const Text('Image from Gallery'),
            ),
          ],
        ),
      ),

    );
  }
}
