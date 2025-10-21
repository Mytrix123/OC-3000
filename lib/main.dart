import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img_lib;
import 'package:image_picker/image_picker.dart';
import 'package:oc_3000/home.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "ONNX Tester",
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const OnnxTesterPage(),
      // home: HomeWidget(),
    );
  }
}

class OnnxTesterPage extends StatefulWidget {
  const OnnxTesterPage({super.key});

  @override
  State<OnnxTesterPage> createState() => _OnnxTesterPageState();
}

class _OnnxTesterPageState extends State<OnnxTesterPage> {
  OnnxRuntime? _ort;
  OrtSession? _session;
  File? _selectedImage;
  File? _outputImage;
  int _detectedCount = 0;
  List<Map<String, dynamic>> _detections = [];
  String _status = 'Model tidak dimuat..';
  final double _confThreshold = 0.5;
  final double _iouThreshold = 0.45;
  final int _imgSize = 640;
  final List<String> _classNames = [
    'Bunch',
  ]; // Adjust based on your model classes
  String? outputPath;
  File? outputFile;
  bool isProcessing = false;
  Directory directory = Directory("");

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    try {
      _ort = OnnxRuntime();
      _session = await _ort!.createSessionFromAsset('assets/best-3000.onnx');
      // _session = await _ort!.createSessionFromAsset('assets/best_augmented.onnx');
      setState(() {
        _status = 'Model berhasil dimuat';
      });
    } catch (e) {
      setState(() {
        _status = 'Error loading model: $e';
      });
    }
  }

  Future<void> clearAppCache() async {
    try {
      Directory tempDir = await getTemporaryDirectory();
      if (tempDir.existsSync()) {
        tempDir.deleteSync(recursive: true);
        print('Cache directory cleared.');
      }
    } catch (e) {
      print('Error clearing cache: $e');
    }
  }

  Future<void> _pickImage({required String source}) async {
    final picker = ImagePicker();
    var pickedFile;
    if (_selectedImage != null) {
      _selectedImage!.deleteSync();
      if (directory.existsSync()) {
        print("directory path : ${directory.path}");
        print("output file path : ${outputFile!.path}");
        await outputFile!.delete;
        print("output file path : ${outputFile!.path}");

        print("output image file : ${_outputImage!.path}");
        await _outputImage!.delete;
        print("output image file : ${_outputImage!.path}");
        // directory.deleteSync(recursive: true);
        await directory.delete(recursive: true);
        print('Cache directory cleared.');
        print("directory path : ${directory.path}");
      } else {
        print('Cache directory does not exist.');
      }
    }
    if (source == 'camera'){
      pickedFile = await picker.pickImage(source: ImageSource.camera);
    }
    if (source == 'gallery') {
      pickedFile = await picker.pickImage(source: ImageSource.gallery);
    }
    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _outputImage = null;
        _detectedCount = 0;
        _detections = [];
        _status =
            'Image telah dipilih. \nMohon ditunggu sedang proses hitung TBS...';
      });
      await _runInference();
    }
  }

  Future<void> _runInference() async {
    if (_selectedImage == null || _session == null) {
      setState(() {
        _status = 'Tidak ada image yang dipilih atau model belum dimuat';
      });
      return;
    }

    try {
      setState(() {
        isProcessing = true;
      });

      // Preprocess image
      final inputTensor = await _preprocessImage(_selectedImage!);

      // Run inference
      final inputs = {
        'images': inputTensor,
      }; // Assuming input name is 'images' for YOLO
      final outputs = await _session!.run(inputs);
      print("output id : ${outputs[0]?.id}");
      print(
        'Outputs keys: ${outputs.keys}',
      ); // Debug: Check available output keys
      final outputTensor =
          outputs['output0'] ??
          outputs['output']; // Adjust output name if needed

      if (outputTensor == null) {
        setState(() {
          _status = 'Tidak ada output dari Model';
        });
        return;
      }

      // Postprocess: Handle nested list for [1, num_outputs, num_dets]
      final List<dynamic> rawOutputList =
          await outputTensor
              .asList(); // List<List<List<double>>> [batch, features, boxes]
      print(
        'rawOutputList length: ${rawOutputList.length}',
      ); // Expected: 1 (batch)
      if (rawOutputList.isEmpty) throw Exception('List output kosong');

      final List<dynamic> featuresList =
          rawOutputList[0]; // List<List<double>> [features, boxes]
      print(
        'featuresList length: ${featuresList.length}',
      ); // Expected: 5 (4 box + 1 class)
      print(
        'First feature length: ${(featuresList[0] as List).length}',
      ); // Expected: 8400

      final List<List<double>> output =
          featuresList
              .map((e) => (e as List).cast<double>())
              .toList(); // 5 x 8400

      _detections = _postprocess(output, _confThreshold, _iouThreshold);

      setState(() {
        _detectedCount = _detections.length;
        _status =
            'Perhitungan TBS selesai.\nTunggu proses penggambaran output dengan bounding box... ';
      });

      // Draw boxes and save output image
      await _drawBoxesAndSave();
    } catch (e) {
      setState(() {
        _status = 'Error during inference: $e';
        print("Error during inference: $e");
      });
    }
  }

  Future<OrtValue> _preprocessImage(File imageFile) async {
    // Load image
    final bytes = await imageFile.readAsBytes();
    img_lib.Image image = img_lib.decodeImage(bytes)!;

    print("ukuran image aslinya : ${image.width} x ${image.height}");

    // Resize to 640x640
    image = img_lib.copyResize(
      image,
      width: _imgSize,
      height: _imgSize,
      interpolation: img_lib.Interpolation.linear,
    );
    print("ukuran image resize: ${image.width} x ${image.height}");

    // Normalize and convert to float32 tensor [1, 3, 640, 640]
    final Float32List inputData = Float32List(1 * 3 * _imgSize * _imgSize);
    int pixelIndex = 0;
    for (int c = 0; c < 3; c++) {
      // Channels first
      for (int y = 0; y < _imgSize; y++) {
        for (int x = 0; x < _imgSize; x++) {
          final pixel = image.getPixel(x, y);
          final value =
              c == 0
                  ? pixel.r / 255.0
                  : (c == 1 ? pixel.g / 255.0 : pixel.b / 255.0);
          inputData[pixelIndex++] = value;
        }
      }
    }

    return OrtValue.fromList(inputData, [1, 3, _imgSize, _imgSize]);
  }

  List<Map<String, dynamic>> _postprocess(
    List<List<double>> output,
    double confThresh,
    double iouThresh,
  ) {
    final int numOutputs = output.length; // Expected: 5 (4 box + num_classes)
    final int numDets = output[0].length; // Expected: 8400
    print("output length : ${output.length}");
    print("jml terdeteksi: ${numDets}");
    for (var list in output) {
      if (list.length != numDets)
        throw Exception('Inconsistent feature lengths in output');
    }

    if (numOutputs != 4 + _classNames.length)
      throw Exception(
        'Unexpected numOutputs: $numOutputs (expected ${4 + _classNames.length})',
      );

    List<Map<String, dynamic>> dets = [];
    for (int i = 0; i < numDets; i++) {
      final double cx = output[0][i];
      final double cy = output[1][i];
      final double w = output[2][i];
      final double h = output[3][i];

      // For 1 class, conf is directly the class confidence at index 4
      final double conf = output[4][i];
      if (conf < confThresh) continue;

      final double x1 = cx - w / 2;
      final double y1 = cy - h / 2;
      final double x2 = cx + w / 2;
      final double y2 = cy + h / 2;

      dets.add({
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'conf': conf,
        'classId': 0, // Since only 1 class
      });
    }

    // Apply NMS
    return _nms(dets, iouThresh);
  }

  List<Map<String, dynamic>> _nms(
    List<Map<String, dynamic>> boxes,
    double iouThresh,
  ) {
    // Sort by confidence descending
    boxes.sort((a, b) => b['conf'].compareTo(a['conf']));

    List<Map<String, dynamic>> result = [];
    while (boxes.isNotEmpty) {
      final best = boxes.removeAt(0);
      result.add(best);

      boxes.removeWhere((box) {
        return _iou(best, box) > iouThresh;
      });
    }
    return result;
  }

  double _iou(Map<String, dynamic> box1, Map<String, dynamic> box2) {
    final double x1 = max(box1['x1'], box2['x1']);
    final double y1 = max(box1['y1'], box2['y1']);
    final double x2 = min(box1['x2'], box2['x2']);
    final double y2 = min(box1['y2'], box2['y2']);

    final double interArea = max(0, x2 - x1) * max(0, y2 - y1);
    final double box1Area =
        (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1']);
    final double box2Area =
        (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1']);

    return interArea / (box1Area + box2Area - interArea);
  }

  Future<void> _drawBoxesAndSave() async {
    if (_selectedImage == null) return;

    final bytes = await _selectedImage!.readAsBytes();
    print("bytes length : ${bytes.length}");
    print("bytes type : ${bytes.runtimeType}");

    img_lib.Image image = img_lib.decodeImage(bytes)!;

    print("Image size: ${image.width}x${image.height}");
    print("Image format: ${image.format}");

    // Scale boxes to original image size
    final originalWidth = image.width;
    final originalHeight = image.height;
    print("originalWidth : $originalWidth");
    print("originalHeight : $originalHeight");

    for (var det in _detections) {
      double x1 = det['x1'] * originalWidth / _imgSize;
      double y1 = det['y1'] * originalHeight / _imgSize;
      double x2 = det['x2'] * originalWidth / _imgSize;
      double y2 = det['y2'] * originalHeight / _imgSize;

      // Draw rectangle
      image = img_lib.drawRect(
        image,
        x1: x1.toInt(),
        y1: y1.toInt(),
        x2: x2.toInt(),
        y2: y2.toInt(),
        color: img_lib.ColorRgb8(245, 138, 66),
        thickness: 5,
      );

      // Draw label
      final label =
          '${_classNames[det['classId']]} ${det['conf'].toStringAsFixed(2)}';
      image = img_lib.drawString(
        image,
        label,
        font: img_lib.arial14,
        x: x1.toInt(),
        y: y1.toInt() - 20,
        color: img_lib.ColorRgb8(255, 0, 0),
      );
    }

    // Save output image with unique filename to avoid caching issues
    final outputBytes = img_lib.encodeJpg(image);
    directory = await getTemporaryDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    outputPath = '${directory.path}/output_image_with_boxes_$timestamp.jpg';
    outputFile = File(outputPath!);
    await outputFile!.writeAsBytes(outputBytes);
    print("output path : $outputPath");
    print("output file : $outputFile");

    setState(() {
      _outputImage = outputFile;
      isProcessing = false;
      _status = "Proses selesai";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('OC 3000 - Prototype')),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text("Status: $_status", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),),
              const SizedBox(height: 20),


              // Image.file(_selectedImage!, height: 200),
              if (!isProcessing)
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: (){_pickImage(source: 'camera');},
                      child: const Text('Camera'),
                    ),
                    SizedBox(width: 20,),
                    ElevatedButton(
                      onPressed: (){_pickImage(source: 'gallery');},
                      child: const Text('Gallery'),
                    ),
                  ],
                ),
              // if (isProcessing)
              //    CircularProgressIndicator(),

              // ElevatedButton(
              //   onPressed: _runInference,
              //   child: const Text('Hitung TBS'),
              // ),

              if (_selectedImage != null)
                Column(
                  children: [
                    const Text('Input Image:'),
                    Image.file(_selectedImage!, height: 200),
                  ],
                ),

              if (_outputImage != null)
                Column(
                  children: [
                    const Text('Output Image:'),
                    Image.file(
                      _outputImage!,
                      height: 300,
                      // key: UniqueKey(),  // Force reload to avoid caching
                    ),
                    Text('Jumlah TBS terdeteksi: $_detectedCount'),
                  ],
                ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Future<void> dispose() async {
    await _outputImage?.delete(recursive: true);
    await outputFile?.delete(recursive: true);
    await _selectedImage?.delete(recursive: true);

    // outputFile!.deleteSync();
    super.dispose();
  }
}

//
// import 'dart:async';
// import 'dart:io';
// import 'dart:math';
// import 'dart:typed_data';
//
// import 'package:flutter/material.dart';
// import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
// import 'package:image/image.dart' as img_lib;
// import 'package:image_picker/image_picker.dart';
// import 'package:path_provider/path_provider.dart';
//
// void main() {
//   runApp(const MyApp());
// }
//
// class MyApp extends StatelessWidget {
//   const MyApp({super.key});
//
//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       title: "ONNX Tester",
//       theme: ThemeData(
//         primarySwatch: Colors.blue,
//       ),
//       home: const OnnxTesterPage(),
//     );
//   }
// }
//
// class OnnxTesterPage extends StatefulWidget {
//   const OnnxTesterPage({super.key});
//
//   @override
//   State<OnnxTesterPage> createState() => _OnnxTesterPageState();
// }
//
// class _OnnxTesterPageState extends State<OnnxTesterPage> {
//   OnnxRuntime? _ort;
//   OrtSession? _session;
//   File? _selectedImage;
//   File? _outputImage;
//   int _detectedCount = 0;
//   List<Map<String, dynamic>> _detections = [];
//   String _status = 'Model tidak dimuat..';
//   final double _confThreshold = 0.5;
//   final double _iouThreshold = 0.45;
//   final int _imgSize = 640;
//   final List<String> _classNames = ['Bunch'];  // Adjust based on your model classes
//   String? outputPath;
//   File? outputFile;
//   bool isProcessing = false;
//   Directory directory = Directory("");
//
//   @override
//   void initState() {
//     super.initState();
//     _initModel();
//   }
//
//   Future<void> _initModel() async {
//     try {
//       _ort = OnnxRuntime();
//       _session = await _ort!.createSessionFromAsset('assets/best-3000.onnx');
//       setState(() {
//         _status = 'Model berhasil dimuat';
//       });
//     } catch (e) {
//       setState(() {
//         _status = 'Error loading model: $e';
//       });
//     }
//   }
//
//   Future<void> clearAppCache() async {
//     try {
//       Directory tempDir = await getTemporaryDirectory();
//       if (tempDir.existsSync()) {
//         tempDir.deleteSync(recursive: true);
//         print('Cache directory cleared.');
//       }
//     } catch (e) {
//       print('Error clearing cache: $e');
//     }
//   }
//
//   Future<void> _pickImage() async {
//     final picker = ImagePicker();
//     if (_selectedImage != null) {
//       _selectedImage!.deleteSync();
//       if (directory.existsSync()) {
//         print("directory path : ${directory.path}");
//         print("output file path : ${outputFile!.path}");
//         await outputFile!.delete();
//         print("output file path : ${outputFile!.path}");
//         print("output image file : ${_outputImage!.path}");
//         await _outputImage!.delete();
//         print("output image file : ${_outputImage!.path}");
//         await directory.delete(recursive: true);
//         print('Cache directory cleared.');
//         print("directory path : ${directory.path}");
//       } else {
//         print('Cache directory does not exist.');
//       }
//     }
//     final pickedFile = await picker.pickImage(source: ImageSource.gallery);
//     if (pickedFile != null) {
//       setState(() {
//         _selectedImage = File(pickedFile.path);
//         _outputImage = null;
//         _detectedCount = 0;
//         _detections = [];
//         _status = 'Image telah dipilih. Selanjutnya Hitung TBS.';
//       });
//     }
//   }
//
//   Future<void> _runInference() async {
//     if (_selectedImage == null || _session == null) {
//       setState(() {
//         _status = 'Tidak ada image yang dipilih atau model belum dimuat';
//       });
//       return;
//     }
//
//     try {
//       setState(() {
//         isProcessing = true;
//       });
//
//       // Preprocess image
//       final inputTensor = await _preprocessImage(_selectedImage!);
//
//       // Run inference
//       final inputs = {'images': inputTensor};  // Assuming input name is 'images' for YOLO
//       final outputs = await _session!.run(inputs);
//       print('Outputs keys: ${outputs.keys}');  // Debug: Check available output keys
//       final outputTensor = outputs['output0'] ?? outputs['output'];  // Adjust output name if needed
//
//       if (outputTensor == null) {
//         setState(() {
//           _status = 'Tidak ada output dari Model';
//         });
//         return;
//       }
//
//       // Postprocess: Handle nested list for [1, num_outputs, num_dets]
//       final List<dynamic> rawOutputList = await outputTensor.asList();  // List<List<List<double>>> [batch, features, boxes]
//       print('rawOutputList length: ${rawOutputList.length}');  // Expected: 1 (batch)
//       if (rawOutputList.isEmpty) throw Exception('List output kosong');
//
//       final List<dynamic> featuresList = rawOutputList[0];  // List<List<double>> [features, boxes]
//       print('featuresList length: ${featuresList.length}');  // Expected: 5 (4 box + 1 class)
//       print('First feature length: ${(featuresList[0] as List).length}');  // Expected: 8400
//
//       final List<List<double>> output = featuresList.map((e) => (e as List).cast<double>()).toList();  // 5 x 8400
//
//       _detections = _postprocess(output, _confThreshold, _iouThreshold);
//
//       setState(() {
//         _detectedCount = _detections.length;
//         _status = 'Perhitungan TBS selesai.\nTunggu proses penggambaran output dengan bounding box... ';
//       });
//
//       // Draw boxes and save output image
//       await _drawBoxesAndSave();
//     } catch (e) {
//       setState(() {
//         _status = 'Error during inference: $e';
//         print("Error during inference: $e");
//       });
//     }
//   }
//
//   Future<OrtValue> _preprocessImage(File imageFile) async {
//     // Load image
//     final bytes = await imageFile.readAsBytes();
//     img_lib.Image image = img_lib.decodeImage(bytes)!;
//
//     print("ukuran imageFile : ${image.width} x ${image.height}");
//
//     // Resize to 640x640
//     image = img_lib.copyResize(image, width: _imgSize, height: _imgSize, interpolation: img_lib.Interpolation.linear);
//     print("ukuran image : ${image.width} x ${image.height}");
//
//     // Normalize and convert to float32 tensor [1, 3, 640, 640]
//     final Float32List inputData = Float32List(1 * 3 * _imgSize * _imgSize);
//     int pixelIndex = 0;
//     for (int c = 0; c < 3; c++) {  // Channels first
//       for (int y = 0; y < _imgSize; y++) {
//         for (int x = 0; x < _imgSize; x++) {
//           final pixel = image.getPixel(x, y);
//           final value = c == 0 ? pixel.r / 255.0 : (c == 1 ? pixel.g / 255.0 : pixel.b / 255.0);
//           inputData[pixelIndex++] = value;
//         }
//       }
//     }
//
//     return OrtValue.fromList(inputData, [1, 3, _imgSize, _imgSize]);
//   }
//
//   List<Map<String, dynamic>> _postprocess(List<List<double>> output, double confThresh, double iouThresh) {
//     final int numOutputs = output.length;  // Expected: 5 (4 box + num_classes)
//     final int numDets = output[0].length;  // Expected: 8400
//     for (var list in output) {
//       if (list.length != numDets) throw Exception('Inconsistent feature lengths in output');
//     }
//
//     if (numOutputs != 4 + _classNames.length) throw Exception('Unexpected numOutputs: $numOutputs (expected ${4 + _classNames.length})');
//
//     List<Map<String, dynamic>> dets = [];
//     for (int i = 0; i < numDets; i++) {
//       final double cx = output[0][i];
//       final double cy = output[1][i];
//       final double w = output[2][i];
//       final double h = output[3][i];
//
//       // For 1 class, conf is directly the class confidence at index 4
//       final double conf = output[4][i];
//       if (conf < confThresh) continue;
//
//       final double x1 = cx - w / 2;
//       final double y1 = cy - h / 2;
//       final double x2 = cx + w / 2;
//       final double y2 = cy + h / 2;
//
//       dets.add({
//         'x1': x1,
//         'y1': y1,
//         'x2': x2,
//         'y2': y2,
//         'conf': conf,
//         'classId': 0,  // Since only 1 class
//       });
//     }
//
//     // Apply NMS
//     return _nms(dets, iouThresh);
//   }
//
//   List<Map<String, dynamic>> _nms(List<Map<String, dynamic>> boxes, double iouThresh) {
//     // Sort by confidence descending
//     boxes.sort((a, b) => b['conf'].compareTo(a['conf']));
//
//     List<Map<String, dynamic>> result = [];
//     while (boxes.isNotEmpty) {
//       final best = boxes.removeAt(0);
//       result.add(best);
//
//       boxes.removeWhere((box) {
//         return _iou(best, box) > iouThresh;
//       });
//     }
//     return result;
//   }
//
//   double _iou(Map<String, dynamic> box1, Map<String, dynamic> box2) {
//     final double x1 = max(box1['x1'], box2['x1']);
//     final double y1 = max(box1['y1'], box2['y1']);
//     final double x2 = min(box1['x2'], box2['x2']);
//     final double y2 = min(box1['y2'], box2['y2']);
//
//     final double interArea = max(0, x2 - x1) * max(0, y2 - y1);
//     final double box1Area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1']);
//     final double box2Area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1']);
//
//     return interArea / (box1Area + box2Area - interArea);
//   }
//
//   Future<void> _drawBoxesAndSave() async {
//     if (_selectedImage == null) return;
//
//     final bytes = await _selectedImage!.readAsBytes();
//     print("bytes length : ${bytes.length}");
//     print("bytes type : ${bytes.runtimeType}");
//
//     img_lib.Image image = img_lib.decodeImage(bytes)!;
//
//     print("Image size: ${image.width}x${image.height}");
//     print("Image format: ${image.format}");
//
//     // Scale boxes to original image size
//     final originalWidth = image.width;
//     final originalHeight = image.height;
//     print("originalWidth : $originalWidth");
//     print("originalHeight : $originalHeight");
//
//     for (var det in _detections) {
//       double x1 = det['x1'] * originalWidth / _imgSize;
//       double y1 = det['y1'] * originalHeight / _imgSize;
//       double x2 = det['x2'] * originalWidth / _imgSize;
//       double y2 = det['y2'] * originalHeight / _imgSize;
//
//       // Draw rectangle
//       image = img_lib.drawRect(
//         image,
//         x1: x1.toInt(),
//         y1: y1.toInt(),
//         x2: x2.toInt(),
//         y2: y2.toInt(),
//         color: img_lib.ColorRgb8(245, 138, 66),
//         thickness: 5,
//       );
//
//       // Draw label
//       final label = '${_classNames[det['classId']]} ${det['conf'].toStringAsFixed(2)}';
//       image = img_lib.drawString(
//         image,
//         label,
//         font: img_lib.arial14,
//         x: x1.toInt(),
//         y: y1.toInt() - 20,
//         color: img_lib.ColorRgb8(255, 0, 0),
//       );
//     }
//
//     // Save output image with unique filename to avoid caching issues
//     final outputBytes = img_lib.encodeJpg(image);
//     directory = await getTemporaryDirectory();
//     final timestamp = DateTime.now().millisecondsSinceEpoch;
//     outputPath = '${directory.path}/output_image_with_boxes_$timestamp.jpg';
//     outputFile = File(outputPath!);
//     await outputFile!.writeAsBytes(outputBytes);
//     print("output path : $outputPath");
//     print("output file : $outputFile");
//
//     setState(() {
//       _outputImage = outputFile;
//       isProcessing = false;
//     });
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: const Text('Object Counting Prototype'),
//       ),
//       body: SingleChildScrollView(
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             Text(_status),
//             const SizedBox(height: 20),
//             if (_selectedImage != null)
//               Column(
//                 children: [
//                   const Text('Input Image:'),
//                   Image.file(_selectedImage!, height: 200),
//                 ],
//               ),
//             ElevatedButton(
//               onPressed: _pickImage,
//               child: const Text('Pilih Image'),
//             ),
//             ElevatedButton(
//               onPressed: _runInference,
//               child: const Text('Hitung TBS'),
//             ),
//             if (_outputImage != null)
//               Column(
//                 children: [
//                   const Text('Output Image:'),
//                   SizedBox(
//                     width: 480,
//                     height: 480,
//                     child: Image.file(
//                       _outputImage!,
//                       fit: BoxFit.contain,
//                     ),
//                   ),
//                   Text('Jumlah TBS terdeteksi: $_detectedCount'),
//                 ],
//               ),
//           ],
//         ),
//       ),
//     );
//   }
//
//   @override
//   Future<void> dispose() async {
//     await _outputImage?.delete(recursive: true);
//     await outputFile?.delete(recursive: true);
//     await _selectedImage?.delete(recursive: true);
//     super.dispose();
//   }
// }
