import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // Diperlukan untuk mengatur orientasi layar
import 'detection_page.dart'; // Pastikan file detection_page.dart ada di folder yang sama (lib/)

void main() async {
  // 1. Wajib: Pastikan binding Flutter terinisialisasi sebelum akses plugin native (Kamera)
  WidgetsFlutterBinding.ensureInitialized();

  // 2. Opsional: Kunci orientasi layar ke Potrait agar tampilan kamera tidak gepeng/rusak
  // Jika aplikasimu nanti butuh landscape, hapus bagian SystemChrome ini.
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // 3. Jalankan Aplikasi
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // Hilangkan banner "DEBUG" di pojok kanan atas
      debugShowCheckedModeBanner: false,

      // Judul Aplikasi (tampil di Recent Apps)
      title: 'BISINDO Detector',

      // Tema Aplikasi
      theme: ThemeData(
        // Menggunakan Color Scheme Material 3
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blueAccent),
        useMaterial3: true,

        // Styling AppBar agar terlihat modern
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.blueAccent,
          foregroundColor: Colors.white,
          centerTitle: true,
          elevation: 2,
        ),
      ),

      // Halaman Awal Langsung ke Deteksi
      home: const DetectionPage(),
    );
  }
}