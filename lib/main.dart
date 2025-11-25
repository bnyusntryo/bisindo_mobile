import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'intro_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Memperbaiki orientasi yang diizinkan
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // Warna Aksen Utama: Dark Teal yang elegan
  static const Color primaryTeal = Color(0xFF004D40);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'BISINDO Detector',
      theme: ThemeData(
        // Skema Warna: Menggunakan warna Teal yang dalam sebagai seed color
        colorScheme: ColorScheme.fromSeed(
          seedColor: primaryTeal,
          brightness: Brightness.light,
          // Menggunakan 'surface' untuk background utama (Putih Bersih)
          surface: Colors.white, 
        ),
        useMaterial3: true,

        // Tipografi: Menggunakan font default Material 3 dengan kontras tinggi
        textTheme: TextTheme(
          titleLarge: TextStyle(fontWeight: FontWeight.bold, color: Colors.grey.shade900),
          bodyMedium: TextStyle(color: Colors.grey.shade700),
        ),

        // AppBar: Datar (Flat), Bersih, dan Minimalis
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.white, 
          foregroundColor: primaryTeal, 
          centerTitle: true,
          elevation: 0, // Menghilangkan bayangan (flat look)
          surfaceTintColor: Colors.transparent, 
          titleTextStyle: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: primaryTeal,
          ),
        ),

        // Perbaikan Visual Tambahan (Button)
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: primaryTeal,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ),
      home: const IntroPage(), // 👈 Changed from DetectionPage to IntroPage
    );
  }
}