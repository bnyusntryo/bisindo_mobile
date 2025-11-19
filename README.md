# BISINDO Realtime Detector 🤟🇮🇩

Aplikasi mobile berbasis Flutter untuk mendeteksi Bahasa Isyarat Indonesia (BISINDO) secara realtime menggunakan Computer Vision (YOLOv8).

Project ini dikembangkan untuk menjembatani komunikasi teman Tuli dengan menerjemahkan isyarat tangan (Huruf & Kata) menjadi teks langsung di layar HP.

## 🧠 Model Architecture
* **Model:** YOLOv8 (Quantized Float16)
* **Format:** `.tflite`
* **Input Resolution:** 640x640
* **Classes:** Huruf (A-Z) dan Kata-kata umum (Aku, Kamu, Makan, Minum, dll).

## 🚀 Fitur Utama

* **Realtime Detection:** Menggunakan camera stream untuk deteksi instan.
* **Dual-Logic Approach:** Memiliki dua pendekatan algoritma untuk menangani bias model (penjelasan di bawah).
* **Stabilisasi Deteksi:** Menggunakan *Confidence Smoothing* (rata-rata 5 frame) dan *Anti-Flicker* agar bounding box tidak berkedip-kedip.
* **Memory Wipe:** Reset otomatis saat berganti mode untuk mencegah "hantu deteksi" (ghosting) dari mode sebelumnya.
* **Flashlight Toggle:** Dukungan pencahayaan tambahan untuk kondisi gelap.

---

## 🛠️ Logic Approaches (Solusi Masalah Deteksi)

Tantangan utama dalam project ini adalah **Model Bias**: Model seringkali salah memprediksi isyarat Huruf (misal: "L") menjadi Kata yang memiliki bentuk tangan awal serupa (misal: "LAKI-LAKI" atau "KAKAK"), atau sebaliknya.

Untuk mengatasi ini, aplikasi ini mengimplementasikan strategi **Strict Switch Mode**:

### 🔀 Strict Switch Mode (Final Solution)
Ini adalah pendekatan yang digunakan pada versi final (`DetectionPage`).

* **Konsep:** Pemisahan mutlak antara deteksi Huruf dan Kata.
* **Cara Kerja:**
    * Terdapat tombol **Switch** di pojok kanan atas (Ikon `ABC` vs `Chat Bubble`).
    * **Mode Abjad:** Sistem secara paksa **MEMBUANG** semua hasil deteksi yang panjang labelnya > 1 karakter (Kata).
    * **Mode Kata:** Sistem secara paksa **MEMBUANG** semua hasil deteksi yang panjang labelnya = 1 karakter (Huruf).
* **Keunggulan:**
    * Akurasi UX 100% sesuai konteks pengguna.
    * Tidak ada lagi kasus isyarat "C" hilang karena kalah skor dengan kata lain.
    * Tidak ada lagi kasus isyarat "L" terdeteksi sebagai kata "KAKAK".

### ⚖️ (Deprecated) Heuristic "Nerf & Buff"
Pendekatan eksperimental sebelumnya (tanpa tombol switch):
* **Konsep:** Memanipulasi nilai *confidence score* secara matematis.
* **Cara Kerja:** Memberikan hukuman (kurangi skor) pada Kata dan memberikan bonus (tambah skor) pada Huruf untuk menyeimbangkan bias dataset.
* **Status:** Digantikan oleh *Strict Switch Mode* karena kurang konsisten pada pencahayaan rendah.

---

## 📦 Instalasi & Setup

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/username/bisindo-detector.git](https://github.com/username/bisindo-detector.git)
    ```

2.  **Setup Assets**
    Pastikan file model dan label sudah ada di folder `assets/`:
    * `assets/best_float16.tflite`
    * `assets/labels.txt`
    * `assets/abjad.jfif` (Untuk panduan isyarat)

3.  **Install Dependencies**
    ```bash
    flutter pub get
    ```

4.  **Run App**
    Pastikan device fisik terhubung (karena emulator tidak support kamera dengan baik).
    ```bash
    flutter run
    ```

## ⚠️ Catatan Penting untuk Developer

1.  **Labels Formatting:**
    File `labels.txt` sangat sensitif terhadap *whitespace*. Kode aplikasi menggunakan `.trim()` dan regex cleaner untuk memastikan tidak ada karakter tersembunyi (seperti `\r` di Windows) yang menyebabkan deteksi "Huruf" terbaca memiliki panjang > 1.

2.  **Threshold:**
    Threshold model di-set sangat rendah (`0.15`) secara sengaja agar objek yang sulit terdeteksi (seperti huruf 'C' yang melengkung) tetap masuk ke dalam pipeline pemrosesan sebelum difilter oleh *Strict Logic*.

## 📱 Tech Stack

* [Flutter](https://flutter.dev/) - UI Framework
* [flutter_vision](https://pub.dev/packages/flutter_vision) - TFLite Interpreter Plugin
* [camera](https://pub.dev/packages/camera) - Camera Streaming

---
*Dibuat dengan ☕ dan 💻 oleh Mobile Developer.*