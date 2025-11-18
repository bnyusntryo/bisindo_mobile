# Aturan yang sudah ada (JANGAN DIHAPUS)
-keep class org.tensorflow.lite.** { *; }
-keep class com.tflite.flutter_vision.** { *; }

# === TAMBAHAN BARU ===
# Abaikan error tentang AutoValue (Google Auto)
-dontwarn com.google.auto.value.**

# Abaikan error tentang komponen Audio/Support TFLite yang tidak kita pakai
-dontwarn org.tensorflow.lite.support.**
-dontwarn org.tensorflow.lite.**