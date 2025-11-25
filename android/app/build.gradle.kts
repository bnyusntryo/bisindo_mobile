plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.flutter_bisindo"
    compileSdk = 36
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    defaultConfig {
        applicationId = "com.example.flutter_bisindo"
        minSdk = flutter.minSdkVersion
        targetSdk = 34
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        // Filter arsitektur CPU agar sesuai dengan library TFLite
        // Ganti bagian yang error dengan ini:
        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            // PENTING: Matikan optimasi kode agar TFLite tidak rusak
            isMinifyEnabled = false
            isShrinkResources = false

            // Gunakan kunci debug agar bisa langsung diinstall (opsional, untuk testing)
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // === BAGIAN PALING PENTING (VERSI KOTLIN DSL) ===
    // Memaksa Android untuk TIDAK mengompres file model
    // Jika dikompres, akurasi hancur atau tidak terdeteksi
    aaptOptions {
        noCompress("tflite")
        noCompress("lite")
    }
    // ================================================
}

flutter {
    source = "../.."
}
