# Flutter Vision & TFLite Rules
-keep class org.tensorflow.** { *; }
-keep class org.tensorflow.lite.** { *; }
-keepclassmembers class org.tensorflow.** { *; }
-keepclassmembers class org.tensorflow.lite.** { *; }

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Flutter
-keep class io.flutter.app.** { *; }
-keep class io.flutter.plugin.** { *; }
-keep class io.flutter.util.** { *; }
-keep class io.flutter.view.** { *; }
-keep class io.flutter.** { *; }
-keep class io.flutter.plugins.** { *; }

# Camera plugin
-keep class io.flutter.plugins.camera.** { *; }

# Flutter Vision specific
-keep class com.visionplugin.flutter_vision.** { *; }
-dontwarn org.tensorflow.**
-dontwarn org.tensorflow.lite.**

-keep class org.tensorflow.** { *; }