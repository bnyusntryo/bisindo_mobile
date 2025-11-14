allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
// ADD THIS NEW BLOCK IN ITS PLACE
subprojects {
    // This is the safer hook. It runs this code *as soon as* the
    // plugin is applied, not after evaluation.
    plugins.withId("com.android.library") {
        // Find the Android library extension and configure it
        extensions.findByType(com.android.build.gradle.LibraryExtension::class.java)?.let { android ->
            if (android.namespace == null) {
                // Set the namespace to the project's group if it's not already set
                android.namespace = group.toString()
            }
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
