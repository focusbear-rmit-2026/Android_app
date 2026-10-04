package com.focusbear.relevance

import android.content.Context
import java.io.File
import java.io.FileOutputStream

/** Copies packaged ONNX assets to app cache so ONNX Runtime can open them by file path. */
internal object OnnxAssetLoader {
    fun materialize(context: Context, assetPath: String): File {
        val target = File(context.cacheDir, "relevance_onnx/$assetPath")
        target.parentFile?.mkdirs()

        context.assets.open(assetPath).use { input ->
            val expectedSize = input.available().toLong()
            if (expectedSize > 0L && target.exists() && target.length() == expectedSize) {
                return target
            }

            FileOutputStream(target, false).use { output ->
                input.copyTo(output)
            }
        }

        return target
    }
}
