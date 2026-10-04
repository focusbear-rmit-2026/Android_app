package com.focusbear.relevance

import android.content.Context
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/** Local developer-only CSV used to inspect Android ONNX shadow predictions. */
object RelevanceShadowLogger {

    data class Entry(
        val url: String,
        val tabTitle: String,
        val focusIntention: String,
        val prediction: RelevanceOnnxModel.Prediction,
        val existingDecision: String,
        val agreement: String,
    )

    @Synchronized
    fun append(context: Context, entry: Entry): File {
        val file = File(context.filesDir, "android-relevance-shadow.csv")
        val newFile = !file.exists() || file.length() == 0L

        FileWriter(file, true).use { writer ->
            if (newFile) {
                writer.append(
                    "timestamp,url,tab_title,focus_intention,onnx_tier,onnx_raw_action," +
                        "onnx_action,onnx_confidence,onnx_cosine,p_block,p_caution,p_allow," +
                        "existing_decision,agreement\n",
                )
            }

            val p = entry.prediction.probabilities
            val row = listOf(
                timestamp(),
                entry.url,
                entry.tabTitle,
                entry.focusIntention,
                entry.prediction.tier.toString(),
                entry.prediction.rawAction,
                entry.prediction.action,
                "%.6f".format(Locale.US, entry.prediction.confidence),
                "%.6f".format(Locale.US, entry.prediction.cosineSimilarity),
                "%.6f".format(Locale.US, p.getOrElse(0) { 0f }),
                "%.6f".format(Locale.US, p.getOrElse(1) { 0f }),
                "%.6f".format(Locale.US, p.getOrElse(2) { 0f }),
                entry.existingDecision,
                entry.agreement,
            )
            writer.append(row.joinToString(",") { csv(it) }).append('\n')
        }
        return file
    }

    private fun timestamp(): String {
        val formatter = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
        formatter.timeZone = TimeZone.getTimeZone("UTC")
        return formatter.format(Date())
    }

    private fun csv(value: String): String = "\"${value.replace("\"", "\"\"")}\""
}
