package com.focusbear.relevance

import android.content.Context
import java.io.File
import java.io.FileWriter
import java.util.Locale

object Tier1BatchEvaluator {

    fun run(context: Context): File {
        val rows = context.assets
            .open("tier1_900_combined_examples.csv")
            .bufferedReader(Charsets.UTF_8)
            .readLines()

        require(rows.isNotEmpty()) { "Input CSV is empty" }

        val header = parseCsvLine(rows.first())
        val index = header.mapIndexed { i, name -> name.trim() to i }.toMap()

        val output = File(context.filesDir, "tier1-android-batch-results.csv")

        FileWriter(output, false).use { writer ->

            writer.append(
                "dataset,example_id,focus_intention,website_link,page_title," +
                    "expected_tier,predicted_tier,raw_action,final_action," +
                    "confidence,cosine,p_block,p_caution,p_allow,correct\n"
            )

            RelevanceOnnxModel(context).use { model ->

                rows.drop(1).forEachIndexed { rowIndex, line ->

                    if (line.isBlank()) return@forEachIndexed

                    val cols = parseCsvLine(line)

                    fun value(name: String): String {
                        val i = index[name] ?: return ""
                        return cols.getOrElse(i) { "" }
                    }

                    val dataset = value("Dataset")
                    val exampleId = value("Example_ID")
                    val focus = value("Focus_Intention")
                    val url = value("Website_Link")
                    val title = value("Expected_Page_Title")

                    val expectedTier =
                        value("Expected_Tier").toIntOrNull() ?: 1

                    try {
                        val prediction = model.predict(
                            focusIntention = focus,
                            tabTitle = title,
                            currentTask = "",
                            urlKeywords = "",
                            url = url,
                        )

                        val correct =
                            if (prediction.tier == expectedTier) "yes" else "no"

                        val p = prediction.probabilities

                        val result = listOf(
                            dataset,
                            exampleId,
                            focus,
                            url,
                            title,
                            expectedTier.toString(),
                            prediction.tier.toString(),
                            prediction.rawAction,
                            prediction.action,
                            "%.6f".format(Locale.US, prediction.confidence),
                            "%.6f".format(Locale.US, prediction.cosineSimilarity),
                            "%.6f".format(Locale.US, p.getOrElse(0) { 0f }),
                            "%.6f".format(Locale.US, p.getOrElse(1) { 0f }),
                            "%.6f".format(Locale.US, p.getOrElse(2) { 0f }),
                            correct,
                        )

                        writer.append(
                            result.joinToString(",") { csv(it) }
                        ).append('\n')

                        writer.flush()

                        println(
                            "Tier1BatchEvaluator ${rowIndex + 1}/900 " +
                                "predicted=${prediction.tier} " +
                                "action=${prediction.action}"
                        )

                    } catch (e: Exception) {
                        println(
                            "Tier1BatchEvaluator ERROR row ${rowIndex + 1}: ${e.message}"
                        )
                    }
                }
            }
        }

        println("Tier1BatchEvaluator FINISHED: ${output.absolutePath}")

        return output
    }

    private fun csv(value: String): String =
        "\"${value.replace("\"", "\"\"")}\""

    private fun parseCsvLine(line: String): List<String> {
        val result = mutableListOf<String>()
        val current = StringBuilder()

        var insideQuotes = false
        var i = 0

        while (i < line.length) {
            val c = line[i]

            when {
                c == '"' && insideQuotes &&
                    i + 1 < line.length &&
                    line[i + 1] == '"' -> {
                    current.append('"')
                    i++
                }

                c == '"' -> {
                    insideQuotes = !insideQuotes
                }

                c == ',' && !insideQuotes -> {
                    result += current.toString()
                    current.clear()
                }

                else -> current.append(c)
            }

            i++
        }

        result += current.toString()
        return result
    }
}
