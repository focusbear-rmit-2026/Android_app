package com.focusbear.relevance

import android.content.Context
import java.io.File
import java.io.FileWriter
import java.util.Locale

object HardCaseLatencyEvaluator {

    fun run(context: Context): File {

        val lines = context.assets
            .open("hard_test_cases_updated.csv")
            .bufferedReader(Charsets.UTF_8)
            .readLines()

        require(lines.isNotEmpty()) { "hard_test_cases_updated.csv is empty" }

        val output = File(
            context.filesDir,
            "hard-test-android-results.csv"
        )

        FileWriter(output, false).use { writer ->

            writer.append(
                "focus_intention,tab_title,expected_tier," +
                "predicted_tier,raw_action,final_action," +
                "confidence,cosine,p_block,p_caution,p_allow," +
                "latency_ms,correct\n"
            )

            RelevanceOnnxModel(context).use { model ->

                lines.drop(1).forEachIndexed { index, line ->

                    if (line.isBlank()) {
                        return@forEachIndexed
                    }

                    val columns = parseCsvLine(line)

                    val focusIntention =
                        columns.getOrElse(0) { "" }.trim()

                    val tabTitle =
                        columns.getOrElse(1) { "" }.trim()

                    val expectedTier =
                        columns.getOrElse(2) { "1" }
                            .trim()
                            .toIntOrNull() ?: 1

                    try {

                        val startNs = System.nanoTime()

                        val prediction = model.predict(
                            focusIntention = focusIntention,
                            tabTitle = tabTitle,
                            currentTask = "",
                            urlKeywords = "",
                            url = ""
                        )

                        val latencyMs =
                            (System.nanoTime() - startNs) / 1_000_000.0

                        val probabilities =
                            prediction.probabilities

                        val correct =
                            if (prediction.tier == expectedTier)
                                "yes"
                            else
                                "no"

                        val result = listOf(
                            focusIntention,
                            tabTitle,
                            expectedTier.toString(),
                            prediction.tier.toString(),
                            prediction.rawAction,
                            prediction.action,
                            "%.6f".format(
                                Locale.US,
                                prediction.confidence
                            ),
                            "%.6f".format(
                                Locale.US,
                                prediction.cosineSimilarity
                            ),
                            "%.6f".format(
                                Locale.US,
                                probabilities.getOrElse(0) { 0f }
                            ),
                            "%.6f".format(
                                Locale.US,
                                probabilities.getOrElse(1) { 0f }
                            ),
                            "%.6f".format(
                                Locale.US,
                                probabilities.getOrElse(2) { 0f }
                            ),
                            "%.3f".format(
                                Locale.US,
                                latencyMs
                            ),
                            correct
                        )

                        writer.append(
                            result.joinToString(",") {
                                csv(it)
                            }
                        )

                        writer.append('\n')
                        writer.flush()

                        println(
                            "HardCaseLatency " +
                            "${index + 1}/${lines.size - 1} " +
                            "tier=${prediction.tier} " +
                            "action=${prediction.action} " +
                            "latency=${"%.2f".format(Locale.US, latencyMs)}ms"
                        )

                    } catch (e: Exception) {

                        println(
                            "HardCaseLatency ERROR row " +
                            "${index + 1}: ${e.message}"
                        )
                    }
                }
            }
        }

        println(
            "HardCaseLatency FINISHED: " +
            output.absolutePath
        )

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

                c == '"' &&
                    insideQuotes &&
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
