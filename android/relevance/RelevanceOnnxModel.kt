package com.focusbear.relevance

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

/** Native three-model distraction classifier used by Android shadow mode. */
class RelevanceOnnxModel(context: Context) : AutoCloseable {

    data class Prediction(
        val tier: Int,
        val rawAction: String,
        val action: String,
        val confidence: Float,
        val cosineSimilarity: Float,
        val probabilities: FloatArray,
    )

    private val environment = OrtEnvironment.getEnvironment()
    private val featureBuilder = RelevanceFeatureBuilder(context.applicationContext)

    private val logisticSession = createSession(
        context,
        "distraction_classifier/logistic_regression.onnx",
    )
    private val xgboostSession = createSession(
        context,
        "distraction_classifier/xgboost.onnx",
    )
    private val xgboostTunedSession = createSession(
        context,
        "distraction_classifier/xgboost_tuned.onnx",
    )

    private fun createSession(context: Context, assetPath: String): OrtSession {
        val file = OnnxAssetLoader.materialize(context.applicationContext, assetPath)
        return environment.createSession(file.absolutePath, OrtSession.SessionOptions())
    }

    fun predict(
        focusIntention: String,
        tabTitle: String,
        currentTask: String = "",
        urlKeywords: String = "",
        url: String = "",
    ): Prediction {
        require(focusIntention.isNotBlank() || tabTitle.isNotBlank()) {
            "Provide at least focusIntention or tabTitle"
        }

        val featureResult = featureBuilder.buildFeatures(
            focusIntention = focusIntention,
            tabTitle = tabTitle,
            currentTask = currentTask,
            urlKeywords = urlKeywords,
            url = url,
        )
        return predictFromFeatures(featureResult.features, featureResult.cosineSimilarity)
    }

    fun predictFromFeatures(features: FloatArray, cosineSimilarity: Float = 0f): Prediction {
        require(features.size == 5545) {
            "Expected 5545 features, got ${features.size}"
        }

        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(features),
            longArrayOf(1, features.size.toLong()),
        ).use { tensor ->
            val logistic = runModel(logisticSession, tensor)
            val xgboost = runModel(xgboostSession, tensor)
            val tuned = runModel(xgboostTunedSession, tensor)

            // Original Python VotingClassifier weights = [1, 2, 2].
            val probabilities = FloatArray(3)
            for (i in probabilities.indices) {
                probabilities[i] = (logistic[i] + 2f * xgboost[i] + 2f * tuned[i]) / 5f
            }

            val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 1
            val confidence = probabilities[predictedClass]
            val rawAction = when (predictedClass) {
                0 -> "BLOCK"
                2 -> "ALLOW"
                else -> "CAUTION"
            }
            // Android shadow mode treats low confidence as caution instead of taking a hard action.
            val action = if (confidence < 0.70f) "CAUTION" else rawAction

            return Prediction(
                tier = predictedClass,
                rawAction = rawAction,
                action = action,
                confidence = confidence,
                cosineSimilarity = cosineSimilarity,
                probabilities = probabilities,
            )
        }
    }

    private fun runModel(session: OrtSession, tensor: OnnxTensor): FloatArray {
        val inputName = session.inputNames.firstOrNull() ?: "input"
        session.run(mapOf(inputName to tensor)).use { result ->
            val probabilityValue: OnnxValue = result.get("probabilities").orElse(null) ?: result[1]
            return decodeProbabilities(probabilityValue.value)
        }
    }

    private fun decodeProbabilities(value: Any?): FloatArray {
        if (value is Array<*>) {
            val first = value.firstOrNull()
            when (first) {
                is FloatArray -> return first.copyOf()
                is DoubleArray -> return FloatArray(first.size) { first[it].toFloat() }
                is Map<*, *> -> return decodeProbabilityMap(first)
            }
        }
        if (value is Map<*, *>) return decodeProbabilityMap(value)
        throw IllegalStateException("Unsupported ONNX probability output: ${value?.javaClass}")
    }

    private fun decodeProbabilityMap(map: Map<*, *>): FloatArray {
        val output = FloatArray(3)
        for ((key, value) in map) {
            val index = when (key) {
                is Number -> key.toInt()
                else -> key?.toString()?.toIntOrNull()
            } ?: continue
            if (index in output.indices && value is Number) {
                output[index] = value.toFloat()
            }
        }
        return output
    }

    override fun close() {
        featureBuilder.close()
        logisticSession.close()
        xgboostSession.close()
        xgboostTunedSession.close()
    }
}
