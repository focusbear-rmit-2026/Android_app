package com.focusbear.relevance

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.LongBuffer
import java.text.Normalizer
import java.util.Locale
import kotlin.math.sqrt

/**
 * Small Android wrapper around sentence-transformers/all-MiniLM-L6-v2 exported to ONNX.
 * The ONNX file returns token embeddings, so SentenceTransformer-style mean pooling is
 * reproduced locally before cosine similarity is calculated.
 */
class MiniLmEmbedder(private val context: Context) : AutoCloseable {

    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val vocabulary: Map<String, Int>

    private val maxLength = 256
    private val clsId: Int
    private val sepId: Int
    private val padId: Int
    private val unkId: Int

    init {
        val modelFile = OnnxAssetLoader.materialize(
            context,
            "distraction_classifier/minilm/model.onnx",
        )
        session = environment.createSession(modelFile.absolutePath, OrtSession.SessionOptions())

        vocabulary = loadVocabulary()
        clsId = vocabulary["[CLS]"] ?: 101
        sepId = vocabulary["[SEP]"] ?: 102
        padId = vocabulary["[PAD]"] ?: 0
        unkId = vocabulary["[UNK]"] ?: 100
    }

    private fun loadVocabulary(): Map<String, Int> =
        context.assets
            .open("distraction_classifier/minilm/vocab.txt")
            .bufferedReader(Charsets.UTF_8)
            .useLines { lines ->
                lines.mapIndexed { index, token -> token.trimEnd() to index }.toMap()
            }

    fun encode(text: String): FloatArray {
        val tokenIds = tokenize(text)
        val inputIds = LongArray(maxLength) { padId.toLong() }
        val attentionMask = LongArray(maxLength)
        val tokenTypeIds = LongArray(maxLength)

        tokenIds.forEachIndexed { index, id ->
            inputIds[index] = id.toLong()
            attentionMask[index] = 1L
        }

        OnnxTensor.createTensor(
            environment,
            LongBuffer.wrap(inputIds),
            longArrayOf(1, maxLength.toLong()),
        ).use { idsTensor ->
            OnnxTensor.createTensor(
                environment,
                LongBuffer.wrap(attentionMask),
                longArrayOf(1, maxLength.toLong()),
            ).use { maskTensor ->
                OnnxTensor.createTensor(
                    environment,
                    LongBuffer.wrap(tokenTypeIds),
                    longArrayOf(1, maxLength.toLong()),
                ).use { typeTensor ->
                    val inputs = mapOf(
                        "input_ids" to idsTensor,
                        "attention_mask" to maskTensor,
                        "token_type_ids" to typeTensor,
                    )

                    session.run(inputs).use { result ->
                        @Suppress("UNCHECKED_CAST")
                        val batch = result[0].value as Array<Array<FloatArray>>
                        return meanPool(batch[0], attentionMask)
                    }
                }
            }
        }
    }

    private fun meanPool(hiddenStates: Array<FloatArray>, attentionMask: LongArray): FloatArray {
        if (hiddenStates.isEmpty()) return FloatArray(384)

        val embeddingSize = hiddenStates[0].size
        val embedding = FloatArray(embeddingSize)
        var validTokens = 0f

        for (i in hiddenStates.indices) {
            if (i >= attentionMask.size || attentionMask[i] == 0L) continue
            validTokens += 1f
            for (j in 0 until embeddingSize) {
                embedding[j] += hiddenStates[i][j]
            }
        }

        if (validTokens > 0f) {
            for (j in embedding.indices) {
                embedding[j] /= validTokens
            }
        }
        return embedding
    }

    fun cosineSimilarity(first: FloatArray, second: FloatArray): Float {
        if (first.size != second.size) return 0f

        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in first.indices) {
            val a = first[i].toDouble()
            val b = second[i].toDouble()
            dot += a * b
            normA += a * a
            normB += b * b
        }
        if (normA == 0.0 || normB == 0.0) return 0f
        return (dot / (sqrt(normA) * sqrt(normB))).toFloat()
    }

    private fun tokenize(text: String): IntArray {
        val pieces = mutableListOf<Int>()
        pieces += clsId

        for (token in basicTokenize(text)) {
            if (pieces.size >= maxLength - 1) break
            for (pieceId in wordPieceTokenize(token)) {
                if (pieces.size >= maxLength - 1) break
                pieces += pieceId
            }
        }

        pieces += sepId
        return pieces.toIntArray()
    }

    /** Approximation of the uncased BERT basic tokenizer used by MiniLM. */
    private fun basicTokenize(text: String): List<String> {
        val lowered = text.lowercase(Locale.ROOT)
        val normalized = Normalizer.normalize(lowered, Normalizer.Form.NFD)
        val withoutAccents = buildString {
            for (char in normalized) {
                if (Character.getType(char) != Character.NON_SPACING_MARK.toInt()) {
                    append(char)
                }
            }
        }

        val spaced = buildString {
            for (char in withoutAccents) {
                when {
                    char.isWhitespace() -> append(' ')
                    char.isLetterOrDigit() || char == '_' -> append(char)
                    Character.isISOControl(char) -> Unit
                    else -> {
                        append(' ')
                        append(char)
                        append(' ')
                    }
                }
            }
        }

        return spaced.trim().split(Regex("\\s+")).filter { it.isNotBlank() }
    }

    private fun wordPieceTokenize(token: String): List<Int> {
        if (token.length > 100) return listOf(unkId)

        val result = mutableListOf<Int>()
        var start = 0
        while (start < token.length) {
            var end = token.length
            var found: Int? = null
            var nextStart = start

            while (end > start) {
                var piece = token.substring(start, end)
                if (start > 0) piece = "##$piece"
                val id = vocabulary[piece]
                if (id != null) {
                    found = id
                    nextStart = end
                    break
                }
                end--
            }

            if (found == null) return listOf(unkId)
            result += found
            start = nextStart
        }
        return result
    }

    override fun close() {
        session.close()
    }
}
