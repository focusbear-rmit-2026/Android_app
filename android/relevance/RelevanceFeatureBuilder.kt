package com.focusbear.relevance

import android.content.Context
import org.json.JSONObject
import java.net.URI
import java.util.Locale
import kotlin.math.ln
import kotlin.math.sqrt

/** Reproduces the Python deployment/predictor.py feature order for Android. */
class RelevanceFeatureBuilder(private val context: Context) : AutoCloseable {

    data class FeatureBuildResult(
        val features: FloatArray,
        val cosineSimilarity: Float,
        val urlKeywords: String,
    )

    private data class TfidfData(
        val vocabulary: Map<String, Int>,
        val idf: FloatArray,
    )

    private data class UrlMetadata(
        val pathDepth: Float,
        val urlLength: Float,
        val hasSubdomain: Float,
        val hasQuery: Float,
    )

    private val miniLm = MiniLmEmbedder(context)
    private val goalTfidf = loadTfidf("distraction_classifier/tfidf_goal.json")
    private val contextTfidf = loadTfidf("distraction_classifier/tfidf_context.json")

    private val feedKeywords = listOf(
        "feed",
        "home",
        "timeline",
        "messages",
        "inbox",
        "notifications",
        "trending",
        "dashboard",
        "news feed",
        "social",
    )

    fun buildFeatures(
        focusIntention: String,
        tabTitle: String,
        currentTask: String = "",
        urlKeywords: String = "",
        url: String = "",
    ): FeatureBuildResult {
        val userGoal = joinNonBlank(focusIntention, currentTask)
        val effectiveUrlKeywords = urlKeywords.ifBlank { extractUrlPathKeywords(url) }
        val websiteContext = joinNonBlank(tabTitle, effectiveUrlKeywords)

        val goalFeatures = createTfidfVector(userGoal, goalTfidf)
        val contextFeatures = createTfidfVector(websiteContext, contextTfidf)

        val embeddings = arrayOf(miniLm.encode(userGoal), miniLm.encode(websiteContext))
        val cosine = miniLm.cosineSimilarity(embeddings[0], embeddings[1])

        val metadata = getUrlMetadata(url)
        val isFeedOrMessaging = if (
            feedKeywords.any { tabTitle.lowercase(Locale.ROOT).contains(it) }
        ) 1f else 0f
        val tabTitleMissing = if (tabTitle.isBlank()) 1f else 0f

        val goalWords = whitespaceWords(userGoal)
        val contextWords = whitespaceWords(websiteContext)
        val goalSet = goalWords.toSet()
        val contextSet = contextWords.toSet()
        val overlap = if (goalSet.isEmpty()) {
            0f
        } else {
            goalSet.intersect(contextSet).size.toFloat() / goalSet.size.toFloat()
        }

        val features = FloatArray(5545)
        var position = 0

        for (value in goalFeatures) features[position++] = value
        for (value in contextFeatures) features[position++] = value
        features[position++] = cosine
        features[position++] = metadata.pathDepth
        features[position++] = metadata.urlLength
        features[position++] = metadata.hasSubdomain
        features[position++] = metadata.hasQuery
        features[position++] = isFeedOrMessaging
        features[position++] = tabTitleMissing
        features[position++] = overlap
        features[position++] = goalWords.size.toFloat()
        features[position++] = contextWords.size.toFloat()
        features[position++] = cosine * overlap
        features[position++] = cosine * isFeedOrMessaging

        check(position == features.size) {
            "Feature construction error: built $position features instead of ${features.size}"
        }

        return FeatureBuildResult(features, cosine, effectiveUrlKeywords)
    }

    private fun joinNonBlank(first: String, second: String): String =
        listOf(first, second).map { it.trim() }.filter { it.isNotEmpty() }.joinToString(" ")

    private fun whitespaceWords(text: String): List<String> =
        text.trim()
            .takeIf { it.isNotEmpty() }
            ?.lowercase(Locale.ROOT)
            ?.split(Regex("\\s+"))
            ?: emptyList()

    private fun loadTfidf(path: String): TfidfData {
        val jsonText = context.assets.open(path).bufferedReader(Charsets.UTF_8).use { it.readText() }
        val json = JSONObject(jsonText)
        val vocabJson = json.getJSONObject("vocabulary")
        val vocabulary = HashMap<String, Int>(vocabJson.length())
        val keys = vocabJson.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            vocabulary[key] = vocabJson.getInt(key)
        }

        val idfJson = json.getJSONArray("idf")
        val idf = FloatArray(idfJson.length())
        for (i in 0 until idfJson.length()) {
            idf[i] = idfJson.getDouble(i).toFloat()
        }
        return TfidfData(vocabulary, idf)
    }

    private fun createTfidfVector(text: String, data: TfidfData): FloatArray {
        val output = FloatArray(data.idf.size)
        val tokens = tfidfTokens(text)
        if (tokens.isEmpty()) return output

        val counts = HashMap<String, Int>()
        fun countTerm(term: String) {
            if (data.vocabulary.containsKey(term)) {
                counts[term] = (counts[term] ?: 0) + 1
            }
        }

        tokens.forEach(::countTerm)
        for (i in 0 until tokens.size - 1) {
            countTerm("${tokens[i]} ${tokens[i + 1]}")
        }

        for ((term, count) in counts) {
            val index = data.vocabulary[term] ?: continue
            val termFrequency = 1.0 + ln(count.toDouble()) // sklearn sublinear_tf=True
            output[index] = (termFrequency * data.idf[index]).toFloat()
        }

        var sumSquares = 0.0
        for (value in output) sumSquares += value.toDouble() * value.toDouble()
        val norm = sqrt(sumSquares)
        if (norm > 0.0) {
            for (i in output.indices) output[i] = (output[i] / norm).toFloat()
        }
        return output
    }

    private fun tfidfTokens(text: String): List<String> {
        // Python vectorizer: lowercase=True, ngram_range=(1,2), token_pattern=(?u)\\b\\w\\w+\\b
        val regex = Regex("[\\p{L}\\p{N}_]{2,}")
        return regex.findAll(text.lowercase(Locale.ROOT)).map { it.value }.toList()
    }

    private fun extractUrlPathKeywords(rawUrl: String): String {
        if (rawUrl.isBlank()) return ""
        return try {
            val prepared = if (rawUrl.startsWith("http")) rawUrl else "https://$rawUrl"
            val path = URI(prepared).rawPath.orEmpty()
            path.split("/")
                .filter { it.isNotBlank() && it.length > 2 && !it.all(Char::isDigit) }
                .joinToString(" ")
        } catch (_: Exception) {
            ""
        }
    }

    private fun getUrlMetadata(rawUrl: String): UrlMetadata {
        if (rawUrl.isBlank()) return UrlMetadata(0f, 0f, 0f, 0f)

        return try {
            val prepared = if (rawUrl.startsWith("http")) rawUrl else "https://$rawUrl"
            val uri = URI(prepared)
            val authority = uri.rawAuthority.orEmpty()
            val hostParts = authority.split(".")
            val hasSubdomain = if (
                hostParts.size > 2 && hostParts.firstOrNull() !in listOf("www", "")
            ) 1f else 0f
            val pathDepth = uri.rawPath.orEmpty().split("/").count { it.isNotBlank() }.toFloat()
            val hasQuery = if (!uri.rawQuery.isNullOrEmpty()) 1f else 0f

            UrlMetadata(
                pathDepth = pathDepth,
                urlLength = prepared.length.toFloat(),
                hasSubdomain = hasSubdomain,
                hasQuery = hasQuery,
            )
        } catch (_: Exception) {
            UrlMetadata(0f, 0f, 0f, 0f)
        }
    }

    override fun close() {
        miniLm.close()
    }
}

