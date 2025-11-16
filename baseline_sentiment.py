import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    MapType, StringType, FloatType
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------
# 0. Initialize Spark + VADER
# ---------------------------------------------------------

spark = SparkSession.builder.appName("VADER Targeted Sentiment").getOrCreate()

analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------
# 1. Define Party Keyword Dictionaries
# ---------------------------------------------------------

LIBERAL_KW = [
    "trudeau", "justin trudeau", "liberal", "liberals",
    "lpc", "carney", "freeland", "chrystia freeland"
]

CONSERVATIVE_KW = [
    "poilievre", "pierre poilievre", "conservative",
    "conservatives", "cpc", "tory", "tories"
]

NDP_KW = [
    "ndp", "new democratic", "singh", "jagmeet", "jagmeet singh"
]

PARTY_KEYWORDS = {
    "Liberal": LIBERAL_KW,
    "Conservative": CONSERVATIVE_KW,
    "NDP": NDP_KW
}


# ---------------------------------------------------------
# 2. Targeted VADER Function (Python)
# ---------------------------------------------------------

def targeted_vader_sentiment(text):
    """
    Given full comment text, detect party keywords, extract
    sentence windows for each party, and compute separate
    VADER sentiment scores per party.
    """
    if text is None:
        return {}

    text = text.lower()

    # Split into rough sentences
    sentences = re.split(r'[.!?]', text)

    result = {}

    for party, keywords in PARTY_KEYWORDS.items():

        # Collect all sentences mentioning this party
        party_sents = []
        for s in sentences:
            for k in keywords:
                pattern = rf'\b{k}\b'
                if re.search(pattern, s):
                    party_sents.append(s.strip())
                    break  # avoid double-adding same sentence

        if len(party_sents) == 0:
            continue
        
        party_text = " ".join(party_sents).strip()

        if len(party_text) > 0:
            score = analyzer.polarity_scores(party_text)['compound']
            result[party] = float(score)

    return result


# ---------------------------------------------------------
# 3. Register Spark UDF
# ---------------------------------------------------------

sentiment_udf = F.udf(
    targeted_vader_sentiment,
    MapType(StringType(), FloatType())
)


# ---------------------------------------------------------
# 4. Load Your Joined Dataset
# joined_rdd/2025-01, joined_rdd/2025-02, ... for each month
# joined_rdd/2025-* for all months
# ---------------------------------------------------------

df = spark.read.json("joined_rdd/2025-01")

# Combine title + body for better context
df = df.withColumn(
    "full_text",
    F.concat_ws(" ", F.col("title"), F.col("body"))
)


# ---------------------------------------------------------
# 5. Apply Targeted VADER per Comment
# ---------------------------------------------------------

df_with_sentiment = df.withColumn(
    "party_sentiment",
    sentiment_udf("full_text")
)


# ---------------------------------------------------------
# 6. Explode to One Row per (comment, party)
# ---------------------------------------------------------

df_exploded = (
    df_with_sentiment
    .select(
        "*",
        F.explode("party_sentiment").alias("party", "sentiment")
    )
    .drop("party_sentiment")
)


# ---------------------------------------------------------
# 7. Add Date / Week for Time-Series Aggregation
# ---------------------------------------------------------

df_final = (
    df_exploded
    .withColumn("date", F.from_unixtime("created_utc").cast("date"))
    .withColumn("week", F.weekofyear("date"))
)


# ---------------------------------------------------------
# 8. Save Final Output
# ---------------------------------------------------------

df_final.write.mode("overwrite").json("results/vader_targeted")

print("Saved targeted VADER sentiment pipeline output → results/vader_targeted")