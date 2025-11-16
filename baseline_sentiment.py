# ---------------------------------------------------------
# CMPT 732 Final Project — Baseline Sentiment Analysis
# Using VADER sentiment on cleaned Reddit comments
# ---------------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------
# 1. Spark Session
# ---------------------------------------------------------
spark = SparkSession.builder \
    .appName("RedditBaselineSentiment") \
    .getOrCreate()

# ---------------------------------------------------------
# 2. Load cleaned comments dataset
# (Replace the path with your actual clean data file)
# ---------------------------------------------------------
df = spark.read.json("clean_comments.json")   # or .parquet("clean_comments.parquet")

# Schema expectation:
# id, link_id_short, parent_id, subreddit, author,
# created_utc, score, body, permalink, title

print("Loaded records:", df.count())

# ---------------------------------------------------------
# 3. Register VADER UDF
# ---------------------------------------------------------

analyzer = SentimentIntensityAnalyzer()

def vader_score(text):
    if text is None:
        return 0.0
    try:
        return float(analyzer.polarity_scores(text)["compound"])
    except:
        return 0.0

vader_udf = F.udf(vader_score, FloatType())

# ---------------------------------------------------------
# 4. Apply VADER to compute sentiment
# ---------------------------------------------------------

df_sent = df.withColumn("sentiment", vader_udf(F.col("body")))

# ---------------------------------------------------------
# 5. Convert time → date, week-of-year
# ---------------------------------------------------------

df_sent = (
    df_sent
    .withColumn("date", F.from_unixtime("created_utc"))
    .withColumn("week", F.weekofyear("date"))
    .withColumn("year", F.year("date"))
)

# ---------------------------------------------------------
# 6. Party tagging from title keywords
# ---------------------------------------------------------

def detect_party(title):
    if title is None:
        return "Other"
    t = title.lower()
    if "trudeau" in t or "liberal" in t or "lpc" in t:
        return "Liberal"
    if "poilievre" in t or "conservative" in t or "cpc" in t:
        return "Conservative"
    if "ndp" in t:
        return "NDP"
    return "Other"

party_udf = F.udf(detect_party, StringType())

df_tagged = df_sent.withColumn("party", party_udf("title"))

# ---------------------------------------------------------
# 7. Weekly sentiment per subreddit
# ---------------------------------------------------------

weekly_subreddit = (
    df_tagged
    .groupBy("subreddit", "year", "week")
    .agg(
        F.avg("sentiment").alias("avg_sentiment"),
        F.count("*").alias("num_comments")
    )
    .orderBy("subreddit", "year", "week")
)

# ---------------------------------------------------------
# 8. Weekly sentiment per party
# ---------------------------------------------------------

weekly_party = (
    df_tagged
    .groupBy("party", "year", "week")
    .agg(
        F.avg("sentiment").alias("avg_sentiment"),
        F.count("*").alias("num_comments")
    )
    .orderBy("party", "year", "week")
)

# ---------------------------------------------------------
# 9. Save outputs (for visualization & report)
# ---------------------------------------------------------

weekly_subreddit.write.mode("overwrite").json("output/weekly_subreddit_sentiment")
weekly_party.write.mode("overwrite").json("output/weekly_party_sentiment")

print("Sentiment analysis complete!")
print("Outputs saved under: ./output/")
spark.stop()