
# ---------------------------------------------------------
# Compute daily sentiment from extracted targeted sentences

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("Daily Sentiment Aggregation - Vader").getOrCreate()

df = spark.read.json("results/trans_vader_scored")

# Ensure correct types
df = df.filter(F.col("party").isNotNull())

# ---------------- Daily aggregation for vader on average score ----------------
daily_v = (
    df.groupBy("party", "date")
      .agg(
          F.avg("vader_score").alias("avg_sentiment"),
          F.count("*").alias("comment_volume")
      )
      .orderBy("party", "date")
)

# ---------------- Daily aggregation for transformer on ratio of positive ----------------
daily_t = (
    df.groupBy("party", "date")
      .agg(
          F.count("trans_sentiment").alias("avg_sentiment"),
          F.count("*").alias("comment_volume")
      )
      .orderBy("party", "date")
)

daily.write.mode("overwrite").json("results/vader_sentiment_daily")

print("✓ Saved DAILY sentiment → results/vader_sentiment_daily")