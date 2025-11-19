
# ---------------------------------------------------------
# Compute daily sentiment from extracted targeted sentences

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("Daily Sentiment Aggregation - Vader").getOrCreate()

df = spark.read.json("results/vader_scored")

# Ensure correct types
df = df.filter(F.col("party").isNotNull())

# ---------------- Daily aggregation ----------------
daily = (
    df.groupBy("party", "date")
      .agg(
          F.avg("vader_score").alias("avg_sentiment"),
          F.count("*").alias("comment_volume")
      )
      .orderBy("party", "date")
)

daily.write.mode("overwrite").json("results/vader_sentiment_daily")

print("✓ Saved DAILY sentiment → results/vader_sentiment_daily")