# ---------------------------------------------------------
# Compute weekly sentiment from extracted targeted sentences

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("Weekly Sentiment Aggregation - Vader").getOrCreate()

df = spark.read.json("results/vader_scored")

df = df.filter(F.col("party").isNotNull())

# ---------------- Weekly aggregation ----------------
weekly = (
    df.groupBy("party", "week")
      .agg(
          F.avg("vader_score").alias("avg_sentiment"),
          F.count("*").alias("comment_volume")
      )
      .orderBy("party", "week")
)

weekly.write.mode("overwrite").json("results/vader_sentiment_weekly")

print("✓ Saved WEEKLY sentiment → results/vader_sentiment_weekly")