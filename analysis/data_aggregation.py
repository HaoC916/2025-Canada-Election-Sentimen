"""
* Program: data_aggregation.py
*
* Modified Date: November 2025
*
* Purpose: Step 4a of Sentiment Analysis Pipeline
*    Aggregate Transformer + VADER sentiment into weekly and daily
*    time series for later correlation and regression analysis.
*
*    Input:
*        - JSON lines containing:
*            party, date, week,
*            trans_sentiment (0/1/2),
*            vader_score (float)
*
*    Processing:
*        - Convert transformer class → indicator columns (pos/neu/neg)
*        - Compute weekly:
*            * avg VADER score
*            * positive / neutral / negative ratios
*            * comment volume
*        - Compute daily:
*            * same metrics but grouped by date
*
*    Output:
*        - WEEKLY results written to:
*              results/sentiment_weekly_updated_key
*        - DAILY results written to:
*              results/sentiment_daily_updated_key
"""

# =============================================================
# Weekly Aggregation for Transformer + VADER Sentiment
# =============================================================

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("Weekly Sentiment Aggregation").getOrCreate()

# Important Notes:
#    Input path is HARD-CODED and must be manually changed before running:
df = spark.read.json("data/vader_tran_scored_sample.json")

df = df.filter(F.col("party").isNotNull())

# -------------------------------------------------------------
# Convert transformer class → indicator columns
#     0 = negative, 1 = neutral, 2 = positive
# -------------------------------------------------------------
df = (
    df.withColumn("is_neg", F.when(F.col("trans_sentiment") == 0, 1).otherwise(0))
      .withColumn("is_neu", F.when(F.col("trans_sentiment") == 1, 1).otherwise(0))
      .withColumn("is_pos", F.when(F.col("trans_sentiment") == 2, 1).otherwise(0))
)

# -------------------------------------------------------------
# Weekly aggregation (per party, per week)
# -------------------------------------------------------------
weekly = (
    df.groupBy("party", "week")
      .agg(
          F.avg("vader_score").alias("vader_avg"),
          F.sum("is_pos").alias("pos_count"),
          F.sum("is_neg").alias("neg_count"),
          F.sum("is_neu").alias("neu_count"),
          F.count("*").alias("comment_volume")
      )
)

# -------------------------------------------------------------
# Convert raw counts → ratios
# -------------------------------------------------------------
weekly = weekly.withColumn(
    "total",
    F.col("pos_count") + F.col("neg_count") + F.col("neu_count")
)

weekly = (
    weekly.withColumn("trans_pos_ratio", F.col("pos_count") / F.col("total"))
          .withColumn("trans_neg_ratio", F.col("neg_count") / F.col("total"))
          .withColumn("trans_neu_ratio", F.col("neu_count") / F.col("total"))
)

weekly = weekly.orderBy("party", "week")


weekly.write.mode("overwrite").json("results/sentiment_weekly_updated_key")

print("✓ Saved WEEKLY sentiment → results/sentiment_weekly_updated_key")

# -------------------------------------------------------------
# Daily aggregation (per party, per week)
# -------------------------------------------------------------
daily = (
    df.groupBy("party", "date")
      .agg(
          F.avg("vader_score").alias("vader_avg"),
          F.sum("is_pos").alias("pos_count"),
          F.sum("is_neg").alias("neg_count"),
          F.sum("is_neu").alias("neu_count"),
          F.count("*").alias("comment_volume")
      )
)

# -------------------------------------------------------------
# Convert raw counts → ratios
# -------------------------------------------------------------
daily = daily.withColumn(
    "total",
    F.col("pos_count") + F.col("neg_count") + F.col("neu_count")
)

daily = (
    daily.withColumn("trans_pos_ratio", F.col("pos_count") / F.col("total"))
          .withColumn("trans_neg_ratio", F.col("neg_count") / F.col("total"))
          .withColumn("trans_neu_ratio", F.col("neu_count") / F.col("total"))
)

daily = daily.orderBy("party", "date")

# Important Notes:
#    output path is HARD-CODED and must be manually changed before running:
daily.write.mode("overwrite").json("results/sentiment_daily_updated_key")

print("✓ Saved DAILY sentiment → results/sentiment_daily_updated_key")