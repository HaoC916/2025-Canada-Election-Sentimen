from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("Weekly Sentiment Aggregation").getOrCreate()

# Load your VADER sentiment output
df = spark.read.json("results/vader_targeted")

# Weighted sentiment: sentiment * log(1 + score)
df = df.withColumn(
    "weighted_sent",
    F.col("sentiment") * F.log(1 + F.col("score"))
)

# Weekly aggregation
df_weekly = (
    df.groupBy("party", "week")
      .agg(
          F.avg("sentiment").alias("avg_sentiment"),
          F.avg("weighted_sent").alias("weighted_sentiment"),
          F.count("*").alias("comment_volume")
      )
      .orderBy("party", "week")
)

df_weekly.write.mode("overwrite").json("results/weekly_sentiment")
print("Saved → results/weekly_sentiment")