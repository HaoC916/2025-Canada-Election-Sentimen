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

spark = SparkSession.builder.appName("Baseline VADER Sentiment Score").getOrCreate()

analyzer = SentimentIntensityAnalyzer()

# ------------------------------------------------
# 1. Load extracted targeted-sentence dataset
# ------------------------------------------------
df = spark.read.json("results/tran_scored_updated_keywords")  
# expected columns: id, party, targeted_sentence, date, week

# ------------------------------------------------
# 2. UDF: run VADER on extracted_text
# ------------------------------------------------
def vader_score(text):
    if text is None:
        return 0.0
    return float(analyzer.polarity_scores(text)["compound"])

vader_udf = F.udf(vader_score, FloatType())

df_scored = df.withColumn("vader_score", vader_udf("targeted_sentence")).drop("full_text")

# ---------------------------------------------------------
# 3. Save Final Output
# ---------------------------------------------------------

df_scored.write.mode("overwrite").json("results/vader_tran_scored")

print("Saved targeted VADER sentiment pipeline output → results/vader_tran_scored")