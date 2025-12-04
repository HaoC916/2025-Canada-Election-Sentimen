"""
* Program: baseline_vader.py
*
* Modified Date: November 2025
*
* Purpose: Step 3c of Sentiment Pipeline
*    Apply baseline VADER sentiment scoring to the
*    targeted political sentences extracted in Step 3a.
*
*    For each (comment, party):
*        - Load `targeted_sentence`
*        - Run VADER to compute the compound score in [-1, 1]
*        - Append output as `vader_score`
*        - Save enriched dataset for later weekly aggregation
*
*    VADER is lexicon-based and fast, suitable as a baseline model.
*    This step allows comparison with Transformer sentiment quality.
*
"""

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
# IMPORTANT NOTES:
# The input path is currently HARD-CODED and MUST be modified before running
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
# IMPORTANT NOTES:
# The output path is currently HARD-CODED and MUST be modified before running
# ---------------------------------------------------------

df_scored.write.mode("overwrite").json("results/vader_tran_scored")

print("Saved targeted VADER sentiment pipeline output → results/vader_tran_scored")