import re
import torch
from transformers import pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType, FloatType, MapType
)

# ---------------------------------------------------------
# 0. Init Spark
# ---------------------------------------------------------
spark = SparkSession.builder \
    .appName("ZeroShot Targeted Sentiment") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------
# 1. Load Zero-Shot Model (CPU Safe)
# ---------------------------------------------------------
print("Loading Zero-Shot NLI model… (may take 20s)")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# 3 political parties
PARTIES = ["Liberal", "Conservative", "NDP"]


# ---------------------------------------------------------
# 2. Zero-Shot targeted function (per sentence)
# ---------------------------------------------------------
def zero_shot_targeted(text):
    """
    For a full Reddit comment:
    Ask: “Sentiment toward Liberal / Conservative / NDP”
    Output: {party: score}
    """
    if text is None or len(text.strip()) == 0:
        return {}

    text = text.strip()

    result = {}

    for p in PARTIES:
        hypotheses = [
            f"The sentiment toward the {p} Party is positive.",
            f"The sentiment toward the {p} Party is negative.",
            f"The sentiment toward the {p} Party is neutral."
        ]

        out = classifier(
            text,
            candidate_labels=hypotheses,
            multi_label=False
        )

        label = out["labels"][0]     # top hypothesis
        score = out["scores"][0]     # probability

        # Convert label → numeric score
        if "positive" in label:
            final = 1.0
        elif "negative" in label:
            final = -1.0
        else:
            final = 0.0

        result[p] = float(final)

    return result


# ---------------------------------------------------------
# 3. Register Spark UDF
# ---------------------------------------------------------
zero_udf = F.udf(zero_shot_targeted, MapType(StringType(), FloatType()))


# ---------------------------------------------------------
# 4. Load your dataset like before (folder OK)
# ---------------------------------------------------------
input_path = "joined_rdd/"

df = spark.read.json(input_path)
print("Loaded rows:", df.count())

# Combine title + body
df = df.withColumn(
    "full_text",
    F.concat_ws(" ", "title", "body")
)


# ---------------------------------------------------------
# 5. Run Zero-Shot targeted sentiment
# ---------------------------------------------------------
df_scored = df.withColumn(
    "zero_shot_sentiment",
    zero_udf("full_text")
)

# Explode to one row per (comment, party)
df_final = df_scored.select(
    "*",
    F.explode("zero_shot_sentiment").alias("party", "zero_shot_score")
).drop("zero_shot_sentiment")


# ---------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------
output_path = "results/zero_shot_spark"
df_final.write.mode("overwrite").json(output_path)

print("🔥 Saved zero-shot Spark results →", output_path)