"""
* Program: extract_party_sentences.py
*
* Modified Date: November 2025
*
* Purpose: Step 3a of Sentiment Pipeline
*    Extract targeted political sentences from each Reddit comment.
*    For every comment:
*        - Build full_text = title + body
*        - Split into sentences and match them with party-specific keywords
*        - Collect and merge all matched sentences per party
*    Output one row per (comment, party, targeted_sentence),
*    which is later used for Transformer and VADER sentiment scoring.
*
"""

import re
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import MapType, StringType

# -------------------------------
# 0. Parse arguments
# -------------------------------
if len(sys.argv) != 3:
    print("Usage: spark-submit extract_party_sentences.py <input_dir> <output_dir>")
    sys.exit(1)

INPUT_DIR = sys.argv[1]     # e.g., results/joined_rdd/2025-*
OUTPUT_DIR = sys.argv[2]    # e.g., results/party_target_updated_keywords

print("INPUT_DIR :", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

spark = SparkSession.builder.appName("Extract Party Sentences").getOrCreate()

# ---------------------------------------------------------
# 1. Keyword dictionaries
# ---------------------------------------------------------

PARTY_KEYWORDS = {
    "Liberal": [
        "trudeau","justin trudeau","liberal","liberals",
        "lpc","carney","freeland","chrystia freeland"
    ],
    "Conservative": [
        "poilievre","pierre poilievre","conservative",
        "conservatives","cpc","tory","tories",
        "trump","maga", "right wing", "right-wing", "right populist",
        "far right", "lower taxes","inflation", "pp",
    ],
    "NDP": [
        "ndp","new democratic","singh","jagmeet","jagmeet singh",
        "orange crush", "progressive left", "left wing", "left-wing"
    ]
}

# ---------------------------------------------------------
# 2. Extract ALL sentences per party → merge them into one string
# ---------------------------------------------------------

def extract_party_text(text):
    """
    Return:
      {party: "merged sentences"}
    Each party appears at most once, even if multiple sentences matched.
    """
    if text is None:
        return {}

    text = text.lower()
    sentences = re.split(r'[.!?]', text)

    output = {}

    for party, kw_list in PARTY_KEYWORDS.items():
        collected = []

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) == 0:
                continue

            # check keyword hits
            for kw in kw_list:
                if re.search(rf'\b{kw}\b', s_clean):
                    collected.append(s_clean)
                    break

        # merge all matched sentences into one text
        if len(collected) > 0:
            merged = ". ".join(collected)
            output[party] = merged

    return output

extract_udf = F.udf(extract_party_text, MapType(StringType(), StringType()))

# ---------------------------------------------------------
# 3. Load data
# ---------------------------------------------------------
# IMPORTANT NOTES:
#    Input path is HARD-CODED and must be manually changed before running:
#df = spark.read.json("joined_rdd/2025-*")
df = spark.read.json(INPUT_DIR)

df = df.withColumn(
    "full_text",
    F.concat_ws(" ", F.col("title"), F.col("body"))
)

df = df.withColumn("party_text", extract_udf("full_text"))

# ---------------------------------------------------------
# 4. Explode (one row per comment & party)
# ---------------------------------------------------------

df_exploded = df.select(
    "*",
    F.explode("party_text").alias("party", "targeted_sentence")
).drop("party_text")

df_final = (
    df_exploded
    .withColumn("date", F.from_unixtime("created_utc").cast("date"))
    .withColumn("week", F.weekofyear("date"))
)

# ---------------------------------------------------------
# 5. Save
# ---------------------------------------------------------
# IMPORTANT NOTES:
#    Output path is HARD-CODED and must be manually changed before running:
#df_final.write.mode("overwrite").json("results/party_target_ updated_keywords")
df_final.write.mode("overwrite").json(OUTPUT_DIR)
#print("Saved → results/party_target_updated_keywords")
print(f"Saved → {OUTPUT_DIR}")