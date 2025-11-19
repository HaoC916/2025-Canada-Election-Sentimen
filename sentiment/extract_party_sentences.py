import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import MapType, StringType

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
        "conservatives","cpc","tory","tories"
    ],
    "NDP": [
        "ndp","new democratic","singh","jagmeet","jagmeet singh"
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

df = spark.read.json("joined_rdd/2025-*")

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

df_final.write.mode("overwrite").json("results/party_target")

print("Saved → results/party_target")