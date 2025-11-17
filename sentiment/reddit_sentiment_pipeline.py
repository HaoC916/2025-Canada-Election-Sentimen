import os
import json
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ======================================================
# 0. CONFIGURATION
# ======================================================

BASE_FOLDER = r"E:\E\university\year6\cmpt732\cmpt732-project\joined_rdd"
OUTPUT_CSV = "weekly_sentiment.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
print("Loading model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    use_safetensors=True,        # FORCE SAFE LOADING
    torch_dtype="auto"
)

model.to(device)
model.eval()

id2label = model.config.id2label
print("Model labels:", id2label)


# ======================================================
# 1. CLEAN TEXT
# ======================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").strip()
    cleaned = []
    for tok in text.split():
        if tok.startswith("@") and len(tok) > 1:
            cleaned.append("@user")
        elif tok.startswith("http"):
            cleaned.append("http")
        else:
            cleaned.append(tok)
    return " ".join(cleaned)


# ======================================================
# 2. LOAD PART FILES (JSON LINES)
# ======================================================

def load_part_file(filepath):
    print("Reading:", filepath)
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line.strip()))
            except:
                pass
    return rows


all_rows = []

# List monthly folders like 2025-01, 2025-02, ...
month_folders = sorted(glob(os.path.join(BASE_FOLDER, "2025-*")))
print("Found folders:", month_folders)

for folder in month_folders:
    part_files = glob(os.path.join(folder, "part-*"))
    for part in part_files:
        rows = load_part_file(part)
        all_rows.extend(rows)

print("Total loaded rows:", len(all_rows))
df = pd.DataFrame(all_rows)
print(df.head())


# ======================================================
# 3. PREPROCESS TEXT + TIME
# ======================================================

df["clean_body"] = df["body"].apply(clean_text)

df["datetime"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
iso = df["datetime"].dt.isocalendar()
df["year"] = iso.year
df["week"] = iso.week


# ======================================================
# 4. TRANSFORMER SENTIMENT (BATCHED)
# ======================================================

def classify_sentiment(texts, batch_size=32):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=1)
            labels = torch.argmax(probs, dim=1).cpu().numpy()
            preds.extend(labels)
    return preds


print("Running sentiment analysis…")
df["sentiment_id"] = classify_sentiment(df["clean_body"].tolist())
df["sentiment"] = df["sentiment_id"].map(id2label)


# ======================================================
# 5. AGGREGATE WEEKLY SENTIMENT
# ======================================================

grouped = df.groupby(["year", "week", "subreddit", "sentiment"]).size().reset_index(name="count")

weekly = grouped.pivot_table(
    index=["year", "week", "subreddit"],
    columns="sentiment",
    values="count",
    fill_value=0
).reset_index()

weekly["total_comments"] = weekly["LABEL_0"] + weekly["LABEL_1"] + weekly["LABEL_2"]
weekly["negative_ratio"] = weekly["LABEL_0"] / weekly["total_comments"]
weekly["neutral_ratio"]  = weekly["LABEL_1"] / weekly["total_comments"]
weekly["positive_ratio"] = weekly["LABEL_2"] / weekly["total_comments"]

weekly.to_csv(OUTPUT_CSV, index=False)
print("Saved sentiment CSV:", OUTPUT_CSV)


# ======================================================
# 6. EXTRACT POLLING-LIKE SIGNALS FROM COMMENTS
# ======================================================

party_map = {
    r"\b(liberal|liberals|lpc|trudeau)\b": "LPC",
    r"\b(conservative|conservatives|cpc|poilievre|pp)\b": "CPC",
    r"\b(ndp|singh|jagmeet)\b": "NDP",
}

poll_regex = re.compile(
    r"(?P<value>\d{1,2})\s*(points?|pts?|percent|%)\s*(?P<direction>down|up|ahead|behind|lead|leads|leading)?",
    flags=re.IGNORECASE
)

def extract_poll_info(text):
    results = []
    t = text.lower()

    # detect party
    party = None
    for pattern, p in party_map.items():
        if re.search(pattern, t):
            party = p
            break
    if party is None:
        return results

    # extract polling numbers
    for m in poll_regex.finditer(text):
        val = int(m.group("value"))
        direction = m.group("direction")

        if direction:
            d = direction.lower()
            if d in ["down", "behind"]:
                val = -val
            elif d in ["up", "ahead", "lead", "leads", "leading"]:
                val = +val

        results.append((party, val))

    return results


poll_records = []
for _, row in df.iterrows():
    found = extract_poll_info(row["body"])
    for party, val in found:
        poll_records.append({
            "party": party,
            "value": val,
            "year": row["year"],
            "week": row["week"],
        })

poll_df = pd.DataFrame(poll_records)
print("Extracted poll-like signals:", len(poll_df))
print(poll_df.head())


# ======================================================
# 7. WEEKLY "POLL SIGNAL" AGGREGATION
# ======================================================

poll_weekly = poll_df.groupby(["year", "week", "party"])["value"].mean().reset_index()
poll_weekly.rename(columns={"value": "poll_signal"}, inplace=True)


# ======================================================
# 8. MAP SUBREDDITS → PARTIES AND MERGE
# ======================================================

subreddit_to_party = {
    "canadianconservative": "CPC",
    "cpc": "CPC",
    "lpc": "LPC",
    "ndp": "NDP",
    "canada": "GEN",
    "canadanews": "GEN",
    "canadapolitics": "GEN",
    "canadianpolitics": "GEN",
}

weekly["party"] = weekly["subreddit"].map(subreddit_to_party).fillna("GEN")

merged = pd.merge(
    weekly,
    poll_weekly,
    how="inner",
    on=["year", "week", "party"]
)

print("Merged sentiment + poll:", merged.head())


# ======================================================
# 9. CORRELATION ANALYSIS
# ======================================================

corr_pos = merged["positive_ratio"].corr(merged["poll_signal"])
corr_neg = merged["negative_ratio"].corr(merged["poll_signal"])
corr_neu = merged["neutral_ratio"].corr(merged["poll_signal"])

print("\n===== CORRELATION RESULTS =====")
print("Positive vs Poll:", corr_pos)
print("Negative vs Poll:", corr_neg)
print("Neutral vs Poll:", corr_neu)


# ======================================================
# 10. PLOTTING
# ======================================================

os.makedirs("plots", exist_ok=True)

for party in ["CPC", "LPC", "NDP"]:
    part_df = merged[merged["party"] == party].sort_values(["year", "week"])

    if len(part_df) == 0:
        continue

    x = part_df["year"] * 100 + part_df["week"]

    plt.figure(figsize=(10, 5))
    plt.plot(x, part_df["positive_ratio"], label="Positive Sentiment")
    plt.plot(x, part_df["poll_signal"] / 40, "--", label="Poll Proxy (scaled)")
    plt.title(f"{party}: Sentiment vs Extracted Poll Mentions")
    plt.xlabel("ISO Week")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fn = f"plots/sentiment_poll_{party}.png"
    plt.savefig(fn)
    plt.close()
    print("Saved:", fn)
