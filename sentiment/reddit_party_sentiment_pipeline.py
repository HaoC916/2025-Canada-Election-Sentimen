import os
import json
import re
import argparse
from datetime import datetime, timezone
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


#######################################################################
# 1) PARTY KEYWORD CLASSIFIER
#######################################################################

PARTY_KEYWORDS = {
    "lpc": [
        "trudeau", "justin trudeau", "liberal", "liberals",
        "lpc", "carney", "freeland", "chrystia freeland"
    ],
    "cpc": [
        "poilievre", "pierre poilievre", "conservative", "conservatives",
        "cpc", "tory", "tories"
    ],
    "ndp": [
        "ndp", "new democratic", "singh", "jagmeet", "jagmeet singh"
    ]
}

# Compile regex patterns
PARTY_REGEX = {
    p: re.compile("|".join(rf"\b{re.escape(k)}\b" for k in sorted(kws, key=lambda s: -len(s))),
                  flags=re.IGNORECASE)
    for p, kws in PARTY_KEYWORDS.items()
}

def classify_party(text: str) -> str:
    if not text:
        return "none"
    counts = {p: len(rx.findall(text)) for p, rx in PARTY_REGEX.items()}
    best_count = max(counts.values())
    if best_count == 0:
        return "none"
    tied = [p for p, c in counts.items() if c == best_count]
    if len(tied) > 1:
        return "ambiguous"
    return tied[0]


#######################################################################
# 2) SAFE SENTIMENT MODEL LOADING (safetensors-only + truncation)
#######################################################################

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

def load_sentiment_model():
    print("Loading safetensors-only model with truncation:", MODEL_NAME)

    device = 0 if torch.cuda.is_available() else -1
    print("Device set to:", "cuda" if device == 0 else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        model_max_length=512,
        padding_side="right",
        truncation=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        local_files_only=False,
        trust_remote_code=False,
        torch_dtype="auto"
    )

    model.to(device)
    model.eval()

    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    return classifier

def sentiment_numeric(label):
    if not label:
        return 0
    l = label.lower()
    if "neg" in l: return -1
    if "pos" in l: return 1
    return 0


#######################################################################
# 3) LOAD DATA FROM ALL MONTH FOLDERS
#######################################################################

def load_all_comments(root_folder):
    all_rows = []

    for month_folder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, month_folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"Reading folder: {month_folder}")

        for fname in sorted(os.listdir(folder_path)):
            if not fname.startswith("part-") or fname.endswith(".crc"):
                continue

            full_path = os.path.join(folder_path, fname)
            print(f"  Loading file: {fname}")

            # Parquet case
            if fname.endswith(".parquet"):
                df = pd.read_parquet(full_path)
                for _, row in df.iterrows():
                    all_rows.append(row.to_dict())
            else:
                with open(full_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            all_rows.append(json.loads(line))
                        except:
                            continue

    print("Total loaded comments:", len(all_rows))
    return all_rows


#######################################################################
# 4) PROCESS: PARTY + SENTIMENT + WEEK
#######################################################################

def process_comments(comments, classifier):
    results = []

    for c in tqdm(comments):
        body = c.get("body", "")

        party = classify_party(body)

        # Sentiment prediction
        out = classifier(body)[0]
        sent_label = out["label"]
        sent_num = sentiment_numeric(sent_label)

        # Convert timestamp
        ts = c.get("created_utc")
        if ts is None:
            year_week = "unknown"
        else:
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            year_week = dt.strftime("%Y-%W")

        results.append({
            "year_week": year_week,
            "party": party,
            "sent_label": sent_label,
            "sent_score_numeric": sent_num
        })

    return pd.DataFrame(results)


#######################################################################
# 5) WEEKLY AGGREGATION
#######################################################################

def weekly_aggregation(df):
    df["pos"] = (df["sent_score_numeric"] == 1).astype(int)
    df["neu"] = (df["sent_score_numeric"] == 0).astype(int)
    df["neg"] = (df["sent_score_numeric"] == -1).astype(int)

    agg = df.groupby(["year_week", "party"]).agg(
        total=("sent_score_numeric", "count"),
        pos=("pos", "sum"),
        neu=("neu", "sum"),
        neg=("neg", "sum")
    ).reset_index()

    agg["prop_pos"] = agg["pos"] / agg["total"]
    agg["prop_neu"] = agg["neu"] / agg["total"]
    agg["prop_neg"] = agg["neg"] / agg["total"]

    return agg


#######################################################################
# 6) MAIN
#######################################################################

def main(root_folder, output_csv):
    comments = load_all_comments(root_folder)
    classifier = load_sentiment_model()
    df = process_comments(comments, classifier)
    weekly = weekly_aggregation(df)
    weekly.to_csv(output_csv, index=False)
    print("Saved weekly sentiment CSV to:", output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.root, args.output)
