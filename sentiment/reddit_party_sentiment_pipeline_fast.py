import os
import json
import re
import argparse
from datetime import datetime, timezone
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



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

PARTY_REGEX = {
    p: re.compile("|".join(rf"\b{re.escape(k)}\b" for k in sorted(kws, key=lambda s: -len(s))),
                  flags=re.IGNORECASE)
    for p, kws in PARTY_KEYWORDS.items()
}

def classify_party(text: str) -> str:
    if not text:
        return "none"
    counts = {p: len(rx.findall(text)) for p, rx in PARTY_REGEX.items()}
    max_count = max(counts.values())
    if max_count == 0:
        return "none"
    ties = [p for p, c in counts.items() if c == max_count]
    if len(ties) > 1:
        return "ambiguous"
    return ties[0]



#######################################################################
# 2) SAFE SENTIMENT MODEL (USE SAFETENSORS, NO torch.load)
#######################################################################

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

def load_sentiment_model():
    print("Loading safetensors-only transformer:", MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        trust_remote_code=False,
        torch_dtype="auto"
    ).to(device)

    model.eval()

    return tokenizer, model, device


def sentiment_numeric(label):
    label = label.lower()
    if "neg" in label:
        return -1
    if "pos" in label:
        return 1
    return 0



#######################################################################
# 3) FAST BATCH SENTIMENT INFERENCE
#######################################################################

def predict_batch(texts, tokenizer, model, device, batch_size=32):
    """Fast GPU batching, safe truncation."""
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        probs = torch.softmax(logits, dim=1)
        scores, labels = probs.max(dim=1)

        for l, s in zip(labels.cpu().tolist(), scores.cpu().tolist()):
            if l == 0:
                lbl = "NEGATIVE"
            elif l == 1:
                lbl = "NEUTRAL"
            else:
                lbl = "POSITIVE"

            results.append((lbl, float(s)))

    return results



#######################################################################
# 4) LOAD ALL COMMENTS
#######################################################################

def load_all_comments(root_folder):
    rows = []

    for month in sorted(os.listdir(root_folder)):
        month_path = os.path.join(root_folder, month)
        if not os.path.isdir(month_path):
            continue

        print(f"Reading folder: {month}")

        for fname in sorted(os.listdir(month_path)):
            if not fname.startswith("part-") or fname.endswith(".crc"):
                continue

            full_path = os.path.join(month_path, fname)
            print("  Loading", fname)

            if fname.endswith(".parquet"):
                df = pd.read_parquet(full_path)
                for _, row in df.iterrows():
                    rows.append(row.to_dict())
            else:
                with open(full_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rows.append(json.loads(line))
                        except:
                            continue

    print("Total loaded comments:", len(rows))
    return rows



#######################################################################
# 5) PROCESS ALL COMMENTS (party + week + sentiment)
#######################################################################

def process_comments(comments, tokenizer, model, device):
    bodies = [c.get("body", "") for c in comments]

    print("Running batched sentiment inference…")
    sent_results = predict_batch(bodies, tokenizer, model, device, batch_size=32)

    out_rows = []

    print("Building final table…")
    for c, (lbl, score) in zip(comments, sent_results):
        party = classify_party(c.get("body", ""))

        ts = c.get("created_utc")
        if ts is None:
            week = "unknown"
        else:
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            week = dt.strftime("%Y-%W")

        out_rows.append({
            "year_week": week,
            "party": party,
            "sent_label": lbl,
            "sent_score_numeric": sentiment_numeric(lbl)
        })

    return pd.DataFrame(out_rows)



#######################################################################
# 6) WEEKLY AGGREGATION
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
# 7) MAIN
#######################################################################

def main(root_folder, output_file):
    comments = load_all_comments(root_folder)
    tokenizer, model, device = load_sentiment_model()
    df = process_comments(comments, tokenizer, model, device)
    weekly = weekly_aggregation(df)
    weekly.to_csv(output_file, index=False)
    print("Saved:", output_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.root, args.output)
