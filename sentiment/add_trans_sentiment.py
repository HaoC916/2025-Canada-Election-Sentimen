import os
import json
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# -----------------------------------------
# PATH CONFIG
# -----------------------------------------
INPUT_DIR = "results/testing"
OUTPUT_DIR = "results/trans_scored_testing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
print("Loading model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_length=128,
    truncation=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    use_safetensors=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device:", device)

# label mapping: 0 negative, 1 neutral, 2 positive
BATCH_SIZE = 32 
def compute_sentiment_batch(sent_list):
    """Run model on a batch of sentences."""
    inputs = tokenizer(
        sent_list,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(**inputs).logits

    preds = torch.argmax(outputs, dim=1).cpu().tolist()
    return preds

def compute_sentiment(text):
    if not text or text.strip() == "":
        return 1

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[0].cpu()

    return int(torch.argmax(scores).item())


# -----------------------------------------
# PROCESS FILES
# -----------------------------------------
all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
print(f"Found {len(all_files)} JSON files.\n")

start_all = time.time()

for filename in tqdm(all_files, desc="Processing files"):
    in_path = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    print(f"\n➡️  Starting file: {filename}")
    start_file = time.time()

    # Read file
    with open(in_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    output_lines = []
    total_lines = len(lines)
    print(f"   → {total_lines} lines to process")

    #-----------------------------------------
    # # Process each line with an inner progress bar
    #-----------------------------------------
    # for line in tqdm(lines, desc=f"   Processing lines in {filename}", leave=False):
    #     try:
    #         record = json.loads(line)
    #     except:
    #         continue

    #     text = record.get("targeted_sentence", "")
    #     sentiment_label = compute_sentiment(text)

    #     record["trans_sentiment"] = sentiment_label
    #     output_lines.append(json.dumps(record, ensure_ascii=False))

    
    #________________________________
    # Process lines in batch
    #________________________________
    batch_text = []
    batch_records = []
    for line in tqdm(lines, desc=f"   Processing lines in {filename}", leave=False):
        try:
            record = json.loads(line)
        except:
            continue

        text = record.get("targeted_sentence", "")
        if not text.strip():
            record["trans_sentiment"] = 1  # default neutral
            output_lines.append(json.dumps(record, ensure_ascii=False))
            continue

        batch_text.append(text)
        batch_records.append(record)

        # If batch full → run inference
        if len(batch_text) >= BATCH_SIZE:
            preds = compute_sentiment_batch(batch_text)
            for r, p in zip(batch_records, preds):
                r["trans_sentiment"] = int(p)
                output_lines.append(json.dumps(r, ensure_ascii=False))
            batch_text = []
            batch_records = []

    # Flush remaining items in last batch
    if batch_text:
        preds = compute_sentiment_batch(batch_text)
        for r, p in zip(batch_records, preds):
            r["trans_sentiment"] = int(p)
            output_lines.append(json.dumps(r, ensure_ascii=False))

    #-----------------------------------------
    # Save output
    #-----------------------------------------
    with open(out_path, "w", encoding="utf-8") as outfile:
        for row in output_lines:
            outfile.write(row + "\n")

    elapsed = time.time() - start_file
    print(f"✔ Finished {filename} in {elapsed:.2f} seconds")

total_time = time.time() - start_all
print(f"\n🎉 ALL DONE in {total_time/60:.2f} minutes")
