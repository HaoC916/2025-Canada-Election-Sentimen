"""
* Program: sample_scored.py
*
* Modified Date: December 2025
*
* Purpose: Randomly down-sample large scored sentiment JSON files
*          (vader_tran_scored_updated_key/part-*.json)
*          into a smaller JSONL file (vader_tran_scored_sample.json)
*          for the dashboard and quick debugging.
"""

import json
import random
import glob
import os

# ===============================
# CONFIG
# ===============================
SAMPLE_RATIO = 0.001   # 0.1% sample 

INPUT_PATTERN = "results/vader_tran_scored_updated_key/part-*.json"
OUTPUT_FILE = "results/vader_tran_scored_sample.json"

# ===============================
# Sampling script
# ===============================
files = sorted(glob.glob(INPUT_PATTERN))

total_lines = 0
kept_lines = 0

out = open(OUTPUT_FILE, "w", encoding="utf-8")

print(f"Sampling {SAMPLE_RATIO*100:.2f}% from {len(files)} files...")

for f in files:
    with open(f, "r", encoding="utf-8") as infile:
        for line in infile:
            total_lines += 1
            if random.random() < SAMPLE_RATIO:
                out.write(line)
                kept_lines += 1

out.close()

print("===================================")
print(" DONE! Sampled dataset created.")
print(" Saved to:", OUTPUT_FILE)
print(f" Total lines scanned: {total_lines:,}")
print(f" Lines kept:         {kept_lines:,}")
print(" Approx size:", os.path.getsize(OUTPUT_FILE)/1024/1024, "MB")
print("===================================")