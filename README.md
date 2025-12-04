# 🇨🇦 Election Sentiment 2025  
### *CMPT 732 — Big Data Final Project*

This project analyzes **Canadian political sentiment** across Reddit and compares it with **federal election polling trends**.  
We implement:

- **Baseline method** — Targeted VADER sentiment
- **Advanced method** — Transformer-based sentiment (RoBERTa)
- **Time-series analysis** — Weekly sentiment per party  
- **Correlation analysis** — Sentiment vs polling numbers

The final output:  
**A complete pipeline that downloads → cleans → processes → aggregates → merges → visualizes**.

---

# 📁 Project Structure

```
election-sentiment-2025/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── sentiment_daily/
│   ├── sentiment_daily_updated_key/
│   ├── sentiment_weekly/
│   ├── sentiment_weekly_updated_key/
│   ├── polling_averages.txt
│   ├── probability_winning.txt
│   ├── vader_tran_scored_sample.json
│   └── sample_scored.py
│
├── ETL/
│   ├── filter_comments_zst.py       # Step 1a: RC_YYYY-MM.zst → filtered JSONL
│   ├── filter_submissions_zst.py    # Step 1b: RS_YYYY-MM.zst → filtered JSONL
│   ├── comments_filter.py           # Step 2a: Spark comment cleanup
│   ├── submissions_filter.py        # Step 2b: Spark submission cleanup
│   ├── join_titles.py               # Step 2c: Join comments + submission titles
│   └── logs.txt
│
├── sentiment/
│   ├── extract_party_sentences.py   # Step 3a: Extract targeted sentences per party
│   ├── add_trans_sentiment.py       # Step 3b: Transformer (RoBERTa) scoring
│   └── baseline_vader.py            # Step 3c: VADER scoring
│
├── analysis/
│   ├── data_aggregation.py          # Step 4a: Weekly/daily aggregation
│   ├── correlation_analysis.py      # Step 4b: Sentiment–polling correlation
│   ├── dashboard.py                 # Front-end: Streamlit dashboard
│   └── sample_scored.py
│
└── results/
    ├── vader_targeted/
    ├── transformer/
    ├── weekly_sentiment/
    ├── sentiment_weekly/
    ├── merged/
    └── figures/
```

---

# ⚙️ Installation

```
pip install -r requirements.txt
```

---

# 🚀 Running the Full Pipeline

## 0. (Optional) Skip ETL — Download Pre-Joined Reddit Data

If you do NOT want to process 1.2 billion Reddit comments yourself,
you can directly download our joined & cleaned ETL output (2025-01 to 2025-04):

📦 Google Drive (recommended):
👉 https://drive.google.com/drive/folders/1BAYBEI2GYo1UPVWAQ_4IhW0m4GPKUKmi

This contains four folders:

2025-01.zip
2025-02.zip
2025-03.zip
2025-04.zip

Each ZIP contains:

joined_rdd/YYYY-MM/
    part-*.json

If you download these files, you may skip all ETL steps (step 1 & 2) and jump directly to:

➡ sentiment/extract_party_sentences.py

---

## 1. ETL - Fast Pre-Filtering
If you prefer to reproduce the ETL pipeline from scratch,
you must first download raw Reddit dumps (January–April 2025):

📥 Original Reddit RC/RS Data (Academic Torrents):
https://academictorrents.com/details/30dee5f0406da7a353aff6a8caa2d54fd01f2ca1

Required files (8 total):

RC_2025-01.zst
RC_2025-02.zst
RC_2025-03.zst
RC_2025-04.zst
RS_2025-01.zst
RS_2025-02.zst
RS_2025-03.zst
RS_2025-04.zst


The ETL step requires running five scripts manually, each with input and output paths.

All commands follow this structure:
python script.py <input_path> <output_path>
spark-submit script.py <input_path> <output_path>


🔹 Step 1a — Filter raw Reddit comments (.zst → filtered JSONL)
```
python ETL/filter_comments_zst.py <raw_comments_zst> <filtered_output_json>
```

Example:
```
python ETL/filter_comments_zst.py \
    /Users/ryan/datasets/reddit/comments/RC_2025-01.zst \
    /Users/ryan/datasets/reddit/comments/RC_2025-01_filtered.json
```

🔹 Step 1b — Filter raw Reddit submissions (.zst → filtered JSONL)
```
python ETL/filter_submissions_zst.py <raw_submissions_zst> <filtered_output_json>
```

Example:
```
python ETL/filter_submissions_zst.py \
    /Users/ryan/datasets/reddit/submissions/RS_2025-01.zst \
    /Users/ryan/datasets/reddit/submissions/RS_2025-01_filtered.json
```

---

## 2. ETL - Spark ETL
🔹 Step 2a — Spark cleaning of comments
```
spark-submit ETL/comments_filter.py <filtered_comments_json> <cleaned_output_dir>
```

Example:
```
spark-submit ETL/comments_filter.py \
    /Users/ryan/datasets/reddit/comments/RC_2025-01_filtered.json \
    /Users/ryan/datasets/reddit/cleaned/comments/2025-01
```

🔹 Step 2b — Spark cleaning of submissions
```
spark-submit ETL/submissions_filter.py <filtered_submissions_json> <cleaned_output_dir>
```

Example:
```
spark-submit ETL/submissions_filter.py \
    /Users/ryan/datasets/reddit/submissions/RS_2025-01_filtered.json \
    /Users/ryan/datasets/reddit/cleaned/submissions/2025-01
```

🔹 Step 2c — Join comments with submission titles
```
spark-submit ETL/join_titles.py <cleaned_comments_dir> <cleaned_submissions_dir> <joined_output_dir>
```

Example:
```
spark-submit ETL/join_titles.py \
    /Users/ryan/datasets/reddit/cleaned/comments/2025-01 \
    /Users/ryan/datasets/reddit/cleaned/submissions/2025-01 \
    /Users/ryan/datasets/reddit/joined/2025-01
```

---

## 3. Sentiment 
🔹 Step 3a — Extract Party Sentences

```
spark-submit sentiment/extract_party_sentences.py <joined_input_dir> <party_target_dir>
```

Example:
```
spark-submit sentiment/extract_party_sentences.py \
    "/Users/ryan/datasets/reddit/joined/2025-*" \
    "/Users/ryan/datasets/reddit/party_target/2025"
```

🔹 Step 3b — Extract Party Sentences
```
python sentiment/add_trans_sentiment.py <party_target_dir> <trans_sentiment_output_dir>
```

Example:
```
python sentiment/add_trans_sentiment.py \
    /Users/ryan/datasets/reddit/party_target/ \
    /Users/ryan/datasets/reddit/trans_sentiment/
```

🔹 Step 3c — Baseline Vader
```
python sentiment/baseline_vader.py <trans_scored_dir> <vader_output_dir>
```

Example:
```
python sentiment/baseline_vader.py \
    /Users/ryan/datasets/reddit/trans_sentiment/2025 \
    /Users/ryan/datasets/reddit/vader_sentiment/2025
```
---

## 4. Analysis


🔹 Step 4a Data Aggregation

🔹 Step 4b Correlation Analysis      
```
python sentiment/transformer_sentiment.py
```

Output saved to:

```
results/transformer/
```

---

# 🧠 Methods Summary

### ✔ Data Cleaning  
We filtered 3×10⁸ monthly Reddit comments down to ~5×10⁵ relevant political comments using:

- Subreddit filtering  
- Joining comments ↔ submissions (to attach titles)  
- Lowercasing, null removal  
- Adding full_text (title + body)

### ✔ Baseline: Targeted VADER  
We detect party-specific sentiment by:

1. Extracting sentences containing party keywords  
2. Running VADER only on those windows  
3. Producing multi-label sentiment if multiple parties appear  
4. Exploding into one row per (comment, party) for analysis

### ✔ Transformer Model  
RoBERTa-base fine-tuned on political sentiment datasets provides high-quality polarity scores.

### ✔ Time-Series + Correlation  
We group by:

- (week, party)
- mean sentiment  
- compare against federal poll averages

---

# 📊 Results Overview

- Weekly sentiment clearly tracks several major political events  
- Conservatives show higher variance around leadership discussions  
- Liberals show polarized sentiment around Trudeau-related topics  
- Transformer model produces smoother trends than VADER  
- Correlation with polls varies by party (discussed in final report)

---

# 📝 Project Summary

This repository contains the full reproducible pipeline required for:

- Cleaning and normalizing Reddit commentary  
- Extracting political sentiment signals  
- Aggregating by time  
- Merging with real polling data  
- Visualizing trends for election prediction analysis  

---

# 👥 Authors
[][][]
CMPT 732 Group — Fall 2025  
SFU School of Computing Science
