# 🇨🇦 Election Sentiment 2025  
### *CMPT 732 — Big Data Final Project*

This project was originally developed as part of a graduate-level Big Data course and has been refactored and published for portfolio purposes.

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
│   └── vader_tran_scored_sample.json
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

```

---

# ⚙️ Installation

```
pip install -r requirements.txt
```

---

# 🧪 Quick Test
If you want to verify functionality without re-running Steps 1–3,
all they need to run is:
```
python analysis/correlation_analysis.py
streamlit run analysis/dashboard.py
```

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

## 3. Sentiment （Requires 5~30 hours to finish）

📦 Skip Step 3? —> Download Pre-Computed Sentiment Data

If you do NOT want to rerun the Transformer/VADER scoring (very slow without GPU),
download our pre-computed Step 3 output from Google Drive:

👉 https://drive.google.com/file/d/1bXJ_JlQu_xzUsRbajEGasY8UCPJ51-57/view?usp=drive_link 

After downloading the data file, extract to the data/ folder, and continue with step 4.

🔹 Step 3a — Extract Party Sentences

```
spark-submit sentiment/extract_party_sentences.py <joined_input_dir> <party_target_dir>
```

Example:
```
spark-submit sentiment/extract_party_sentences.py data/joined/2025-* data/party_target_updated_keywords
```

🔹 Step 3b — Extract Party Sentences
```
spark-submit sentiment/extract_party_sentences.py \
    'data/joined/2025-*' \
    data/party_target_updated_keywords
```

Example:
```
python sentiment/add_trans_sentiment.py \
    'data/party_target_updated_keywords' \
    data/trans_sentiment
```

🔹 Step 3c — Baseline Vader
```
python sentiment/baseline_vader.py <trans_scored_dir> <vader_output_dir>
```

Example:
```
python sentiment/baseline_vader.py \
    'data/trans_sentiment' \
    data/vader_tran_scored_updated_keywords
```
---

## 4. Analysis

### Important Notes
Before running Step 4, you **must have the scored sentiment dataset**.

You can obtain it in one of two ways:

1. **Download our pre-scored dataset**  
   👉 https://drive.google.com/file/d/1bXJ_JlQu_xzUsRbajEGasY8UCPJ51-57/view?usp=drive_link  
   Extract the downloaded file into the `data/` folder.

2. **Or generate it** by running Step 3 sentiment scoring scripts  
   (see Step 3 section in this README).

---

## What Step 4 Does

This stage performs:

- Weekly & daily sentiment aggregation (Transformer + VADER)  
- Correlation analysis against Canadian polling data  
- Regression modeling  
- Visualization (heatmaps + regression plots)  
- Interactive dashboard (Streamlit)

You may run Step 4 using either:

- Your own Step 3 computation outputs, **or**  
- Our included sample dataset (recommended for quick testing)

---

## 🔹 Step 4a — Weekly & Daily Aggregation

Run:
```
python analysis/data_aggregation.py
```

**Default Input:**
```
data/vader_tran_scored_updated_key
```

**Outputs:**
```
data/sentiment_weekly_updated_key/
data/sentiment_daily_updated_key/
```

These files will be used in Step 4b.

---

## 🔹 Step 4b — Sentiment–Polling Correlation Analysis

Run:
```
python analysis/correlation_analysis.py
```

This step:

- Computes **Pearson & Spearman** correlations  
- Computes **lag correlations**  
- Fits **OLS regressions**  
- Generates all visualizations

Results saved to:
```
results/plots/
```

Printed log saved to:
```
results/output_log.txt
```

---

## 🔹 Step 4c — Interactive Dashboard (Streamlit)

Run the dashboard:

```
streamlit run analysis/dashboard.py
```

Then open:

👉 http://localhost:8501

Dashboard features:

- Weekly sentiment trends  
- Daily sentiment trends  
- Party comparisons  
- Correlation visualizations  
- Volume statistics  
- Transformer vs VADER comparison  

---

## 👥 Authors
**Luna Sang**, **Ryan Chen**, **Zili Ding**  
CMPT 732 Group — Fall 2025  
SFU School of Computing Science
