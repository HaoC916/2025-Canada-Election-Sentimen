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
│   └──  vader_tran_scored_sample.json
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
    ├── sentiment_daily/
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

# 🧪 Quick Test
If you want to verify functionality without re-running Steps 1–3,
all they need to run is:
```
python analysis/data_aggregation.py
python analysis/correlation_analysis.py
streamlit run analysis/dashboard.py
```
All of these use the sample data stored inside data/.

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

📦 (Optional) Skip Step 3 — Download Pre-Computed Sentiment Data

If you do NOT want to rerun the Transformer/VADER scoring (very slow without GPU),
download our pre-computed Step 3 output from Google Drive:

👉 https://drive.google.com/file/d/1bXJ_JlQu_xzUsRbajEGasY8UCPJ51-57/view?usp=drive_link 

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
This stage performs:

Weekly & Daily sentiment aggregation (Transformer + VADER)

Correlation analysis against Canadian polling data

Regression modeling

Visualization (heatmaps + regression plots)

Interactive dashboard using Streamlit

You can run the analysis on either:

Your own computed sentiment outputs (from Step 3), OR

Our sample dataset included in data/ (recommended for quick testing).

🔹 Step 4a — Weekly & Daily Aggregation

```
python analysis/data_aggregation.py
```

Input used by default:
```
data/vader_tran_scored_sample.json
```

Outputs:
```
results/sentiment_weekly_updated_key/
results/sentiment_daily_updated_key/
```
These files become inputs for Step 4b.

🔹 Step 4b Sentiment–Polling Correlation Analysis     
```
python analysis/correlation_analysis.py
```

This step:

Computes Pearson & Spearman correlations

Computes lag correlations

Fits OLS regressions

Saves all visualizations to:

```
results/plots_old/
```

and logs printed output to:
```
results/output_log_old.txt
```

🔹 Step 4c — Interactive Dashboard (Streamlit)

To view the web-based dashboard:
```
streamlit run analysis/dashboard.py
```

Once running, open:

👉 http://localhost:8501

The dashboard includes:

Weekly sentiment trends

Daily sentiment trends

Party comparisons

Correlation visualizations

Volume statistics

Transformer vs VADER comparison

---

# 👥 Authors
[][][]
CMPT 732 Group — Fall 2025  
SFU School of Computing Science
