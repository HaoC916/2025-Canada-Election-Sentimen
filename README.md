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
├── data/
│   ├── reddit_raw/                    # Original comments + submissions
│   ├── polls/                         # Polling CSV files
│   └── cleaned/                       # Joined and filtered Reddit data
│
├── sentiment/
│   ├── vader_sentiment.py             # Targeted VADER baseline sentiment
│   ├── transformer_sentiment.py       # Transformer-based sentiment
│   └── keywords.py                    # Party keyword dictionaries
│
├── analysis/
│   ├── weekly_aggregation.py          # Weekly sentiment per party
│   ├── merge_polls.py                 # Merge sentiment with polls
│   ├── plot_results.py                # Final visualizations
│   └── utils.py                       # Helper functions
│
├── results/
│   ├── vader_targeted/                # Baseline sentiment results
│   ├── transformer/                   # Transformer sentiment results
│   ├── weekly_sentiment/              # Week-level aggregated outputs
│   ├── merged/                        # Sentiment + polls
│   └── figures/                       # Final plots
│
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

```
pip install -r requirements.txt
```

---

# 🚀 Running the Full Pipeline

## 1. Data Cleaning

```
python data_cleaning/join_comments_submissions.py
```

Produces:

```
data/cleaned/joined_rdd/YYYY-MM/
```

---

## 2. Baseline Sentiment (VADER)

```
python sentiment/vader_sentiment.py
```

Output saved to:

```
results/vader_targeted/
```

---

## 3. Transformer Sentiment

```
python sentiment/transformer_sentiment.py
```

Output saved to:

```
results/transformer/
```

---

## 4. Weekly Aggregation

```
python analysis/weekly_aggregation.py
```

Output:

```
results/weekly_sentiment/
```

---

## 5. Merge Sentiment With Polling Data

```
python analysis/merge_polls.py
```

Output:

```
results/merged/
```

---

## 6. Generate Final Plots

```
python analysis/plot_results.py
```

Figures saved to:

```
results/figures/
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

CMPT 732 Group — Fall 2025  
SFU School of Computing Science