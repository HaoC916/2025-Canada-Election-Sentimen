"""
* Program: correlation_analysis.py
*
* Modified Date: November 2025
*
* Purpose: Step 4b of Sentiment Analysis Pipeline
*    Compare weekly and daily Reddit sentiment with national polling data,
*    compute correlations, generate regression models, and produce plots.
*
*    This script performs:
*        - Pearson & Spearman correlations (weekly + daily)
*        - Lag analysis:
*              weekly  : lag 0-1
*              daily   : lag 0-5 days
*        - Regression models (OLS) for each party:
*              PollSupport_t = β0 + β1 * Sentiment_t + ε
*        - Visualization:
*              * Heatmaps for TRANS/VADER correlations
*              * Regression scatterplots with fitted lines
*        - Logging all printed output into a log file
*
* This script completes the analysis pipeline by quantifying how closely
* Reddit sentiment aligns with real-world polling trends.
*
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import sys

# ============================================================
# CUSTOM OUTPUT DIRECTORY (only change here)
# ============================================================

OUTPUT_DIR = "results"          
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots_old")
LOG_FILE = os.path.join(OUTPUT_DIR, "output_log_old.txt")

os.makedirs(PLOT_DIR, exist_ok=True)

# Redirect print() to both console + log file
class Tee:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Tee(LOG_FILE)

# ============================================================
# 1. Load data
# Important Notes:
#    - Paths for sentiment and polling files are HARD-CODED and MUST be
#      updated before running if directory structure differs.
#    - Only the first part-00000 file is loaded by default — modify as needed.
# ============================================================

# sent_daily = pd.read_json("data/sentiment_daily_updated_key/part-00000-26623bf6-32d4-42a0-be59-19b77768579d-c000.json", lines=True)
# sent_weekly = pd.read_json("data/sentiment_weekly_updated_key/part-00000-781e3a70-de8c-4e41-b49b-ec42dfd96bbe-c000.json", lines=True)

sent_daily = pd.read_json("data/sentiment_daily/part-00000-d71794f2-a3cc-4580-8037-16a0a9af8fc0-c000.json", lines=True)
sent_weekly = pd.read_json("data/sentiment_weekly/part-00000-bfe96b1c-7e79-4691-943b-29703d5791c9-c000.json", lines=True)


polling = pd.read_csv("data/polling_averages.txt", sep="\s+")

# convert percentage fields
for col in ["lpc", "cpc", "ndp", "oth"]:
    polling[col] = polling[col].str.replace("%", "").astype(float)

polling["date"] = pd.to_datetime(polling["date"])
sent_daily["date"] = pd.to_datetime(sent_daily["date"])
sent_weekly["week"] = sent_weekly["week"].astype(int)

# ============================================================
# 2. Poll split
# ============================================================

poll_daily = polling[polling["freq"] == "daily"].copy()

poll_weekly = (
    polling.groupby("week")[["lpc", "cpc", "ndp", "oth"]]
    .mean()
    .reset_index()
)

# ============================================================
# 3. WEEKLY MERGE
# ============================================================

weekly_merged = sent_weekly.merge(
    poll_weekly,
    on="week",
    suffixes=("_sent", "_poll")
)

# ============================================================
# 4. DAILY MERGE
# ============================================================

daily_merged = sent_daily.merge(
    poll_daily,
    on="date",
    suffixes=("_sent", "_poll")
)

# ============================================================
# 5. Correlation function
# ============================================================

def compute_corr(df, sent_col, poll_col):
    valid = df[[sent_col, poll_col]].dropna()
    if len(valid) < 3:
        return None, None
    return pearsonr(valid[sent_col], valid[poll_col])[0], spearmanr(valid[sent_col], valid[poll_col])[0]


parties = {"Liberal": "lpc", "Conservative": "cpc", "NDP": "ndp"}
sent_cols = ["vader_avg", "trans_pos_ratio"]

# ============================================================
# WEEKLY — lag 0 & 1
# ============================================================

weekly_corr = []
for party, poll_col in parties.items():
    dfp = weekly_merged[weekly_merged["party"] == party]
    row = {"party": party, "lag": 0}
    for sent_col in sent_cols:
        pear, spear = compute_corr(dfp, sent_col, poll_col)
        row[f"{sent_col}_pear"] = pear
        row[f"{sent_col}_spear"] = spear
    weekly_corr.append(row)

# lag = 1
poll_weekly_lagged = poll_weekly.copy()
poll_weekly_lagged["week"] += 1

weekly_lagged_merged = sent_weekly.merge(poll_weekly_lagged, on="week", how="inner")

for party, poll_col in parties.items():
    dfp = weekly_lagged_merged[weekly_lagged_merged["party"] == party]
    row = {"party": party, "lag": 1}
    for sent_col in sent_cols:
        pear, spear = compute_corr(dfp, sent_col, poll_col)
        row[f"{sent_col}_pear"] = pear
        row[f"{sent_col}_spear"] = spear
    weekly_corr.append(row)

weekly_corr_df = pd.DataFrame(weekly_corr)

# ============================================================
# DAILY — lag 0–5
# ============================================================

daily_corr = []
for lag in range(0, 6):
    temp_poll = poll_daily.copy()
    temp_poll["date"] = temp_poll["date"] + pd.Timedelta(days=lag)

    merged = sent_daily.merge(temp_poll, on="date", how="inner")

    for party, poll_col in parties.items():
        dfp = merged[merged["party"] == party]
        row = {"party": party, "lag": lag}
        for sent_col in sent_cols:
            pear, spear = compute_corr(dfp, sent_col, poll_col)
            row[f"{sent_col}_pear"] = pear
            row[f"{sent_col}_spear"] = spear
        daily_corr.append(row)

daily_corr_df = pd.DataFrame(daily_corr)

# ============================================================
# HEATMAP FUNCTION
# ============================================================

def make_heatmap(df, value_col, title, filename):
    pivot = df.pivot(index="party", columns="lag", values=value_col)
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

# weekly heatmaps
make_heatmap(weekly_corr_df, "trans_pos_ratio_pear", "Weekly TRANS Pearson Correlations (Lag 0–1)", "weekly_trans.png")
make_heatmap(weekly_corr_df, "vader_avg_pear", "Weekly VADER Pearson Correlations (Lag 0–1)", "weekly_vader.png")

# daily heatmaps
make_heatmap(daily_corr_df, "trans_pos_ratio_pear", "Daily TRANS Pearson Correlations (Lag 0–5)", "daily_trans.png")
make_heatmap(daily_corr_df, "vader_avg_pear", "Daily VADER Pearson Correlations (Lag 0–5)", "daily_vader.png")

print("\n===== WEEKLY CORRELATIONS =====")
print(weekly_corr_df)

print("\n===== DAILY CORRELATIONS =====")
print(daily_corr_df)

print(f"\nHeatmaps saved to: {PLOT_DIR}")

# ============================================================
# REGRESSION + PLOT
# ============================================================

def regression_and_plot(df, party, sent_col, poll_col,
                        index_col, title, filename, prefix):

    dfp = df[df["party"] == party].copy().sort_values(index_col)
    dfp = dfp.dropna(subset=[sent_col, poll_col])

    if len(dfp) < 5:
        return None

    X = dfp[sent_col].to_numpy()
    y = dfp[poll_col].to_numpy()

    X_design = sm.add_constant(X)
    model = sm.OLS(y, X_design).fit()

    x_line = np.linspace(X.min(), X.max(), 100)
    X_line = sm.add_constant(x_line)
    y_line = model.predict(X_line)

    plt.figure(figsize=(5, 4))
    plt.scatter(X, y, alpha=0.7)
    plt.plot(x_line, y_line, linewidth=2)
    plt.title(title)
    plt.xlabel(f"{prefix} sentiment")
    plt.ylabel("Polling support (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

    slope = model.params[1]
    tval = model.tvalues[1]
    pval = model.pvalues[1]
    r2 = model.rsquared

    print(f"[{title}] slope={slope:.4f}, t={tval:.2f}, p={pval:.3g}, R2={r2:.3f}")

    return slope, tval, pval, r2

print("\n===== WEEKLY REGRESSIONS =====")
for party, poll_col in parties.items():
    regression_and_plot(weekly_merged, party, "vader_avg", poll_col,
                        "week", f"Weekly VADER vs Poll – {party}",
                        f"weekly_vader_reg_{party}.png", "VADER")

    regression_and_plot(weekly_merged, party, "trans_pos_ratio", poll_col,
                        "week", f"Weekly TRANS vs Poll – {party}",
                        f"weekly_trans_reg_{party}.png", "Transformer")

print("\n===== DAILY REGRESSIONS =====")
for party, poll_col in parties.items():
    regression_and_plot(daily_merged, party, "vader_avg", poll_col,
                        "date", f"Daily VADER vs Poll – {party}",
                        f"daily_vader_reg_{party}.png", "VADER")

    regression_and_plot(daily_merged, party, "trans_pos_ratio", poll_col,
                        "date", f"Daily TRANS vs Poll – {party}",
                        f"daily_trans_reg_{party}.png", "Transformer")

print(f"\nAll plots saved under {PLOT_DIR}")
print(f"\nAll print output saved at: {LOG_FILE}")