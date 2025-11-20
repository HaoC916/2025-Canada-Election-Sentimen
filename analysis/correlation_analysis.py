import pandas as pd
from scipy.stats import pearsonr, spearmanr
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

# ============================================================
# 1. Load data
# ============================================================

sent_daily = pd.read_json("data/sentiment_daily/part-00000-d71794f2-a3cc-4580-8037-16a0a9af8fc0-c000.json", lines=True)
sent_weekly = pd.read_json("data/sentiment_weekly/part-00000-bfe96b1c-7e79-4691-943b-29703d5791c9-c000.json", lines=True)

polling = pd.read_csv("data/polling_averages.txt", sep="\s+")

# convert percentages "44.20%" → 44.20
for col in ["lpc", "cpc", "ndp", "oth"]:
    polling[col] = polling[col].str.replace("%", "").astype(float)

polling["date"] = pd.to_datetime(polling["date"])
sent_daily["date"] = pd.to_datetime(sent_daily["date"])
sent_weekly["week"] = sent_weekly["week"].astype(int)

# ============================================================
# 2. Split daily + weekly polling
# ============================================================

poll_daily = polling[polling["freq"] == "daily"].copy()
poll_daily["date"] = pd.to_datetime(poll_daily["date"])

# IMPORTANT: use manual week column, do NOT recompute week
poll_weekly = (
    polling
    .groupby("week")[["lpc", "cpc", "ndp", "oth"]]
    .mean()
    .reset_index()
)

# ============================================================
# 3. Merge WEEKLY sentiment with WEEKLY polling
# ============================================================

weekly_merged = sent_weekly.merge(
    poll_weekly,
    on="week",
    suffixes=("_sent", "_poll")
)

# ============================================================
# 4. Merge DAILY sentiment with DAILY polling
# ============================================================

daily_merged = sent_daily.merge(
    poll_daily,
    on="date",
    suffixes=("_sent", "_poll")
)

# ============================================================
# 5. Compute correlations
# ------------------------------------------------------------
# We compute correlation for each party:
#   - sentiment (vader_avg)
#   - transformer sentiment (trans_pos_ratio)
# with polling support %
# ============================================================

# ============================================================
# Correlation function
# ============================================================

def compute_corr(df, sent_col, poll_col):
    valid = df[[sent_col, poll_col]].dropna()
    if len(valid) < 3:
        return None
    return pearsonr(valid[sent_col], valid[poll_col])[0], spearmanr(valid[sent_col], valid[poll_col])[0]


parties = {"Liberal": "lpc", "Conservative": "cpc", "NDP": "ndp"}

sent_cols = ["vader_avg", "trans_pos_ratio"]


# ============================================================
# WEEKLY — NO LAG
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

# ============================================================
# WEEKLY — lag = 1
# ============================================================

poll_weekly_lagged = poll_weekly.copy()
poll_weekly_lagged["week"] += 1   # shift sentiment forward 1 week

weekly_lagged_merged = sent_weekly.merge(
    poll_weekly_lagged,
    on="week",
    how="inner"
)

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
# DAILY — lag 1~5
# ============================================================

daily_corr = []

for lag in range(0, 6):  # include lag 0
    temp_poll = poll_daily.copy()
    temp_poll["date"] = temp_poll["date"] + pd.Timedelta(days=lag)

    merged = sent_daily.merge(
        temp_poll,
        on="date",
        how="inner"
    )

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
# Save correlation heatmaps
# ============================================================

os.makedirs("results/plots", exist_ok=True)

def make_heatmap(df, value_col, title, filename):
    pivot = df.pivot(index="party", columns="lag", values=value_col)
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"results/plots/{filename}")
    plt.close()

# Weekly heatmap (lag 0 & 1)
make_heatmap(weekly_corr_df, "trans_pos_ratio_pear", "Weekly TRANS Pearson Correlations (Lag 0–1)", "weekly_heatmap.png")
make_heatmap(weekly_corr_df, "vader_avg_pear", "Weekly VADER Pearson Correlations (Lag 0–1)", "weekly_vader_heatmap.png")
# Daily heatmap (lag 0–5)
make_heatmap(daily_corr_df, "trans_pos_ratio_pear", "Daily TRANS Pearson Correlations (Lag 0–5)", "daily_heatmap.png")
make_heatmap(daily_corr_df, "vader_avg_pear", "Daily VADER Pearson Correlations (Lag 0–5)", "daily_vader_heatmap.png")
# ============================================================
# Print summary
# ============================================================

print("\n===== WEEKLY CORRELATIONS =====\n")
print(weekly_corr_df)

print("\n===== DAILY CORRELATIONS =====\n")
print(daily_corr_df)

print("\nHeatmaps saved to results/plots/")

# ============================================================
# 7. Regression scatter + regression line + T-test
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

    # 回归线
    x_line = np.linspace(X.min(), X.max(), 100)
    X_line = sm.add_constant(x_line)
    y_line = model.predict(X_line)

    plt.figure(figsize=(5, 4))
    plt.scatter(X, y, alpha=0.7, label="data")
    plt.plot(x_line, y_line, linewidth=2, label="OLS fit")
    plt.title(title)
    plt.xlabel(f"{prefix} sentiment")
    plt.ylabel("Polling support (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{filename}")
    plt.close()

    slope = model.params[1]
    tval = model.tvalues[1]
    pval = model.pvalues[1]
    r2 = model.rsquared

    print(f"[{title}] slope={slope:.4f}, t={tval:.2f}, p={pval:.3g}, R2={r2:.3f}")
    return slope, tval, pval, r2


print("\n===== WEEKLY REGRESSIONS =====")
for party, poll_col in parties.items():
    regression_and_plot(
        weekly_merged, party,
        sent_col="vader_avg", poll_col=poll_col,
        index_col="week",
        title=f"Weekly VADER vs Poll – {party}",
        filename=f"weekly_vader_reg_{party}.png",
        prefix="VADER"
    )
    regression_and_plot(
        weekly_merged, party,
        sent_col="trans_pos_ratio", poll_col=poll_col,
        index_col="week",
        title=f"Weekly Transformer vs Poll – {party}",
        filename=f"weekly_trans_reg_{party}.png",
        prefix="Transformer"
    )

print("\n===== DAILY REGRESSIONS =====")
for party, poll_col in parties.items():
    regression_and_plot(
        daily_merged, party,
        sent_col="vader_avg", poll_col=poll_col,
        index_col="date",
        title=f"Daily VADER vs Poll – {party}",
        filename=f"daily_vader_reg_{party}.png",
        prefix="VADER"
    )
    regression_and_plot(
        daily_merged, party,
        sent_col="trans_pos_ratio", poll_col=poll_col,
        index_col="date",
        title=f"Daily Transformer vs Poll – {party}",
        filename=f"daily_trans_reg_{party}.png",
        prefix="Transformer"
    )

print("\nAll plots saved under results/plots/")