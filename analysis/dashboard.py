"""
* Program: dashboard.py
*
* Modified Date: December 2025
*
* Purpose: Interactive Streamlit dashboard to explore Reddit sentiment and polling:
*          loads pre-aggregated daily/weekly sentiment, merges with polls,
*          and visualizes time-series, correlations, regressions, and raw samples.
*          
*          Data is loaded from (already stored in the data/ folder):
*              - data/sentiment_daily_updated_key/part-*.json
*              - data/sentiment_weekly_updated_key/part-*.json
*              - data/polling_averages.txt
*              - data/vader_tran_scored_sample.json (sampled scored comments)
*          Modify paths if needed.
*         
* Note: Used ChatGPT to help learning streamlit in order to implement UI dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go

# ============================================
# Base directory: project root (parent of analysis/)
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================
# Helper: Load JSON folder with part-*.json files
# ============================================
def load_json_folder(rel_folder, pattern="part-*.json", time_col=None, int_cols=None, name=""):
    """
    Load all JSON files matching pattern from a folder.
    Supports large line-delimited JSON files.
    """
    folder = BASE_DIR / rel_folder
    files = sorted(folder.glob(pattern))

    if not files:
        st.error(f"[{name}] No files found in: {folder}/{pattern}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_json(f, lines=True)
            dfs.append(df)
        except Exception as e:
            st.error(f"[{name}] Failed to read {f}: {e}")

    df = pd.concat(dfs, ignore_index=True)

    # Parse datetime columns
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Convert integer-like columns
    if int_cols:
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df

# ============================================
# Loaders for each dataset
# ============================================
@st.cache_data
def load_daily():
    """Load pre-aggregated daily sentiment."""
    df = load_json_folder(
        rel_folder="data/sentiment_daily_updated_key",
        pattern="part-*.json",
        time_col="date",
        name="DAILY_SENTIMENT",
    )
    if "party" in df.columns:
        df["party"] = df["party"].astype(str)
    return df


@st.cache_data
def load_weekly():
    """Load pre-aggregated weekly sentiment."""
    df = load_json_folder(
        rel_folder="data/sentiment_weekly_updated_key",
        pattern="part-*.json",
        int_cols=["week"],
        name="WEEKLY_SENTIMENT",
    )
    if "party" in df.columns:
        df["party"] = df["party"].astype(str)
    return df


@st.cache_data
def load_polling():
    """
    Load polling_averages.txt and clean percentage values.

    Returns
    -------
    poll_daily : DataFrame
        Rows where freq == 'daily' (date-based).
    poll_weekly : DataFrame
        Polling averaged by week (week-based), matching your offline script.
    """
    path = BASE_DIR / "data" / "polling_averages.txt"
    df = pd.read_csv(path, sep=r"\s+")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Clean percentage fields (handles %, invisible chars, etc.)
    def clean_percent(x):
        if pd.isna(x):
            return None
        x = str(x).strip().replace("%", "").replace("\u200b", "")
        try:
            return float(x)
        except Exception:
            return None

    for col in ["lpc", "cpc", "ndp", "oth"]:
        df[col] = df[col].apply(clean_percent)

    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    # Daily polling (freq == daily)
    poll_daily = df[df["freq"] == "daily"].copy()

    # Weekly polling grouped by week (same as correlation script)
    poll_weekly = (
        df.groupby("week")[["lpc", "cpc", "ndp", "oth"]]
        .mean()
        .reset_index()
    )

    return poll_daily, poll_weekly


@st.cache_data
# def load_scored():
#     """Load sentence-level sentiment scored with VADER + Transformer."""
#     df = load_json_folder(
#         rel_folder="results/vader_tran_scored_updated_key",
#         pattern="part-*.json",
#         time_col="date",
#         int_cols=["week"],
#         name="SCORED_COMMENTS",
#     )
#     if "party" in df.columns:
#         df["party"] = df["party"].astype(str)
#     return df

@st.cache_data
def load_scored():
    df = pd.read_json(
        BASE_DIR / "data/vader_tran_scored_sample.json",
        lines=True
    )
    if "party" in df.columns:
        df["party"] = df["party"].astype(str)
    return df

# ============================================
# Load all data before running UI
# ============================================
sent_daily = load_daily()
sent_weekly = load_weekly()
poll_daily, poll_weekly = load_polling()
scored_df = load_scored()

# Party → polling column mapping
PARTIES = {
    "Liberal": "lpc",
    "Conservative": "cpc",
    "NDP": "ndp",
}

# ============================================
# Correlation calculation (lag 0, matches offline logic)
# Used ChatGPT to help write codes
# ============================================
def compute_corr_table_daily(sent_daily: pd.DataFrame, poll_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag=0 correlations for daily data.
    Merge on date, consistent with the offline script.
    """
    results = []

    for party, poll_col in PARTIES.items():
        df_s = sent_daily[sent_daily["party"] == party]
        merged = df_s.merge(poll_daily, on="date", how="inner")

        for model in ["vader_avg", "trans_pos_ratio"]:
            if model not in merged.columns or poll_col not in merged.columns:
                pear = spear = np.nan
            else:
                valid = merged[[model, poll_col]].dropna()
                if len(valid) < 5:
                    pear = spear = np.nan
                else:
                    pear = pearsonr(valid[model], valid[poll_col])[0]
                    spear = spearmanr(valid[model], valid[poll_col])[0]

            results.append(
                {
                    "party": party,
                    "model": model,
                    "pearson": pear,
                    "spearman": spear,
                }
            )

    return pd.DataFrame(results)


def compute_corr_table_weekly(sent_weekly: pd.DataFrame, poll_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag=0 correlations for weekly data.
    Merge on week, consistent with the offline script.
    """
    results = []

    for party, poll_col in PARTIES.items():
        df_s = sent_weekly[sent_weekly["party"] == party]
        merged = df_s.merge(poll_weekly, on="week", how="inner")

        for model in ["vader_avg", "trans_pos_ratio"]:
            if model not in merged.columns or poll_col not in merged.columns:
                pear = spear = np.nan
            else:
                valid = merged[[model, poll_col]].dropna()
                if len(valid) < 5:
                    pear = spear = np.nan
                else:
                    pear = pearsonr(valid[model], valid[poll_col])[0]
                    spear = spearmanr(valid[model], valid[poll_col])[0]

            results.append(
                {
                    "party": party,
                    "model": model,
                    "pearson": pear,
                    "spearman": spear,
                }
            )

    return pd.DataFrame(results)

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.header("Configuration")

    MODEL_MAP = {
        "Transformer": "trans_pos_ratio",
        "VADER": "vader_avg",
    }

    model_display = st.selectbox("Sentiment Model", list(MODEL_MAP.keys()))
    model = MODEL_MAP[model_display]

    party = st.selectbox("Select Party", list(PARTIES.keys()))
    timescale = st.selectbox("Timescale", ["Weekly", "Daily"])
    num_comments = st.slider("Example Comments", 3, 50, 10)
# ============================================
# Main Layout
# ============================================
st.title("🇨🇦 Election Sentiment Dashboard (2025)")
tab1, tab2, tab3 = st.tabs(["📊 Summary", "📈 Analysis", "📚 Data"])

# ======================================
# TAB 1 — SUMMARY
# ======================================
with tab1:
    st.header("Sentiment vs Polls Summary")

    poll_col = PARTIES[party]
    ELECTION_DAY = pd.Timestamp("2025-04-27")

    # --------------------------------------------
    # DAILY — clip by date
    # --------------------------------------------
    if timescale == "Daily":
        df_sent = sent_daily[
            (sent_daily["party"] == party) &
            (sent_daily["date"] <= ELECTION_DAY)
        ].copy()

        df_merge = df_sent.merge(poll_daily, on="date", how="left")
        df_merge = df_merge.sort_values("date")
        df_merge["x_axis"] = df_merge["date"]

    # --------------------------------------------
    # WEEKLY — clip by week (NOT by date!)
    # --------------------------------------------
    else:
        df_sent = sent_weekly[
            (sent_weekly["party"] == party) &
            (sent_weekly["week"] <= 17)
        ].copy()

        df_merge = df_sent.merge(poll_weekly, on="week", how="left")
        df_merge = df_merge.sort_values("week")

        # Convert week to an artificial week-start date
        df_merge["x_axis"] = (
            pd.to_datetime("2025-01-06") +
            pd.to_timedelta((df_merge["week"] - 1) * 7 , unit="days")
        )

    # Stop if empty
    if df_merge.empty:
        st.error("No merged data available for the selected configuration.")
        st.stop()

    # --------------------------------------------
    # Interactive Plotly figure
    # --------------------------------------------
    fig = go.Figure()

    # Sentiment line (left axis)
    fig.add_trace(
        go.Scatter(
            x=df_merge["x_axis"],
            y=df_merge[model],
            mode="lines+markers",
            name="Sentiment",
            line=dict(color="steelblue"),
            hovertemplate="Date: %{x}<br>Sentiment: %{y:.4f}<extra></extra>",
        )
    )

    # Polling line (right axis)
    fig.add_trace(
        go.Scatter(
            x=df_merge["x_axis"],
            y=df_merge[poll_col],
            mode="lines+markers",
            name="Polling %",
            yaxis="y2",
            line=dict(color="crimson"),
            hovertemplate="Date: %{x}<br>Polling: %{y:.2f}%<extra></extra>",
        )
    )

    # --------------------------------------------
    # Election Day vertical line
    # --------------------------------------------
    fig.add_shape(
        type="line",
        x0=ELECTION_DAY,
        x1=ELECTION_DAY,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="black", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=ELECTION_DAY,
        y=1.02,
        xref="x",
        yref="paper",
        text="Election Day",
        showarrow=False,
        align="left",
    )

    # --------------------------------------------
    # Final election results (scatter point)
    # --------------------------------------------
    final_results = {"lpc": 42, "cpc": 38, "ndp": 17}

    if poll_col in final_results:
        fig.add_trace(
            go.Scatter(
                x=[ELECTION_DAY],
                y=[final_results[poll_col]],
                mode="markers+text",
                name=f"Final Result ({final_results[poll_col]}%)",
                marker=dict(size=12, color="black"),
                text=[f"{final_results[poll_col]}%"],
                textposition="top center",
                yaxis="y2",
                hovertemplate="Final Election Result: %{y}%<extra></extra>",
            )
        )

    fig.update_layout(
        height=450,
        title=f"{party} — Sentiment vs Polls ({timescale})",
        xaxis_title="Date",
        yaxis=dict(title="Sentiment"),
        yaxis2=dict(
            title="Polling %",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------
    # Comment volume (if available)
    # --------------------------------------------
    if "comment_volume" in df_merge.columns:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=df_merge["x_axis"],
                y=df_merge["comment_volume"],
                mode="lines+markers",
                line=dict(color="orange"),
                hovertemplate="Date: %{x}<br>Comments: %{y}<extra></extra>",
            )
        )
        fig2.update_layout(
            height=350,
            title="Comment Volume Over Time",
            xaxis_title="Date",
            yaxis_title="Comment Count",
        )
        st.plotly_chart(fig2, use_container_width=True)
# ============================================
# TAB 2 — Analysis
# ============================================
with tab2:
    st.header("Statistical Analysis")

    # Correlation tables (lag 0, consistent with offline script)
    corr_daily = compute_corr_table_daily(sent_daily, poll_daily)
    corr_weekly = compute_corr_table_weekly(sent_weekly, poll_weekly)

    st.subheader("Daily Correlations (Pearson & Spearman)")
    st.dataframe(corr_daily)

    st.subheader("Weekly Correlations (Pearson & Spearman)")
    st.dataframe(corr_weekly)

    # Heatmaps
    st.subheader("Correlation Heatmaps (Pearson)")

    def heatmap(df_corr: pd.DataFrame, title: str):
        if df_corr.empty:
            st.warning("No data for heatmap.")
            return
        pivot = df_corr.pivot(index="party", columns="model", values="pearson")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot, cmap="coolwarm", vmin=-1, vmax=1, annot=True, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)

    heatmap(corr_daily, "Daily Pearson Correlation")
    heatmap(corr_weekly, "Weekly Pearson Correlation")

    # Regression scatter plot
    st.subheader("Regression Scatter Plot (Sentiment vs Poll)")

    poll_col = PARTIES[party]
    if timescale == "Daily":
        df_reg = sent_daily[sent_daily["party"] == party].merge(
            poll_daily, on="date", how="inner"
        )
    else:
        df_reg = sent_weekly[sent_weekly["party"] == party].merge(
            poll_weekly, on="week", how="inner"
        )

    if df_reg.empty:
        st.warning("Not enough data for regression plot.")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.regplot(x=df_reg[model], y=df_reg[poll_col], ax=ax)
        ax.set_xlabel(model)
        ax.set_ylabel(f"{party} Polling (%)")
        ax.set_title(f"{party} — {model} vs Polling")
        st.pyplot(fig)

# ============================================
# TAB 3 — Raw Data
# ============================================

with tab3:
    st.header("Raw Data Samples")

    # ==========================
    # 1. Example Comments
    # ==========================
    st.subheader("Example Comments (Filtered by Party)")
    if "party" in scored_df.columns:
        df_c = scored_df[scored_df["party"] == party]
        if not df_c.empty:
            cols = [
                "date", "party",
                "targeted_sentence",
                "trans_sentiment",
                "vader_score",
            ]
            cols = [c for c in cols if c in df_c.columns]
            st.dataframe(df_c.sample(min(num_comments, len(df_c)))[cols])
        else:
            st.warning(f"No comments found for {party}.")
    else:
        st.warning("Scored dataset is missing 'party' column.")

    # ==========================
    # 2. Sentiment Data (Daily OR Weekly)
    # ==========================
    if timescale == "Daily":
        st.subheader(f"Daily Sentiment — {party}")
        df_daily_party = sent_daily[sent_daily["party"] == party]
        st.dataframe(df_daily_party.head(50))
    else:
        st.subheader(f"Weekly Sentiment — {party}")
        df_weekly_party = sent_weekly[sent_weekly["party"] == party]
        st.dataframe(df_weekly_party.head(50))

    # ==========================
    # 3. Polling Data (Daily OR Weekly)
    # ==========================
    poll_col = PARTIES[party]  # e.g., "lpc"

    if timescale == "Daily":
        st.subheader(f"Daily Polling — {party}")
        poll_daily_party = poll_daily[["date", poll_col]].dropna().sort_values("date")
        st.dataframe(poll_daily_party.head(50))
    else:
        st.subheader(f"Weekly Polling — {party}")
        poll_weekly_party = poll_weekly[["week", poll_col]].dropna().sort_values("week")
        st.dataframe(poll_weekly_party.head(50))