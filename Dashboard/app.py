import streamlit as st
import pandas as pd
from pathlib import Path

# --- Diagnostic dashboard (v1) ---
st.set_page_config(page_title="US Market Diagnostic Dashboard", layout="wide")

# Resolve paths relative to this file
# app.py is in: <project_root>/Dashboard/app.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # go up from Dashboard/ to project root
DATA_DIR = PROJECT_ROOT / "Dashboard" / "dashboard_datas"  # CSVs are inside Dashboard/dashboard_datas

TIME_SERIES_PATH = DATA_DIR / "time_series.csv"
EVENTS_PATH = DATA_DIR / "events.csv"
LEADLAG_PATH = DATA_DIR / "leadlag_corr.csv"
ML_SUMMARY_PATH = DATA_DIR / "ml_model_summary_final.csv"
DOWN_STATS_PATH = DATA_DIR / "downturn_stats.csv"
RIDGE_CM_PATH = DATA_DIR / "ridge_confusion_matrix.csv"


# --- Header ---
st.title("US Market Diagnostic Dashboard")
st.caption(
    "We built this dashboard not to pretend we can time markets, "
    "but to transparently show how cycles and crashes look in the data — "
    "and where machine learning fails to extract a robust edge."
)

# --- Navigation tabs (diagnostic structure) ---
tab_home, tab_market, tab_leadlag, tab_ml = st.tabs(
    ["Home", "Market Lab", "Lead–Lag Lab", "ML Reality Check"]
)


# --- Guard: require core CSVs ---
if not TIME_SERIES_PATH.exists() or not LEADLAG_PATH.exists():
    st.error("Missing core CSV files. Check folder names and file locations.")
    st.stop()

# Load data
ts = pd.read_csv(TIME_SERIES_PATH, parse_dates=["date"]).sort_values("date")
ccf = pd.read_csv(LEADLAG_PATH)
events = pd.read_csv(EVENTS_PATH, parse_dates=["start_date", "end_date"]) if EVENTS_PATH.exists() else None


# Load ML summary (optional but expected)
ml_summary = pd.read_csv(ML_SUMMARY_PATH) if ML_SUMMARY_PATH.exists() else None
down_stats = pd.read_csv(DOWN_STATS_PATH) if DOWN_STATS_PATH.exists() else None
ridge_cm = pd.read_csv(RIDGE_CM_PATH, index_col=0) if RIDGE_CM_PATH.exists() else None


# =========================
# Home
# =========================
with tab_home:
    st.subheader("Diagnostic overview")

    st.info(
        "This dashboard is diagnostic (not predictive). It visualizes market co-movement, lead–lag behavior, "
        "and why rare downturn prediction is inherently difficult with this feature set."
    )

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Monthly observations", f"{len(ts):,}")

    with col2:
        # Lead–lag: strongest absolute correlation across all pairs/lags
        best_global = ccf.iloc[(ccf["correlation"].abs()).argmax()]
        st.metric("Max |lag corr|", f"{abs(best_global['correlation']):.3f}")

    with col3:
        st.metric("Best lag (months)", int(best_global["lag"]))

    with col4:
        if ml_summary is not None and "ROC_AUC" in ml_summary.columns:
            st.metric("Best ROC-AUC (ML)", f"{float(ml_summary['ROC_AUC'].max()):.3f}")
        else:
            st.metric("Best ROC-AUC (ML)", "N/A")

    st.caption(
        "Interpretation: co-movement exists, but stable lead–lag structure is weak; ML shows limited edge "
        "under class imbalance and low signal-to-noise."
    )

    # --- Quick evidence table (top 10 lag correlations by abs value) ---
    top10 = ccf.assign(abs_corr=ccf["correlation"].abs()).sort_values("abs_corr", ascending=False).head(10)
    top10 = top10[["var_x", "var_y", "lag", "correlation", "n_obs"]]

    with st.expander("Top 10 lag correlations (by absolute value)", expanded=False):
        st.dataframe(top10, use_container_width=True)

# =========================
# Market Lab
# =========================
with tab_market:
    st.subheader("Market Lab")

    st.caption("Monthly log-differences. Use rolling correlation to inspect co-movement stability.")

    # --- Controls ---
    series_cols = ["DJT", "SP500", "housing"]
    colA, colB, colC = st.columns([2, 2, 2])

    with colA:
        x_var = st.selectbox("Series X", series_cols, index=0)
    with colB:
        y_var = st.selectbox("Series Y", series_cols, index=1)
    with colC:
        window = st.selectbox("Rolling window (months)", [6, 12, 24, 36], index=1)

    # --- Plots ---
    st.subheader("Market Dynamics")
    import matplotlib.pyplot as plt

    # Plot with shaded crisis windows (matplotlib)
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_df = ts.set_index("date")[series_cols]
    plot_df.plot(ax=ax)

    ax.set_title("Market Dynamics (log-diff monthly) with crisis windows")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log-diff")

    # Add shaded spans for events
    if events is not None and len(events) > 0:
        for _, r in events.iterrows():
            ax.axvspan(r["start_date"], r["end_date"], alpha=0.15, label=r["label"])

        # Prevent duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq.append(h)
                uniq_labels.append(l)
                seen.add(l)
        ax.legend(uniq, uniq_labels, loc="upper left", fontsize=9)

    plt.tight_layout()
    
    # Save the figure for the report (high-resolution PNG)
    FIG_DIR = PROJECT_ROOT / "Dashboard" / "dashboard_figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_market_dynamics_with_events.png", dpi=300, bbox_inches="tight")  # Export for report
    
    st.pyplot(fig)
    plt.close(fig)


    # Rolling correlation
    roll = ts.set_index("date")[[x_var, y_var]].rolling(window=window).corr()
    roll_corr = roll.loc[(slice(None), y_var), :].droplevel(1)[x_var]
    roll_corr = roll_corr.rename("rolling_corr").dropna()

    st.subheader(f"Rolling correlation: {x_var} vs {y_var} ({window}m)")
    st.line_chart(roll_corr.to_frame())


# =========================
# Lead–Lag Lab
# =========================
with tab_leadlag:
    st.subheader("Lead–Lag Lab")

    # --- Controls ---
    # Build pair options like "DJT → SP500"
    pairs = (
        ccf[["var_x", "var_y"]]
        .drop_duplicates()
        .sort_values(["var_x", "var_y"])
        .apply(lambda r: f"{r['var_x']} → {r['var_y']}", axis=1)
        .tolist()
    )

    selected_pair = st.selectbox("Select a pair", pairs, index=0)  # English label is fine in UI
    var_x, var_y = [s.strip() for s in selected_pair.split("→")]

    min_lag = int(ccf["lag"].min())
    max_lag = int(ccf["lag"].max())

    lag_range = st.slider(
        "Lag range (months)",
        min_value=min_lag,
        max_value=max_lag,
        value=(min_lag, max_lag),
        step=1
    )

    # --- Filter data for selection ---
    sub = ccf[(ccf["var_x"] == var_x) & (ccf["var_y"] == var_y)].copy()
    sub = sub[(sub["lag"] >= lag_range[0]) & (sub["lag"] <= lag_range[1])].sort_values("lag")

    # --- KPIs ---
    # Find the lag with the strongest absolute correlation
    best = sub.iloc[(sub["correlation"].abs()).argmax()]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best lag (months)", int(best["lag"]))
    with col2:
        st.metric("Correlation at best lag", f"{best['correlation']:.3f}")
    with col3:
        st.metric("Obs used (n)", int(best["n_obs"]))

    # --- Plot ---
    st.subheader("Correlation vs lag")
    st.line_chart(sub.set_index("lag")[["correlation"]])
    # Top lags table (strongest absolute correlations)
    top5 = sub.assign(abs_corr=sub["correlation"].abs()).sort_values("abs_corr", ascending=False).head(5)
    top5 = top5[["lag", "correlation", "n_obs"]]

    st.subheader("Top 5 lags (by |correlation|)")
    st.dataframe(top5, use_container_width=True)



# =========================
# ML Reality Check
# =========================
with tab_ml:
    st.subheader("Downturn rarity (class imbalance)")
    if down_stats is None:
        st.warning("downturn_stats.csv not found in DATA_DIR.")
    else:
        # Expect rows for class 0 and 1
        c0 = down_stats.loc[down_stats["class"] == 0, "count"].values[0]
        c1 = down_stats.loc[down_stats["class"] == 1, "count"].values[0]
        p1 = down_stats.loc[down_stats["class"] == 1, "share_pct"].values[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Non-downturn (0) count", int(c0))
        with col2:
            st.metric("Downturn (1) count", int(c1))
        with col3:
            st.metric("Downturn share", f"{p1:.2f}%")

        with st.expander("Show downturn_stats.csv", expanded=False):
            st.dataframe(down_stats, use_container_width=True)

    st.subheader("Ridge confusion matrix")
    if ridge_cm is None:
        st.warning("ridge_confusion_matrix.csv not found in DATA_DIR.")
    else:
        st.dataframe(ridge_cm, use_container_width=True)

    st.subheader("ML Summary (Final)")
    if ml_summary is None:
        st.warning("ml_model_summary_final.csv not found in DATA_DIR.")
    else:
        st.dataframe(ml_summary, use_container_width=True)

