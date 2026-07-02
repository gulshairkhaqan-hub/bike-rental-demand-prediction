"""Streamlit — Bike Rental Demand Prediction
Dark-themed, Plotly charts, number_input controls, animated cards.
"""
from __future__ import annotations
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import train_model
import utils

# ── brand palette ───────────────────────────────────────────────────────────
C_BG       = "#0F172A"   # page background  (deep navy)
C_CARD     = "#1E293B"   # card / panel background
C_BORDER   = "#334155"   # subtle borders
C_PRIMARY  = "#38BDF8"   # sky-blue accent
C_TEXT     = "#E2E8F0"   # body text
C_MUTED    = "#94A3B8"   # secondary text
C_GREEN    = "#22C55E"
C_AMBER    = "#F59E0B"
C_RED      = "#EF4444"
C_PLOT_BG  = "#1E293B"
C_GRID     = "#334155"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C_PLOT_BG, plot_bgcolor=C_PLOT_BG,
    font=dict(color="#E2E8F0", family="Inter, sans-serif", size=13),
    title=dict(font=dict(color="#FFFFFF", size=15, family="Inter, sans-serif")),
    xaxis=dict(gridcolor=C_GRID, showline=False, tickfont=dict(color="#E2E8F0"), title_font=dict(color="#E2E8F0")),
    yaxis=dict(gridcolor=C_GRID, showline=False, title="", tickfont=dict(color="#E2E8F0"), title_font=dict(color="#E2E8F0")),
    legend=dict(font=dict(color="#E2E8F0")),
    margin=dict(l=40, r=20, t=55, b=40),
)
# Use this helper instead of passing yaxis= again in update_layout (prevents duplicate key error)
def _layout(**extra):
    """Return PLOTLY_LAYOUT merged with extra overrides safely."""
    base = dict(PLOTLY_LAYOUT)
    if "yaxis" in extra:
        base["yaxis"] = {**base["yaxis"], **extra.pop("yaxis")}
    if "xaxis" in extra:
        base["xaxis"] = {**base["xaxis"], **extra.pop("xaxis")}
    if "title" in extra and isinstance(extra["title"], dict):
        base["title"] = {**base.get("title", {}), **extra.pop("title")}
    base.update(extra)
    return base

DATASET_PATH  = "dataset/bike_rental_100_rows.csv"
SEASON_LABELS = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="📊",
    layout="wide",
)

# ── global dark CSS ──────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {{
    background-color: {C_BG} !important;
    color: {C_TEXT};
    font-family: 'Inter', sans-serif;
}}
[data-testid="stSidebar"] {{
    background: {C_CARD} !important;
    border-right: 1px solid {C_BORDER};
}}
[data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}

/* ── metric cards — fadeUp + hover lift ── */
[data-testid="stMetric"] {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 16px 20px;
    border-left: 3px solid {C_PRIMARY};
    animation: fadeUp 0.5s ease both;
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease !important;
    cursor: default;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-5px) !important;
    box-shadow: 0 10px 30px rgba(56,189,248,0.18) !important;
    border-color: {C_PRIMARY} !important;
}}
@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(14px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
[data-testid="stMetricLabel"] {{ color: {C_MUTED} !important; font-size:0.8rem; }}
[data-testid="stMetricValue"] {{ color: {C_TEXT}  !important; font-size:1.9rem; font-weight:700; }}

/* ── tabs ── */
[data-testid="stTabs"] button {{
    color: {C_MUTED} !important;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 8px 16px;
    border-radius: 6px 6px 0 0;
    transition: color .2s ease, background .2s ease, transform .15s ease !important;
}}
[data-testid="stTabs"] button:hover {{
    color: {C_PRIMARY} !important;
    background: rgba(56,189,248,0.08) !important;
    transform: translateY(-2px) !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C_PRIMARY} !important;
    border-bottom: 2px solid {C_PRIMARY} !important;
    background: rgba(56,189,248,0.05) !important;
}}
/* tab panel fade-in */
[data-testid="stTabsContent"] {{
    animation: fadeUp 0.35s ease both;
}}

/* number inputs */
input[type="number"] {{
    background: {C_CARD} !important;
    color: {C_TEXT} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 8px !important;
}}
/* number input container — match selectbox style */
[data-testid="stNumberInput"] > div {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 8px !important;
}}
/* style the + / - buttons */
[data-testid="stNumberInputStepDown"],
[data-testid="stNumberInputStepUp"] {{
    background: {C_BORDER} !important;
    color: {C_TEXT} !important;
    border: none !important;
    border-radius: 6px !important;
    transition: background .15s ease !important;
}}
[data-testid="stNumberInputStepDown"]:hover,
[data-testid="stNumberInputStepUp"]:hover {{
    background: {C_PRIMARY} !important;
    color: #0F172A !important;
}}

/* selectbox */
[data-baseweb="select"] > div {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 8px !important;
    color: {C_TEXT} !important;
}}
[data-baseweb="popover"] li {{
    background: {C_CARD} !important;
    color: {C_TEXT} !important;
}}

/* buttons */
[data-testid="stButton"] > button {{
    background: {C_PRIMARY};
    color: #0F172A;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    transition: opacity .2s;
}}
[data-testid="stButton"] > button:hover {{ opacity: 0.85; }}

/* ── dark dataframe / table ── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] iframe,
.dvn-scroller, .stDataFrame {{
    background: {C_CARD} !important;
    border-radius: 10px !important;
    color: {C_TEXT} !important;
}}
/* force iframe content dark (best-effort via CSS vars) */
[data-testid="stDataFrame"] * {{
    color: {C_TEXT} !important;
    background-color: transparent !important;
}}

/* ── hover lift animation on custom HTML cards ── */
.hover-card {{
    transition: transform .22s ease, box-shadow .22s ease !important;
}}
.hover-card:hover {{
    transform: translateY(-6px) !important;
    box-shadow: 0 12px 36px rgba(56,189,248,0.18) !important;
}}

/* ── hide Streamlit's chart expand/fullscreen button ──
   The expand view opens a white Streamlit header overlay that cannot be
   themed via CSS. Plotly's own zoom/pan tools are sufficient. */
button[title="View fullscreen"],
button[data-testid="StyledFullScreenButton"] {{
    display: none !important;
}}

/* expander dark */
[data-testid="stExpander"] {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
}}
[data-testid="stExpander"] summary {{
    color: {C_TEXT} !important;
}}

/* ── fullscreen / expanded chart mode ── */
/* removed — replaced by hiding the expand button above */
[data-testid="stCaptionContainer"] p {{ color: {C_MUTED} !important; }}

/* divider */
hr {{ border-color: {C_BORDER}; }}

/* headings */
h1 {{ color: {C_TEXT} !important; font-weight:800; }}
h2, h3 {{ color: {C_PRIMARY} !important; font-weight:700; }}
p, li {{ color: {C_TEXT}; }}

/* ── page-level fadeIn ── */
[data-testid="stMain"] > div > div {{
    animation: fadeUp 0.4s ease both;
}}

/* ── sidebar — clean nav ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div:first-child {{
    display: none !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {{
    display: flex !important;
    align-items: center !important;
    padding: 9px 14px !important;
    border-radius: 8px !important;
    margin: 1px 0 !important;
    cursor: pointer !important;
    transition: background .18s ease, padding-left .2s ease !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
    background: rgba(56,189,248,0.1) !important;
    padding-left: 20px !important;
}}
/* nav link text */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:last-child p {{
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: {C_TEXT} !important;
    margin: 0 !important;
}}
/* active item highlight */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {{
    background: rgba(56,189,248,0.12) !important;
    border-left: 3px solid {C_PRIMARY} !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] p {{
    color: {C_PRIMARY} !important;
}}

/* ── buttons — scale on hover ── */
[data-testid="stButton"] > button {{
    transition: opacity .2s ease, transform .18s ease, box-shadow .18s ease !important;
}}
[data-testid="stButton"] > button:hover {{
    opacity: 0.9 !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 20px rgba(56,189,248,0.3) !important;
}}
[data-testid="stButton"] > button:active {{
    transform: scale(0.97) !important;
}}

/* ── download button ── */
[data-testid="stDownloadButton"] > button {{
    background: {C_CARD} !important;
    border: 1px solid {C_PRIMARY} !important;
    color: {C_PRIMARY} !important;
    border-radius: 8px;
    font-weight: 600;
    transition: background .2s, transform .18s, box-shadow .18s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: rgba(56,189,248,0.12) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(56,189,248,0.2) !important;
}}

/* ── expander — fade in ── */
[data-testid="stExpander"] {{
    animation: fadeUp 0.4s ease both;
}}

/* ── alerts (success / warning / info) — slide in ── */
[data-testid="stAlert"] {{
    animation: fadeUp 0.35s ease both;
    border-radius: 10px !important;
}}

/* ── dataframe — fade in ── */
[data-testid="stDataFrame"] {{
    animation: fadeUp 0.45s ease both;
}}

/* ── plotly chart container — fade in ── */
[data-testid="stPlotlyChart"] {{
    animation: fadeUp 0.5s ease both;
}}

/* ── selectbox hover ── */
[data-baseweb="select"] > div {{
    transition: border-color .18s ease, box-shadow .18s ease !important;
}}
[data-baseweb="select"] > div:hover {{
    border-color: {C_PRIMARY} !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}}

/* ── number input hover glow ── */
[data-testid="stNumberInput"] > div {{
    transition: border-color .18s ease, box-shadow .18s ease !important;
}}
[data-testid="stNumberInput"] > div:focus-within {{
    border-color: {C_PRIMARY} !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.2) !important;
}}

/* ── staggered delays for multiple metric cards ── */
[data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetric"] {{ animation-delay: 0.0s; }}
[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] {{ animation-delay: 0.1s; }}
[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] {{ animation-delay: 0.2s; }}
[data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] {{ animation-delay: 0.3s; }}

/* ── hover-card class (used on HTML div cards) ── */
.hover-card {{
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease !important;
}}
.hover-card:hover {{
    transform: translateY(-7px) !important;
    box-shadow: 0 14px 40px rgba(56,189,248,0.2) !important;
}}

/* ── Plotly chart iframe — remove white bg and fix height ── */
[data-testid="stPlotlyChart"] > div {{
    background: transparent !important;
}}
[data-testid="stPlotlyChart"] iframe {{
    background: transparent !important;
    display: block !important;
}}
/* Hide the empty modebar container that causes white space */
.modebar-container {{
    background: transparent !important;
}}
.js-plotly-plot .plotly .modebar {{
    background: transparent !important;
}}
/* Remove extra bottom padding Streamlit adds under charts */
[data-testid="stPlotlyChart"] {{
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}}
</style>
""", unsafe_allow_html=True)

# ── dark HTML table helper (replaces st.dataframe for dark theme) ────────────
def _dark_table(df: pd.DataFrame) -> None:
    """Render a DataFrame as a dark-themed HTML table."""
    rows_html = ""
    for _, row in df.iterrows():
        cells = "".join(
            f"<td style='padding:9px 14px;border-bottom:1px solid {C_BORDER};"
            f"color:{C_TEXT};font-size:0.88rem;'>{v}</td>"
            for v in row
        )
        rows_html += f"<tr style='transition:background .15s;' onmouseover=\"this.style.background='rgba(56,189,248,0.06)'\" onmouseout=\"this.style.background='transparent'\">{cells}</tr>"

    headers = "".join(
        f"<th style='padding:10px 14px;text-align:left;color:{C_PRIMARY};"
        f"font-size:0.8rem;font-weight:700;letter-spacing:.05em;"
        f"border-bottom:2px solid {C_BORDER};text-transform:uppercase;'>{c}</th>"
        for c in df.columns
    )

    st.markdown(f"""
    <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;
                overflow-x:auto;animation:fadeUp 0.45s ease both;">
        <table style="width:100%;border-collapse:collapse;">
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)
@st.cache_data
def load_dataset() -> pd.DataFrame:
    return utils.clean_data(utils.load_data(DATASET_PATH))


# ── reusable plotly config ───────────────────────────────────────────────────
# Remove the fullscreen button — it opens a white modal in Streamlit's dark theme.
# Users can still zoom/pan interactively inside the chart.
PLOTLY_CONFIG = {
    "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
    "displaylogo": False,
    "modeBarStyle": {"backgroundColor": C_CARD, "color": "#E2E8F0"},
}

def _fig(fig, **extra):
    """Apply shared dark layout with optional overrides and return fig."""
    fig.update_layout(**_layout(**extra))
    return fig


# ============================================================================
# PAGE 1 — HOME
# ============================================================================
def show_home(df: pd.DataFrame) -> None:
    # Hero banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1E3A5F 0%,#2563EB 100%);
                border-radius:16px;padding:44px 52px;margin-bottom:32px;">
        <div style="font-size:0.85rem;color:#93C5FD;letter-spacing:.1em;
                    font-weight:600;margin-bottom:10px;">ML PORTFOLIO PROJECT</div>
        <div style="color:#fff;font-size:2.4rem;font-weight:800;
                    line-height:1.15;margin-bottom:14px;">
            Bike Rental Demand<br>Prediction
        </div>
        <div style="color:#CBD5E1;font-size:1rem;max-width:620px;line-height:1.7;">
            Predict daily bike rental demand from weather and seasonal conditions
            using a <strong style="color:#fff;">Decision Tree Regression</strong> model.
            Explore data, visualize patterns, train the model, and get live forecasts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",     len(df))
    c2.metric("Features Used",     4,    help="temp · humidity · windspeed · season")
    c3.metric("Avg Rentals / Day", f"{df['count'].mean():.0f}")
    c4.metric("Max Rentals / Day", int(df["count"].max()))

    st.markdown("---")

    # Season bar chart — Plotly
    st.subheader("Average Rentals by Season")
    season_avg = (df.groupby("season")["count"].mean().reset_index()
                    .assign(Season=lambda d: d["season"].map(SEASON_LABELS)))
    colors = ["#22C55E","#F59E0B","#EF4444","#38BDF8"]
    fig = go.Figure(go.Bar(
        x=season_avg["Season"], y=season_avg["count"],
        marker_color=colors, text=season_avg["count"].round(0).astype(int),
        textposition="outside", textfont=dict(color=C_TEXT, size=12),
    ))
    fig.update_layout(**_layout(
        title="Average Bike Rentals per Season",
        bargap=0.35, showlegend=False,
        yaxis=dict(title="Avg Count"),
    ))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")

    # How-to cards
    st.subheader("How to use this app")
    cards = [
        ("#2563EB","📂","Dataset",       "Preview rows, types, missing values & stats."),
        ("#7C3AED","📊","Visualize",     "Charts, heatmap, weather impact analysis."),
        ("#0891B2","⚙️","Train Model",   "Train Decision Tree, view MAE / R² / RMSE."),
        ("#059669","🔮","Predict",       "Enter conditions → instant demand forecast."),
    ]
    cols = st.columns(4)
    for col, (color, icon, title, desc) in zip(cols, cards):
        col.markdown(f"""
        <div class="hover-card" style="background:{C_CARD};border-radius:12px;padding:22px 18px;
                    border-top:4px solid {color};cursor:default;">
            <div style="font-size:1.6rem;margin-bottom:8px;">{icon}</div>
            <div style="font-weight:700;color:{C_PRIMARY};font-size:1rem;
                        margin-bottom:6px;">{title}</div>
            <div style="font-size:0.82rem;color:{C_MUTED};line-height:1.5;">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ============================================================================
# PAGE 2 — DATASET
# ============================================================================
def show_dataset(df: pd.DataFrame) -> None:
    st.title("Dataset Explorer")
    st.caption(f"Source: `{DATASET_PATH}`")
    info = utils.get_dataset_info(df)

    t1, t2, t3, t4 = st.tabs(["Preview","Shape & Types","Missing Values","Statistics"])

    with t1:
        n = st.number_input("Rows to display", min_value=5, max_value=len(df), value=10, step=5)
        _dark_table(df.head(int(n)))

    with t2:
        rows, cols = info["shape"]
        ca, cb = st.columns(2)
        ca.metric("Rows", rows); cb.metric("Columns", cols)
        dtype_df = pd.DataFrame({
            "Column":    list(info["dtypes"].keys()),
            "Data Type": [str(v) for v in info["dtypes"].values()],
        })
        _dark_table(dtype_df)

    with t3:
        mv = pd.DataFrame({
            "Column":        list(info["missing_values"].keys()),
            "Missing Count": list(info["missing_values"].values()),
        })
        mv["Status"] = mv["Missing Count"].apply(lambda x: "✅ None" if x == 0 else f"⚠️ {x}")
        _dark_table(mv)
        if mv["Missing Count"].sum() == 0:
            st.success("No missing values in any column.")
        if info["duplicates"] == 0:
            st.success("No duplicate rows found.")
        else:
            st.warning(f"{info['duplicates']} duplicate row(s) removed during cleaning.")

    with t4:
        _dark_table(df.describe().round(2).reset_index().rename(columns={"index": "stat"}))


# ============================================================================
# PAGE 3 — VISUALIZATIONS  (all Plotly)
# ============================================================================
def show_visualizations(df: pd.DataFrame) -> None:
    st.title("Visualizations")

    t1, t2, t3, t4, t5 = st.tabs(
        ["Distributions","Box Plots","Scatter","Correlation","Weather Impact"]
    )

    # ── Distributions ───────────────────────────────────────────────────────
    with t1:
        st.subheader("Feature Distributions")
        feat = st.selectbox("Feature", ["temp","humidity","windspeed","count"],
                            format_func=str.capitalize, key="dist_feat")
        fig = px.histogram(df, x=feat, nbins=20,
                           color_discrete_sequence=[C_PRIMARY],
                           title=f"Distribution of {feat.capitalize()}")
        fig.update_traces(marker_line_color=C_BG, marker_line_width=1)
        st.plotly_chart(_fig(fig), use_container_width=True, config=PLOTLY_CONFIG)

    # ── Box Plots ────────────────────────────────────────────────────────────
    with t2:
        st.subheader("Box Plots")
        df2 = df.copy()
        df2["Season"] = df2["season"].map(SEASON_LABELS)

        fig1 = px.box(df2, x="Season", y="count",
                      color="Season",
                      color_discrete_sequence=[C_GREEN, C_AMBER, C_RED, C_PRIMARY],
                      title="Rental Count by Season",
                      category_orders={"Season":["Spring","Summer","Fall","Winter"]})
        st.plotly_chart(_fig(fig1), use_container_width=True, config=PLOTLY_CONFIG)

        feat2 = st.selectbox("Weather feature", ["temp","humidity","windspeed"],
                             format_func=str.capitalize, key="box_feat")
        fig2 = px.box(df, y=feat2, color_discrete_sequence=[C_PRIMARY],
                      title=f"Outlier Check — {feat2.capitalize()}")
        st.plotly_chart(_fig(fig2), use_container_width=True, config=PLOTLY_CONFIG)

    # ── Scatter ──────────────────────────────────────────────────────────────
    with t3:
        st.subheader("Feature vs Rental Count")
        feat3 = st.selectbox("X-axis feature", ["temp","humidity","windspeed"],
                             format_func=str.capitalize, key="scat_feat")
        # Manual trend line — no statsmodels dependency
        m_coef, b_coef = np.polyfit(df[feat3], df["count"], 1)
        x_line = np.linspace(df[feat3].min(), df[feat3].max(), 100)
        y_line = m_coef * x_line + b_coef

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df[feat3], y=df["count"], mode="markers",
            marker=dict(color=C_PRIMARY, size=8, opacity=0.8),
            name="Data points",
        ))
        fig3.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color=C_AMBER, width=2, dash="dash"),
            name="Trend",
        ))
        fig3.update_layout(**_layout(
            title=f"{feat3.capitalize()} vs Rental Count",
            xaxis=dict(title=feat3.capitalize()),
            yaxis=dict(title="Rental Count"),
        ))
        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Correlation ──────────────────────────────────────────────────────────
    with t4:
        st.subheader("Correlation Heatmap")
        corr_cols  = ["temp","humidity","windspeed","season","count"]
        corr       = utils.get_correlation_matrix(df, corr_cols).round(2)
        fig4 = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0, text=corr.values,
            texttemplate="%{text:.2f}", showscale=True,
        ))
        fig4.update_layout(**_layout(title="Correlation Matrix"))
        st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CONFIG)

        top = corr["count"].drop("count").abs().sort_values(ascending=False)
        st.info(f"**{top.index[0].capitalize()}** has the strongest correlation with "
                f"rental count (|r| = {top.iloc[0]:.2f}).")

    # ── Weather Impact ───────────────────────────────────────────────────────
    with t5:
        st.subheader("Weather Impact on Rentals")
        dw = df.copy()
        dw["Temp Range"] = pd.cut(dw["temp"], bins=[0,.25,.5,.75,1.0],
            labels=["Cold (0–.25)","Cool (.25–.5)","Warm (.5–.75)","Hot (.75–1)"])
        dw["Humidity Range"] = pd.cut(dw["humidity"], bins=[0,.33,.66,1.0],
            labels=["Low (0–.33)","Medium (.33–.66)","High (.66–1)"])

        ca, cb = st.columns(2)
        with ca:
            ta = dw.groupby("Temp Range", observed=True)["count"].mean().reset_index()
            fig5 = px.bar(ta, x="Temp Range", y="count",
                          color="Temp Range",
                          color_discrete_sequence=["#93C5FD","#60A5FA","#F59E0B","#EF4444"],
                          text="count", title="Avg Rentals by Temperature")
            fig5.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            st.plotly_chart(_fig(fig5), use_container_width=True, config=PLOTLY_CONFIG)
        with cb:
            ha = dw.groupby("Humidity Range", observed=True)["count"].mean().reset_index()
            fig6 = px.bar(ha, x="Humidity Range", y="count",
                          color="Humidity Range",
                          color_discrete_sequence=[C_GREEN, C_AMBER, C_PRIMARY],
                          text="count", title="Avg Rentals by Humidity")
            fig6.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            st.plotly_chart(_fig(fig6), use_container_width=True, config=PLOTLY_CONFIG)


# ============================================================================
# PAGE 4 — MODEL TRAINING  (all Plotly)
# ============================================================================
def show_model_training(df: pd.DataFrame) -> None:
    st.title("Model Training")

    with st.expander("Model Configuration", expanded=True):
        cfg = pd.DataFrame({
            "Parameter": ["Algorithm","Features","Target","Test Split",
                          "max_depth","min_samples_leaf","random_state"],
            "Value":     ["Decision Tree Regressor",
                          "temp, humidity, windspeed, season",
                          "count","20% (80/20)","5","3","42"],
        })
        _dark_table(cfg)

    st.markdown("---")

    if st.button("Train Model", type="primary"):
        with st.spinner("Training…"):
            X, y           = train_model.prepare_features_target(df)
            Xtr,Xte,ytr,yte = train_model.split_data(X, y)
            model          = train_model.train_decision_tree(Xtr, ytr)
            metrics        = train_model.evaluate_model(model, Xte, yte)
            train_model.save_model(model)
            st.session_state.update(model=model, metrics=metrics,
                                    X_test=Xte, y_test=yte)
        st.success("Model trained and saved to `model.pkl`.")

    if "metrics" not in st.session_state:
        st.info("Click **Train Model** to begin.")
        return

    m   = st.session_state["metrics"]
    mdl = st.session_state["model"]
    yte = st.session_state["y_test"]

    # Metrics
    st.subheader("Evaluation Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",      f"{m['mae']:.2f}")
    c1.caption("Average absolute error in bike rentals.")
    c2.metric("RMSE",     f"{m['rmse']:.2f}")
    c2.caption("Penalises large errors more than MAE.")
    c3.metric("R² Score", f"{m['r2']:.4f}")
    c3.caption("1.0 = perfect. Variance explained by the model.")

    st.markdown("---")
    ca, cb = st.columns(2)

    # Feature importance
    with ca:
        st.subheader("Feature Importance")
        fi = train_model.get_feature_importance(mdl, train_model.FEATURE_COLUMNS)
        fig = px.bar(fi, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale="Blues",
                     text="importance", title="Decision Tree Feature Importances")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                          textfont=dict(color="#E2E8F0"))
        fig.update_layout(**_layout(
            title=dict(text="Decision Tree Feature Importances",
                       font=dict(color="#FFFFFF", size=15)),
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed", tickfont=dict(color="#E2E8F0", size=12)),
            xaxis=dict(range=[0, fi["importance"].max() * 1.25],
                       tickfont=dict(color="#E2E8F0")),
        ))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Actual vs Predicted
    with cb:
        st.subheader("Actual vs Predicted")
        ypred = m["y_pred"]
        lim   = [min(float(yte.min()), float(ypred.min()))-10,
                 max(float(yte.max()), float(ypred.max()))+10]
        fig2  = go.Figure()
        fig2.add_trace(go.Scatter(x=list(yte), y=list(ypred), mode="markers",
                                  marker=dict(color=C_PRIMARY, size=9, opacity=0.85),
                                  name="Predictions"))
        fig2.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                  line=dict(color=C_RED, dash="dash", width=2),
                                  name="Perfect fit"))
        fig2.update_layout(**_layout(
            title=dict(text="Actual vs Predicted (Test Set)",
                       font=dict(color="#FFFFFF", size=15)),
            xaxis=dict(title="Actual Count", range=lim, tickfont=dict(color="#E2E8F0"),
                       title_font=dict(color="#E2E8F0")),
            yaxis=dict(title="Predicted Count", range=lim, tickfont=dict(color="#E2E8F0"),
                       title_font=dict(color="#E2E8F0")),
            legend=dict(font=dict(color="#E2E8F0")),
        ))
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

    st.caption(f"Note: test set = {len(yte)} rows (20% of 100). "
               "Metrics can vary with different random splits.")


# ============================================================================
# PAGE 5 — PREDICTION
# ============================================================================
def show_prediction() -> None:
    st.title("Prediction")

    if "model" not in st.session_state:
        # Try loading a previously saved model.pkl from disk before giving up
        try:
            st.session_state["model"] = train_model.load_model("model.pkl")
        except FileNotFoundError:
            st.warning("No trained model found. Go to **Model Training** and click **Train Model** first.")
            return

    model = st.session_state["model"]
    st.markdown(f"<p style='color:{C_MUTED};'>Enter conditions to get a demand forecast.</p>",
                unsafe_allow_html=True)

    # Load example button
    if st.button("Load Assignment Example  (temp=0.5, humidity=0.6, windspeed=0.2, season=3)"):
        st.session_state.update(ex_temp=0.5, ex_humidity=0.6,
                                ex_windspeed=0.2, ex_season=3)

    st.markdown("---")

    # ── Input boxes (number_input, not sliders) ──────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        temp = st.number_input(
            "Temperature (normalised  0.0 – 1.0)",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(st.session_state.get("ex_temp", 0.5)),
            format="%.2f",
        )
        humidity = st.number_input(
            "Humidity (normalised  0.0 – 1.0)",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(st.session_state.get("ex_humidity", 0.6)),
            format="%.2f",
        )
    with c2:
        windspeed = st.number_input(
            "Wind Speed (normalised  0.0 – 1.0)",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(st.session_state.get("ex_windspeed", 0.2)),
            format="%.2f",
        )
        season_choice = st.selectbox(
            "Season",
            options=[1, 2, 3, 4],
            format_func=lambda s: f"{s} — {SEASON_LABELS[s]}",
            index=int(st.session_state.get("ex_season", 3)) - 1,
        )

    st.markdown("---")

    if st.button("Predict", type="primary"):
        try:
            prediction    = train_model.predict_single(
                model, temp=temp, humidity=humidity,
                windspeed=windspeed, season=int(season_choice),
            )
            predicted_count = round(prediction)

            if predicted_count < 100:
                badge_color, badge_label, badge_icon = C_RED,   "LOW DEMAND",      "🔴"
            elif predicted_count < 200:
                badge_color, badge_label, badge_icon = C_AMBER, "MODERATE DEMAND", "🟡"
            else:
                badge_color, badge_label, badge_icon = C_GREEN, "HIGH DEMAND",     "🟢"

            st.markdown(f"""
            <div class="hover-card" style="background:{C_CARD};border:1px solid {C_BORDER};
                        border-radius:16px;padding:36px 44px;text-align:center;
                        margin-top:20px;box-shadow:0 8px 32px rgba(0,0,0,0.4);
                        animation:fadeUp 0.5s ease both;">
                <div style="color:{C_MUTED};font-size:0.8rem;letter-spacing:.12em;
                            font-weight:600;margin-bottom:10px;">PREDICTED BIKE RENTALS</div>
                <div style="color:{C_TEXT};font-size:4rem;font-weight:800;
                            line-height:1.1;margin-bottom:16px;">{predicted_count}</div>
                <span style="background:{badge_color};color:#fff;border-radius:20px;
                             padding:5px 20px;font-size:0.85rem;font-weight:700;
                             letter-spacing:.06em;">{badge_icon} {badge_label}</span>
                <div style="color:{C_MUTED};font-size:0.92rem;margin-top:20px;
                            line-height:1.7;max-width:520px;margin-left:auto;margin-right:auto;">
                    Temp <strong style="color:{C_TEXT};">{temp}</strong> ·
                    Humidity <strong style="color:{C_TEXT};">{humidity}</strong> ·
                    Wind <strong style="color:{C_TEXT};">{windspeed}</strong> ·
                    <strong style="color:{C_TEXT};">{SEASON_LABELS[int(season_choice)]}</strong>
                    → approximately <strong style="color:{C_PRIMARY};">{predicted_count}</strong> rentals.
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(" ")

            # Download CSV
            result_df = pd.DataFrame([{
                "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "temp":            temp,
                "humidity":        humidity,
                "windspeed":       windspeed,
                "season":          int(season_choice),
                "season_label":    SEASON_LABELS[int(season_choice)],
                "predicted_count": predicted_count,
            }])
            st.download_button(
                "Download Prediction as CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name=f"bike_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        except ValueError as e:
            st.error(f"Validation error: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ============================================================================
# PAGE 6 — ABOUT
# ============================================================================
def show_about() -> None:
    st.title("About")

    # Author card with real clickable icons
    st.markdown(f"""
    <div class="hover-card" style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:14px;
                padding:32px 36px;margin-bottom:28px;">
        <div style="font-size:1.4rem;font-weight:800;color:{C_TEXT};margin-bottom:4px;">
            Gul Shair
        </div>
        <div style="color:{C_MUTED};font-size:0.9rem;margin-bottom:18px;">
            ML &amp; Python Developer · Portfolio Project
        </div>
        <div style="display:flex;gap:14px;flex-wrap:wrap;">
            <a href="https://www.linkedin.com/in/gulshair/" target="_blank"
               style="display:inline-flex;align-items:center;gap:8px;
                      background:#0A66C2;color:#fff;text-decoration:none;
                      padding:9px 20px;border-radius:8px;font-weight:600;font-size:0.88rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                     fill="currentColor" viewBox="0 0 24 24">
                  <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761
                  2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11
                  19h-3v-10h3v10zm-1.5-11.268c-.966 0-1.75-.784-1.75-1.75s.784-1.75
                  1.75-1.75 1.75.784 1.75 1.75-.784 1.75-1.75 1.75zm13.5
                  11.268h-3v-5.604c0-1.337-.027-3.058-1.863-3.058-1.863 0-2.148
                  1.455-2.148 2.959v5.703h-3v-10h2.881v1.367h.041c.401-.761
                  1.381-1.563 2.844-1.563 3.042 0 3.604 2.002 3.604 4.604v5.592z"/>
                </svg>
                LinkedIn
            </a>
            <a href="https://github.com/gulshairkhaqan-hub" target="_blank"
               style="display:inline-flex;align-items:center;gap:8px;
                      background:#24292F;color:#fff;text-decoration:none;
                      padding:9px 20px;border-radius:8px;font-weight:600;font-size:0.88rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                     fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438
                  9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726
                  -4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089
                  -.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07
                  1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665
                  -.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124
                  -.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266
                  1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552
                  3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235
                  1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823
                  1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589
                  8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                GitHub
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hover-card" style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:14px;
                padding:28px 36px;margin-bottom:20px;">
        <h3 style="color:{C_PRIMARY};margin-top:0;">Project Overview</h3>
        <p style="color:{C_TEXT};line-height:1.7;">
            End-to-end ML workflow predicting bike rental demand from weather and
            seasonal data. Built as a portfolio assignment covering Decision Tree
            Regression, interactive EDA, and a live prediction dashboard.
        </p>
        <h3 style="color:{C_PRIMARY};">Tech Stack</h3>
        <table style="width:100%;border-collapse:collapse;color:{C_TEXT};font-size:0.9rem;">
            <tr style="border-bottom:1px solid {C_BORDER};">
                <td style="padding:8px 12px;font-weight:600;color:{C_PRIMARY};">Python</td>
                <td style="padding:8px 12px;">Core language</td>
            </tr>
            <tr style="border-bottom:1px solid {C_BORDER};">
                <td style="padding:8px 12px;font-weight:600;color:{C_PRIMARY};">Streamlit</td>
                <td style="padding:8px 12px;">Interactive web dashboard</td>
            </tr>
            <tr style="border-bottom:1px solid {C_BORDER};">
                <td style="padding:8px 12px;font-weight:600;color:{C_PRIMARY};">scikit-learn</td>
                <td style="padding:8px 12px;">DecisionTreeRegressor, metrics, train/test split</td>
            </tr>
            <tr style="border-bottom:1px solid {C_BORDER};">
                <td style="padding:8px 12px;font-weight:600;color:{C_PRIMARY};">Plotly</td>
                <td style="padding:8px 12px;">All interactive charts</td>
            </tr>
            <tr>
                <td style="padding:8px 12px;font-weight:600;color:{C_PRIMARY};">pandas / NumPy / joblib</td>
                <td style="padding:8px 12px;">Data handling and model persistence</td>
            </tr>
        </table>
        <h3 style="color:{C_PRIMARY};margin-top:22px;">Model</h3>
        <p style="color:{C_TEXT};line-height:1.7;">
            Decision Tree Regressor with <code>max_depth=5</code> and
            <code>min_samples_leaf=3</code> to prevent overfitting on the
            100-row dataset. R² = 0.853 on the held-out test set.
        </p>
        <h3 style="color:{C_PRIMARY};">Limitations</h3>
        <p style="color:{C_AMBER};line-height:1.7;">
            Trained on 100 records only. Predictions are indicative, not
            production-grade. A larger dataset is needed for operational use.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN — sidebar + routing
# ============================================================================
def main() -> None:
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 0 8px;">
            <div style="font-size:2rem;">📊</div>
            <div style="font-weight:800;font-size:1.1rem;color:{C_TEXT};
                        line-height:1.3;margin-top:6px;">
                Bike Rental<br>Demand Prediction
            </div>
            <div style="font-size:0.72rem;color:{C_MUTED};margin-top:4px;">
                Decision Tree Regression
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=["Home","Dataset","Visualizations","Model Training","Prediction","About"],
            label_visibility="hidden",
        )
        st.markdown("---")

        # Model status badge
        if "model" in st.session_state:
            st.markdown(
                f"<div style='background:{C_GREEN};border-radius:8px;padding:8px 12px;"
                f"text-align:center;font-size:0.78rem;color:#fff;font-weight:700;'>"
                "✅ Model Trained</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background:#78350F;border-radius:8px;padding:8px 12px;"
                f"text-align:center;font-size:0.78rem;color:#FDE68A;font-weight:700;'>"
                "⚠️ Model Not Trained</div>", unsafe_allow_html=True)

    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    if   page == "Home":           show_home(df)
    elif page == "Dataset":        show_dataset(df)
    elif page == "Visualizations": show_visualizations(df)
    elif page == "Model Training": show_model_training(df)
    elif page == "Prediction":     show_prediction()
    elif page == "About":          show_about()


if __name__ == "__main__":
    main()


