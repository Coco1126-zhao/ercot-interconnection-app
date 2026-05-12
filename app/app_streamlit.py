"""
app/app_streamlit.py
====================
ERCOT Interconnection Predictor — interface.

Design language:
  - ERCOT brand palette: teal #00B0AC, slate grey, white.
  - Inter for UI; JetBrains Mono for data values.
  - Light, data-dense, clean — inspired by GridStatus, Modo Energy, Aurora.

Layout:
  LEFT  (5/13)  ERCOT live queue: map · quick stats · scrollable feed
  RIGHT (8/13)  Main function: project input form  →  Layer 1 + Layer 2 outputs
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ercot.loader     import load_gis_large_gen, MILESTONE_COLS_LARGE
from src.ercot.geo_utils  import attach_coordinates
from src.ercot.similarity import SimilarityEngine
from src.ercot.predictor  import Predictor
from src.ercot.narratives import explain_driver

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="ERCOT Interconnection Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme: ERCOT brand colours · clean · monospace data ─────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --teal:        #00B0AC;
  --teal-700:    #008884;
  --teal-50:     #E6F8F7;
  --slate-900:   #0F172A;
  --slate-700:   #334155;
  --slate-500:   #64748B;
  --slate-300:   #CBD5E1;
  --slate-200:   #E2E8F0;
  --slate-100:   #F1F5F9;
  --slate-50:    #F8FAFC;
  --white:       #FFFFFF;
  --amber:       #D97706;
  --rose:        #BE123C;
}

html, body, [class*="css"], .stApp, .stMarkdown, .stText {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--slate-900);
}
.stApp { background: var(--white); }

/* Streamlit chrome */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 8px; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 1.2rem !important; max-width: 1500px; }

/* Brand bar */
.brand-bar {
  display: flex; align-items: flex-end; gap: 14px;
  padding: 4px 0 18px;
  border-bottom: 1px solid var(--slate-200);
  margin-bottom: 22px;
}
.brand-title-block {
  display: flex; flex-direction: column; gap: 4px;
  border-left: 4px solid var(--teal);
  padding-left: 14px;
}
.brand-title {
  font-size: 44px; font-weight: 700;
  letter-spacing: -0.035em; line-height: 1;
  color: var(--slate-900);
  text-transform: uppercase;
}
.brand-sub-title {
  font-size: 17px; font-weight: 500;
  letter-spacing: -0.005em; line-height: 1.2;
  color: var(--slate-500);
}
.brand-status {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: 0.04em;
  color: var(--slate-500); text-transform: uppercase;
  white-space: nowrap;
  padding-bottom: 4px;
}
.brand-dot {
  width: 6px; height: 6px; border-radius: 999px;
  background: var(--teal); display: inline-block; margin-right: 6px;
}

/* Headings */
h1, h2, h3, h4 {
  color: var(--slate-900) !important;
  font-weight: 600; letter-spacing: -0.015em;
}

/* Section caption (uppercase mono micro-label) */
.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; font-weight: 600;
  letter-spacing: 0.10em; text-transform: uppercase;
  color: var(--slate-500);
  margin: 0 0 6px;
}

/* Cards */
.card {
  background: var(--white);
  border: 1px solid var(--slate-200);
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.card-tight { padding: 10px 12px; margin-bottom: 8px; }

/* Quick stats grid */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat {
  border: 1px solid var(--slate-200); border-radius: 8px;
  padding: 10px 12px; background: var(--slate-50);
}
.stat-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 0.08em;
  color: var(--slate-500); text-transform: uppercase;
}
.stat-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 22px; font-weight: 600; color: var(--slate-900);
  line-height: 1.1; margin-top: 4px;
}
.stat-sub { font-size: 11px; color: var(--slate-500); margin-top: 2px; }

/* Headline metric (used in prediction output) */
.metric {
  background: var(--white);
  border: 1px solid var(--slate-200); border-radius: 10px;
  padding: 14px 16px;
  min-height: 110px;
  display: flex; flex-direction: column; justify-content: space-between;
}
.metric .label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--slate-500);
  line-height: 1.3;
}
.metric .value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 34px; font-weight: 600;
  color: var(--teal-700); line-height: 1.05; margin-top: 6px;
}
.metric .unit { font-size: 14px; color: var(--slate-500); margin-left: 4px; }

/* Queue feed list */
.feed { max-height: 1400px; overflow-y: auto; padding-right: 4px; }
.feed::-webkit-scrollbar { width: 6px; }
.feed::-webkit-scrollbar-thumb { background: var(--slate-300); border-radius: 3px; }
.feed::-webkit-scrollbar-track { background: transparent; }
.feed-row {
  display: grid; grid-template-columns: 1fr auto;
  align-items: start;
  padding: 8px 4px;
  border-bottom: 1px solid var(--slate-100);
}
.feed-row:hover { background: var(--slate-50); }
.feed-name { font-size: 13px; font-weight: 500; color: var(--slate-900); }
.feed-meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--slate-500); margin-top: 2px;
}
.feed-meta .inr { color: var(--teal-700); font-weight: 500; }
.feed-stage {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; color: var(--slate-500);
  text-align: right; white-space: nowrap;
}
.feed-pct {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px; font-weight: 600;
  color: var(--teal-700); text-align: right;
}
.feed-pct-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 8.5px; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--slate-500); text-align: right;
}

/* Stage progress bar */
.bar-track {
  width: 90px; height: 4px; background: var(--slate-200); border-radius: 2px;
  margin-top: 4px; margin-left: auto;
}
.bar-fill { height: 100%; background: var(--teal); border-radius: 2px; }

/* Buttons */
.stButton > button, button[kind="primary"], button[kind="secondary"] {
  background: var(--teal) !important;
  color: var(--white) !important;
  border: 1px solid var(--teal) !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 8px 18px !important;
  transition: background 0.15s !important;
}
.stButton > button:hover { background: var(--teal-700) !important; border-color: var(--teal-700) !important; }
.stButton > button:focus { box-shadow: 0 0 0 3px var(--teal-50) !important; }

/* Form inputs */
input, .stSelectbox > div > div, .stMultiSelect > div > div,
.stNumberInput input, .stTextInput input, .stDateInput input {
  font-family: 'Inter', sans-serif !important;
  border-radius: 6px !important;
  border-color: var(--slate-200) !important;
}
label { font-size: 12px !important; color: var(--slate-700) !important; font-weight: 500 !important; }

/* SHAP factor rows */
.factor {
  display: grid;
  grid-template-columns: 14px 1fr auto;
  align-items: center; gap: 8px;
  padding: 4px 0;
  font-size: 12.5px;
}
.factor .name { color: var(--slate-700); font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.factor .val  { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--slate-500); }
.dot-pos { width: 8px; height: 8px; border-radius: 999px; background: var(--teal); display: inline-block; }
.dot-neg { width: 8px; height: 8px; border-radius: 999px; background: var(--rose); display: inline-block; }
.dot-amb { width: 8px; height: 8px; border-radius: 999px; background: var(--amber); display: inline-block; }

/* Mini milestone strip (Layer 2) */
.timeline { display: flex; gap: 3px; margin-top: 6px; }
.tl-cell {
  width: 18px; height: 6px; border-radius: 1px;
  background: var(--slate-200);
}
.tl-cell.done { background: var(--teal); }

/* Helper notes */
.note {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--slate-500);
  background: var(--slate-50);
  border-left: 2px solid var(--teal);
  padding: 8px 10px; border-radius: 0 6px 6px 0;
}
.placeholder {
  border: 1px dashed var(--slate-300); border-radius: 10px;
  padding: 32px 20px; text-align: center;
  color: var(--slate-500); font-size: 13px;
  background: var(--slate-50);
}

hr { border-color: var(--slate-200); margin: 16px 0; }
.streamlit-expanderHeader { font-size: 12px !important; color: var(--slate-700) !important; }

/* Top navigation tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  border-bottom: 1px solid var(--slate-200);
  margin-bottom: 14px;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  color: var(--slate-500) !important;
  padding: 8px 16px !important;
  border-radius: 6px 6px 0 0 !important;
  border: none !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: var(--teal-700) !important;
  border-bottom: 2px solid var(--teal) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }

/* Insight cards (driver narratives) */
.insight {
  border: 1px solid var(--slate-200);
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 8px;
  background: var(--white);
}
.insight-pos { border-left: 3px solid var(--teal); }
.insight-neg { border-left: 3px solid var(--rose); }
.insight-head {
  display: flex; justify-content: space-between;
  align-items: baseline; margin-bottom: 6px; gap: 8px;
}
.insight-title { font-size: 13px; font-weight: 600; color: var(--slate-900); }
.insight-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--slate-500); white-space: nowrap;
}
.insight-body {
  font-size: 12.5px; color: var(--slate-700); line-height: 1.5;
}
.insight-suggest {
  margin-top: 8px; padding: 7px 10px;
  background: var(--teal-50); border-radius: 6px;
  font-size: 12px; color: var(--teal-700); line-height: 1.4;
}
.insight-suggest::before {
  content: "→ "; font-weight: 700; color: var(--teal-700);
}
.context-tag {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px; letter-spacing: 0.05em;
  color: var(--slate-500); background: var(--slate-100);
  padding: 1px 6px; border-radius: 3px; text-transform: uppercase;
  margin-left: 6px;
}

/* Methodology / Data page styling */
.doc-section { margin: 0 0 28px; }
.doc-section h2 {
  font-size: 22px !important; margin: 0 0 4px !important;
  color: var(--slate-900) !important;
}
.doc-section .doc-sub {
  font-size: 13px; color: var(--slate-500);
  margin-bottom: 14px;
}
.doc-section h3 {
  font-size: 16px !important; margin: 18px 0 6px !important;
  color: var(--slate-900) !important;
}
.doc-section p, .doc-section li {
  font-size: 14px; color: var(--slate-700);
  line-height: 1.65;
}
.doc-section code {
  background: var(--slate-100); color: var(--slate-900);
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  padding: 1px 5px; border-radius: 3px;
}
.doc-table {
  width: 100%; border-collapse: collapse; margin: 8px 0 16px;
  font-size: 13px;
}
.doc-table th, .doc-table td {
  border-bottom: 1px solid var(--slate-200);
  text-align: left; padding: 8px 10px;
  color: var(--slate-700);
}
.doc-table th {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--slate-500);
  font-weight: 600;
}
.doc-callout {
  background: var(--teal-50);
  border-left: 3px solid var(--teal);
  border-radius: 0 6px 6px 0;
  padding: 10px 14px; margin: 12px 0;
  font-size: 13px; color: var(--teal-700); line-height: 1.5;
}

/* Output frame — light grey card wrapping the prediction results */
[data-testid="stVerticalBlock"]:has(> div > [data-testid="stMarkdownContainer"] > .output-frame-marker) {
  background: var(--slate-50);
  border: 1px solid var(--slate-200);
  border-radius: 12px;
  padding: 18px 20px;
  margin-top: 14px;
}
.output-frame-marker { display: none; }

/* Layer subtitle (sits under each layer header) */
.layer-subtitle {
  font-size: 12px; color: var(--slate-500);
  line-height: 1.5; margin: -2px 0 10px;
}

/* Match card (Project Clustering) */
.match {
  background: var(--white);
  border: 1px solid var(--slate-200);
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 8px;
  position: relative;
}
.match-rank {
  position: absolute; top: 10px; right: 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 600;
  color: var(--slate-500);
  background: var(--slate-100);
  padding: 2px 6px; border-radius: 4px;
}
.match-name {
  font-size: 13px; font-weight: 600; color: var(--slate-900);
  margin-right: 40px;
}
.match-meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--slate-500); margin-top: 2px;
}
.match-meta .inr { color: var(--teal-700); font-weight: 500; }

.similarity-row {
  display: grid; grid-template-columns: 90px 1fr auto;
  gap: 8px; align-items: center;
  margin-top: 6px;
}
.similarity-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; color: var(--slate-500);
  text-transform: uppercase; letter-spacing: 0.06em;
}
.sim-track {
  height: 5px; background: var(--slate-200); border-radius: 3px;
}
.sim-fill { height: 100%; background: var(--teal); border-radius: 3px; }
.sim-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--teal-700); font-weight: 600;
}

.stage-row {
  display: grid; grid-template-columns: 90px 1fr auto;
  gap: 8px; align-items: center;
  margin-top: 4px;
}
.stage-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; color: var(--slate-500);
  text-transform: uppercase; letter-spacing: 0.06em;
}
.stage-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--slate-700);
}
.stage-pct {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--teal-700); font-weight: 600;
}

/* Performance table for methodology */
.perf-table {
  width: 100%; border-collapse: collapse;
  font-size: 12.5px; margin: 8px 0 16px;
}
.perf-table th, .perf-table td {
  border-bottom: 1px solid var(--slate-200);
  padding: 8px 10px; text-align: left;
}
.perf-table th {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--slate-500);
  font-weight: 600; background: var(--slate-50);
}
.perf-table td.num {
  font-family: 'JetBrains Mono', monospace;
  color: var(--slate-900); font-size: 13px;
}
.perf-table tr.highlight td.num { color: var(--teal-700); font-weight: 600; }

.method-figure-caption {
  font-size: 11.5px; color: var(--slate-500);
  text-align: center; margin: -8px 0 18px;
  font-style: italic;
}

/* Methodology figure explanation block (sits in right column) */
.figure-block {
  padding: 0 4px;
}
.figure-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--teal-700);
  font-weight: 600;
  margin: 0 0 8px;
}
.figure-body {
  font-size: 13px;
  color: var(--slate-700);
  line-height: 1.6;
  margin: 0;
}
.figure-body b { color: var(--slate-900); }
.figure-takeaway {
  margin-top: 10px;
  background: var(--teal-50);
  border-left: 3px solid var(--teal);
  border-radius: 0 6px 6px 0;
  padding: 8px 12px;
  font-size: 12px;
  color: var(--teal-700);
  line-height: 1.5;
}
.figure-takeaway b { color: var(--teal-700); }

/* Subtle divider between figure rows */
.figure-divider {
  border-top: 1px solid var(--slate-200);
  margin: 22px 0;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Cached loaders ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading ERCOT GIS report…")
def load_ercot():
    df = load_gis_large_gen("data/ercot/latest.xlsx")
    df = attach_coordinates(df)
    return df


@st.cache_resource(show_spinner="Loading trained models…")
def load_predictor():
    try:
        return Predictor()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def get_similarity_engine(_n_rows: int):
    return SimilarityEngine(load_ercot())


ercot_df  = load_ercot()
predictor = load_predictor()
n_proj    = len(ercot_df)


# ── Brand header ────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="brand-bar">
      <div class="brand-title-block">
        <div class="brand-title">ERCOT</div>
        <div class="brand-sub-title">Interconnection Probability &amp; Duration Predictor</div>
      </div>
      <div class="brand-status">
        <span class="brand-dot"></span>data current · LBNL 2024 + ERCOT GIS
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Stage progression palette (monochromatic teal, light → dark) ───
STAGE_PROGRESSION = [
    "Pre-Screening",
    "Screening Study Started",
    "Screening Study Complete",
    "FIS Requested",
    "FIS Approved",
    "IA Signed",
    "Construction Start",
    "Construction End",
    "Approved for Energization",
    "Approved for Synchronization",
]
STAGE_COLOR = {
    "Pre-Screening":                "#E2E8F0",
    "Screening Study Started":      "#A7E8E5",
    "Screening Study Complete":     "#7CDCD9",
    "FIS Requested":                "#5ECDC9",
    "FIS Approved":                 "#3FBEB9",
    "IA Signed":                    "#22A8A4",
    "Construction Start":           "#179490",
    "Construction End":             "#0F8682",
    "Approved for Energization":    "#007D79",
    "Approved for Synchronization": "#005C59",
}


# ── Helpers ─────────────────────────────────────────────────────────
def fmt_mw(x):
    return "n/a" if pd.isna(x) else f"{x:,.0f}"


def fmt_int(x):
    return "n/a" if pd.isna(x) else f"{int(x):,}"


def milestone_strip(row) -> str:
    cells = []
    for col in MILESTONE_COLS_LARGE:
        done = bool(row.get(f"_done_{col}", False))
        cls = "tl-cell done" if done else "tl-cell"
        cells.append(f"<span class='{cls}' title='{col}'></span>")
    return f"<div class='timeline'>{''.join(cells)}</div>"


# ── Tab navigation ─────────────────────────────────────────────────
tab_predict, tab_method, tab_data = st.tabs(["Predictor", "Methodology", "Data"])


# ═══════════════════════════════════════════════════════════════════
# Tab 1 · Predictor (original layout)
# ═══════════════════════════════════════════════════════════════════
with tab_predict:
    LEFT, RIGHT = st.columns([5, 8], gap="large")

    # ── LEFT: live ERCOT queue ─────────────────────────────────────
    with LEFT:
        st.markdown('<div class="section-label">ERCOT live queue · map</div>',
                    unsafe_allow_html=True)

        plot_df = ercot_df.dropna(subset=["lat", "lon"]).copy()
        fmap = folium.Map(
            location=[31.0, -99.5], zoom_start=5,
            tiles="CartoDB positron",
            control_scale=False,
        )
        for _, r in plot_df.iterrows():
            mw_raw = r["capacity_mw"]
            mw = 0.0 if pd.isna(mw_raw) else abs(float(mw_raw))
            radius = max(1.5, min(5.0, mw ** 0.3))
            c = STAGE_COLOR.get(r["current_stage"], "#CBD5E1")
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=radius, color=c, fill=True, fill_color=c,
                fill_opacity=0.65, weight=0,
                tooltip=(
                    f"<b>{r['Project Name']}</b><br>"
                    f"<span style='font-family:monospace'>{r['INR']}</span><br>"
                    f"{r['fuel_label']} · {fmt_mw(mw_raw)} MW<br>"
                    f"{r['current_stage']} · {r['progress_pct']:.0f}%"
                ),
            ).add_to(fmap)
        st_folium(fmap, height=320, use_container_width=True, returned_objects=[])

        st.markdown('<div class="section-label">queue overview</div>',
                    unsafe_allow_html=True)
        tot_mw   = ercot_df["capacity_mw"].clip(lower=0).sum()
        top_fuel = ercot_df["fuel_label"].value_counts().idxmax()
        top_zone = ercot_df["CDR Reporting Zone"].value_counts().idxmax()
        op_count = (ercot_df["current_stage"] == "Approved for Synchronization").sum()

        st.markdown(
            f"""
            <div class="stat-grid">
              <div class="stat">
                <div class="stat-label">active projects</div>
                <div class="stat-value">{n_proj:,}</div>
                <div class="stat-sub">large-gen sheet</div>
              </div>
              <div class="stat">
                <div class="stat-label">total MW (gross)</div>
                <div class="stat-value">{tot_mw/1000:,.1f}<span class="unit"> GW</span></div>
                <div class="stat-sub">queued + commissioning</div>
              </div>
              <div class="stat">
                <div class="stat-label">approved · sync</div>
                <div class="stat-value">{op_count:,}</div>
                <div class="stat-sub">final ERCOT milestone</div>
              </div>
              <div class="stat">
                <div class="stat-label">top fuel · zone</div>
                <div class="stat-value" style="font-size:16px">{top_fuel}<br>{top_zone}</div>
                <div class="stat-sub">most-common labels</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label" style="margin-top:14px">queue feed</div>',
                    unsafe_allow_html=True)
        ctl1, ctl2 = st.columns([1, 1])
        sort_by = ctl1.selectbox("Sort by",
                                 ["progress ↓", "capacity ↓", "months in queue ↓", "INR"],
                                 label_visibility="collapsed", index=0)
        fuels = sorted([f for f in ercot_df["fuel_label"].dropna().unique() if f])
        fuel_filter = ctl2.multiselect("Filter fuel", fuels,
                                       placeholder="filter fuel…",
                                       label_visibility="collapsed")

        key, asc = {
            "progress ↓":         ("progress_pct",   False),
            "capacity ↓":         ("capacity_mw",    False),
            "months in queue ↓":  ("months_in_queue",False),
            "INR":                ("INR",            True),
        }[sort_by]
        feed = ercot_df.sort_values(key, ascending=asc)
        if fuel_filter:
            feed = feed[feed["fuel_label"].isin(fuel_filter)]

        feed_html = '<div class="feed">'
        for _, r in feed.head(120).iterrows():
            mw_str  = fmt_mw(r["capacity_mw"])
            miq     = r["months_in_queue"]
            miq_str = "n/a" if pd.isna(miq) else f"{miq:.0f}mo"
            pct     = r["progress_pct"]
            feed_html += (
                f"<div class='feed-row'>"
                f"  <div>"
                f"    <div class='feed-name'>{r['Project Name']}</div>"
                f"    <div class='feed-meta'>"
                f"      <span class='inr'>{r['INR']}</span> · "
                f"      {r['fuel_label']} · {mw_str} MW · {r['County']} · {r['CDR Reporting Zone']}"
                f"    </div>"
                f"  </div>"
                f"  <div>"
                f"    <div class='feed-pct-label'>stage progress</div>"
                f"    <div class='feed-pct'>{pct:.0f}%</div>"
                f"    <div class='feed-stage'>{r['current_stage']} · {miq_str}</div>"
                f"    <div class='bar-track'><div class='bar-fill' style='width:{pct:.0f}%'></div></div>"
                f"  </div>"
                f"</div>"
            )
        feed_html += "</div>"
        st.markdown(feed_html, unsafe_allow_html=True)

    # ── RIGHT: prediction (main function) ──────────────────────────
    with RIGHT:
        st.markdown('<div class="section-label">new project · predict viability</div>',
                    unsafe_allow_html=True)

        with st.form("project_form", clear_on_submit=False, border=False):
            f1, f2, f3 = st.columns(3)
            cap = f1.number_input("Capacity (MW)", min_value=1.0, value=200.0, step=10.0)
            fuel = f2.selectbox("Fuel / technology",
                                ["Solar", "Wind", "Battery", "Gas", "Hydro",
                                 "Biomass", "Coal", "Oil", "Nuclear", "Other"], index=2)
            cap_factor = f3.slider("Capacity factor", 0.05, 0.95, 0.30, 0.05,
                                   help="Used by Layer 2 similarity only — not in LBNL model.")

            g1, g2, g3 = st.columns(3)
            counties = sorted([c for c in ercot_df["County"].dropna().unique() if c])
            county = g1.selectbox("County", counties,
                                  index=counties.index("Crockett") if "Crockett" in counties else 0)
            zones = sorted([z for z in ercot_df["CDR Reporting Zone"].dropna().unique() if z])
            zone = g2.selectbox("CDR zone", zones,
                                index=zones.index("WEST") if "WEST" in zones else 0)
            poi = g3.text_input("POI / substation", value="", placeholder="optional")

            h1, h2, h3 = st.columns(3)
            cod = h1.date_input("Projected COD",
                                value=pd.Timestamp.today().normalize() + pd.Timedelta(days=730))
            service = h2.selectbox("Service request", ["NRIS+ERIS", "NRIS", "ERIS"],
                                   index=0,
                                   help="LBNL feature; ERCOT does not use NRIS/ERIS — defaulted.")
            is_hybrid = h3.checkbox("Hybrid project", value=False,
                                    help="e.g., Solar + Battery")

            submitted = st.form_submit_button("⚡  Get prediction & suggestions",
                                              use_container_width=False)

        if not submitted:
            st.markdown(
                '<div class="placeholder">Submit a project above to see '
                '<b>Interconnection Prediction</b> (probability + expected duration with drivers and suggestions) '
                'and <b>Project Clustering</b> (5 most-similar live ERCOT projects with their actual progress).</div>',
                unsafe_allow_html=True,
            )
        else:
            user_input = {
                "capacity_mw":     float(cap),
                "fuel_label":      fuel,
                "is_hybrid":       int(is_hybrid),
                "county":          county,
                "cdr_zone":        zone,
                "poi":             poi,
                "projected_cod":   str(cod),
                "service_type":    service,
                "capacity_factor": cap_factor,
            }

            # Light-grey output frame (CSS targets the container with this marker)
            results = st.container()
            with results:
                st.markdown('<div class="output-frame-marker"></div>',
                            unsafe_allow_html=True)

                L1col, L2col = st.columns([1, 1], gap="medium")

                # ═══════ Interconnection Prediction (Probability & Duration) ═══════
                with L1col:
                    st.markdown(
                        '<div class="section-label">interconnection prediction · probability &amp; duration</div>'
                        '<div class="layer-subtitle">Trained on 24,690 historical projects (LBNL Queued Up, 1970–2024). '
                        'For general reference — the regressor beats an ISO-mean baseline by '
                        '<b style="color:var(--teal-700)">35%</b> (RMSE 14.3 vs 21.9 months).</div>',
                        unsafe_allow_html=True,
                    )

                    if predictor is None:
                        st.warning(
                            "Model artifacts not found. Run "
                            "`python -m src.ercot.train_and_save` from the project root."
                        )
                    else:
                        res = predictor.predict(user_input, current_queue_size=n_proj)

                        m1, m2 = st.columns(2)
                        m1.markdown(
                            f"""<div class="metric">
                                <div class="label">completion probability</div>
                                <div class="value">{res['prob_complete']*100:.1f}<span class="unit">%</span></div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        m2.markdown(
                            f"""<div class="metric">
                                <div class="label">expected duration</div>
                                <div class="value">{res['duration_months']:.0f}<span class="unit">months</span></div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                        # Compact completion-driver list -------------
                        st.markdown(
                            '<div class="section-label" style="margin-top:14px">'
                            'drivers · completion probability</div>',
                            unsafe_allow_html=True,
                        )
                        rows = []
                        for n, v in res["clf_top_positive"][:4]:
                            rows.append(f"<div class='factor'><span class='dot-pos'></span>"
                                        f"<span class='name'>{n}</span><span class='val'>+{v:.3f}</span></div>")
                        for n, v in res["clf_top_negative"][:4]:
                            rows.append(f"<div class='factor'><span class='dot-neg'></span>"
                                        f"<span class='name'>{n}</span><span class='val'>{v:+.3f}</span></div>")
                        st.markdown("".join(rows), unsafe_allow_html=True)

                        # Insights & suggestions for completion -------
                        st.markdown(
                            '<div class="section-label" style="margin-top:16px">'
                            'insights · completion probability</div>',
                            unsafe_allow_html=True,
                        )
                        insight_html = ""
                        seen = set()
                        pos_d = res["clf_top_positive"][:4]
                        neg_d = res["clf_top_negative"][:4]
                        interleaved = []
                        for i in range(max(len(pos_d), len(neg_d))):
                            if i < len(pos_d): interleaved.append(pos_d[i])
                            if i < len(neg_d): interleaved.append(neg_d[i])
                        for name, val in interleaved:
                            ins = explain_driver(name, val, context="completion")
                            if ins["label"] in seen:
                                continue
                            seen.add(ins["label"])
                            side = "insight-pos" if ins["direction"] == "positive" else "insight-neg"
                            tag = "" if ins["actionable"] else \
                                  '<span class="context-tag">context · no action</span>'
                            sug = (f"<div class='insight-suggest'>{ins['suggestion']}</div>"
                                   if ins.get("suggestion") else "")
                            insight_html += (
                                f"<div class='insight {side}'>"
                                f"<div class='insight-head'>"
                                f"<div class='insight-title'>{ins['label']}{tag}</div>"
                                f"<div class='insight-val'>SHAP {val:+.3f}</div>"
                                f"</div>"
                                f"<div class='insight-body'>{ins['narrative']}</div>"
                                f"{sug}</div>"
                            )
                        st.markdown(insight_html, unsafe_allow_html=True)

                        # Compact duration-driver list ---------------
                        st.markdown(
                            '<div class="section-label" style="margin-top:14px">'
                            'drivers · duration  '
                            '<span style="color:var(--teal-700)">teal = shortens</span> · '
                            '<span style="color:var(--rose)">rose = lengthens</span></div>',
                            unsafe_allow_html=True,
                        )
                        rows = []
                        for n, v in res["reg_top_speedup"][:3]:
                            rows.append(f"<div class='factor'><span class='dot-pos'></span>"
                                        f"<span class='name'>{n}</span><span class='val'>{v:+.2f} mo</span></div>")
                        for n, v in res["reg_top_slowdown"][:3]:
                            rows.append(f"<div class='factor'><span class='dot-neg'></span>"
                                        f"<span class='name'>{n}</span><span class='val'>{v:+.2f} mo</span></div>")
                        st.markdown("".join(rows), unsafe_allow_html=True)

                        # Insights & suggestions for DURATION (NEW) ---
                        st.markdown(
                            '<div class="section-label" style="margin-top:16px">'
                            'insights · duration</div>',
                            unsafe_allow_html=True,
                        )
                        insight_html = ""
                        seen = set()
                        # reg_top_slowdown = positive SHAP (lengthens, BAD)
                        # reg_top_speedup  = negative SHAP (shortens, GOOD)
                        slow = res["reg_top_slowdown"][:3]
                        fast = res["reg_top_speedup"][:3]
                        interleaved = []
                        for i in range(max(len(slow), len(fast))):
                            if i < len(slow): interleaved.append(slow[i])
                            if i < len(fast): interleaved.append(fast[i])
                        for name, val in interleaved:
                            ins = explain_driver(name, val, context="duration")
                            if ins["label"] in seen:
                                continue
                            seen.add(ins["label"])
                            # In duration context: is_bad = positive SHAP → rose
                            side = "insight-neg" if ins["is_bad"] else "insight-pos"
                            tag = "" if ins["actionable"] else \
                                  '<span class="context-tag">context · no action</span>'
                            sug = (f"<div class='insight-suggest'>{ins['suggestion']}</div>"
                                   if ins.get("suggestion") else "")
                            insight_html += (
                                f"<div class='insight {side}'>"
                                f"<div class='insight-head'>"
                                f"<div class='insight-title'>{ins['label']}{tag}</div>"
                                f"<div class='insight-val'>SHAP {val:+.2f} mo</div>"
                                f"</div>"
                                f"<div class='insight-body'>{ins['narrative']}</div>"
                                f"{sug}</div>"
                            )
                        st.markdown(insight_html, unsafe_allow_html=True)

                # ═══════ Project Clustering ═══════
                with L2col:
                    st.markdown(
                        '<div class="section-label">project clustering · live ERCOT queue</div>'
                        '<div class="layer-subtitle">k-Nearest-Neighbours over the current '
                        f'{n_proj:,} active ERCOT projects, weighted by fuel · zone · county · '
                        'capacity · vintage · planned lead time. Shows the 5 closest live peers '
                        'and their actual milestone progress as a reality check on the historical prediction.</div>',
                        unsafe_allow_html=True,
                    )
                    eng = get_similarity_engine(n_proj)
                    sim = eng.find_similar(user_input, n=5)

                    # ── Mini map of matches + user ─────────────────
                    from src.ercot.geo_utils import _ensure_county_centroids, ZONE_CENTROIDS
                    counties_df = _ensure_county_centroids()
                    cmap = counties_df.set_index("county_norm")[["lat", "lon"]].to_dict("index")
                    user_county = user_input["county"]
                    user_lat, user_lon = None, None
                    if user_county in cmap:
                        user_lat = cmap[user_county]["lat"]
                        user_lon = cmap[user_county]["lon"]
                    elif user_input["cdr_zone"] in ZONE_CENTROIDS:
                        user_lat, user_lon = ZONE_CENTROIDS[user_input["cdr_zone"]]

                    sim_with_geo = sim.merge(
                        ercot_df[["INR", "lat", "lon"]], on="INR", how="left"
                    )
                    valid_geo = sim_with_geo.dropna(subset=["lat", "lon"])

                    center_lat = user_lat if user_lat else (
                        valid_geo["lat"].mean() if len(valid_geo) else 31.0)
                    center_lon = user_lon if user_lon else (
                        valid_geo["lon"].mean() if len(valid_geo) else -99.5)
                    mini = folium.Map(location=[center_lat, center_lon],
                                      zoom_start=6, tiles="CartoDB positron",
                                      control_scale=False)
                    for i, r in valid_geo.reset_index(drop=True).iterrows():
                        folium.CircleMarker(
                            location=[r["lat"], r["lon"]],
                            radius=4, color="#00B0AC", fill=True,
                            fill_color="#00B0AC", fill_opacity=0.9, weight=1,
                            tooltip=(f"<b>#{i+1} · {r['Project Name']}</b><br>"
                                     f"{r['fuel_label']} · {fmt_mw(r['capacity_mw'])} MW<br>"
                                     f"Similarity: {r['similarity_pct']:.0f}%"),
                        ).add_to(mini)
                    if user_lat is not None and user_lon is not None:
                        folium.Marker(
                            location=[user_lat, user_lon],
                            icon=folium.Icon(color="darkgreen", icon="star", prefix="fa"),
                            tooltip=f"<b>Your project</b><br>{user_input['fuel_label']} · "
                                    f"{user_input['capacity_mw']:.0f} MW · {user_county}",
                        ).add_to(mini)
                    st_folium(mini, height=240, use_container_width=True,
                              returned_objects=[])
                    st.markdown(
                        "<div class='method-figure-caption'>Geographic spread — "
                        "teal dots = top 5 matches, star = your project</div>",
                        unsafe_allow_html=True,
                    )

                    # ── 2D feature-space scatter ──────────────────
                    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=120)
                    bg = ercot_df.dropna(subset=["capacity_mw", "months_in_queue"]).copy()
                    bg = bg[bg["capacity_mw"] > 0]
                    ax.scatter(
                        np.log1p(bg["capacity_mw"]),
                        bg["months_in_queue"].clip(lower=0, upper=120),
                        s=6, c="#CBD5E1", alpha=0.35, edgecolor="none",
                        label=f"All ERCOT queue (n={len(bg):,})",
                    )
                    sim_plot = sim.dropna(subset=["capacity_mw", "months_in_queue"]).copy()
                    sim_plot = sim_plot[sim_plot["capacity_mw"] > 0]
                    ax.scatter(
                        np.log1p(sim_plot["capacity_mw"]),
                        sim_plot["months_in_queue"].clip(lower=0, upper=120),
                        s=80, c="#00B0AC", alpha=0.95, edgecolor="white",
                        linewidth=1.0, label="Top 5 matches",
                    )
                    ax.scatter(
                        [np.log1p(max(user_input["capacity_mw"], 0.01))],
                        [0],
                        s=220, marker="*", c="#0F766E", edgecolor="white",
                        linewidth=1.2, label="Your project",
                    )
                    ax.set_xlabel("log(1 + Capacity MW)", fontsize=9, color="#475569")
                    ax.set_ylabel("Months in queue", fontsize=9, color="#475569")
                    ax.legend(fontsize=8, frameon=False, loc="upper right")
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
                    ax.tick_params(labelsize=8, colors="#64748B")
                    ax.set_facecolor("#FFFFFF")
                    fig.patch.set_facecolor("#F8FAFC")
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True, use_container_width=True)
                    st.markdown(
                        "<div class='method-figure-caption'>Feature-space view — "
                        "your project sits at month 0 (just filed); matches scatter by their "
                        "current capacity and how long they've already been in queue</div>",
                        unsafe_allow_html=True,
                    )

                    # ── Match cards with similarity bar + clear labels ──
                    st.markdown(
                        '<div class="section-label" style="margin-top:8px">'
                        'top 5 matches · ranked by similarity</div>',
                        unsafe_allow_html=True,
                    )
                    cards = []
                    for i, r in sim.iterrows():
                        mw_str  = fmt_mw(r["capacity_mw"])
                        miq     = r["months_in_queue"]
                        miq_str = "n/a" if pd.isna(miq) else f"{miq:.0f} mo"
                        sim_pct = float(r.get("similarity_pct", 0))
                        stage_pct = float(r["progress_pct"])
                        full_row = ercot_df[ercot_df["INR"] == r["INR"]].iloc[0]
                        strip = milestone_strip(full_row)
                        cards.append(
                            f"""<div class="match">
                              <div class="match-rank">#{i+1}</div>
                              <div class="match-name">{r['Project Name']}</div>
                              <div class="match-meta">
                                <span class="inr">{r['INR']}</span> · {r['fuel_label']} ·
                                {mw_str} MW · {r['County']} · {r['CDR Reporting Zone']}
                              </div>
                              <div class="similarity-row">
                                <span class="similarity-label">Similarity</span>
                                <div class="sim-track"><div class="sim-fill" style="width:{sim_pct:.0f}%"></div></div>
                                <span class="sim-value">{sim_pct:.0f}%</span>
                              </div>
                              <div class="stage-row">
                                <span class="stage-label">Stage progress</span>
                                <span class="stage-value">{r['current_stage']} · {miq_str} in queue</span>
                                <span class="stage-pct">{stage_pct:.0f}%</span>
                              </div>
                              {strip}
                            </div>"""
                        )
                    st.markdown("".join(cards), unsafe_allow_html=True)

                    st.markdown(
                        '<div class="note"><b>Reading this section.</b><br>'
                        '• <b>Similarity</b> is a 0–100% score derived from a weighted Gower-style '
                        'distance over fuel, zone, county, log-capacity, vintage, and planned lead time. '
                        '100% = identical input parameters.<br>'
                        '• <b>Stage progress</b> is the percentage of ERCOT GIM milestones already '
                        'achieved (e.g., 33% = 3 of 9 milestones).<br>'
                        '• The mini strip below each card shows those 9 milestones — '
                        'teal = achieved, grey = pending. Left = early studies, right = energised.</div>',
                        unsafe_allow_html=True,
                    )


# ═══════════════════════════════════════════════════════════════════
# Tab 2 · Methodology
# ═══════════════════════════════════════════════════════════════════
with tab_method:
    # ── Intro ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="doc-section">
          <h2>How the prediction works</h2>
          <div class="doc-sub">A two-layer system. Layer 1 is a frozen ML model trained on
          24,690 historical projects across all major US ISOs. Layer 2 is a live similarity
          match against today's ERCOT queue. Together they answer two questions:
          <b>will this project be built?</b> and <b>when?</b></div>

          <h3>The problem</h3>
          <p>83.8% of renewable energy projects that enter a US interconnection queue never get built.
          ERCOT alone holds <b>~420 GW</b> across 1,800+ large-gen projects today (up from a
          fraction of that in 2019). Developers and investors need a data-driven answer to project
          viability <i>at the moment of application</i>, before spending capital on interconnection
          studies — but historical averages alone don't capture today's queue dynamics.</p>

          <h3>Why two layers</h3>
          <p>The LBNL training data ends 2024 but most post-2019 ERCOT filings haven't yet
          completed or withdrawn — they show up as "active" in the data and are excluded from
          supervised training. A model fit on the full history under-represents today's congestion
          regime. Layer 2 corrects for that by matching the user's project to the most similar
          <i>live</i> ERCOT projects and showing their actual milestone progress as a reality check.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section: Interconnection Prediction ─────────────────────────
    st.markdown(
        """
        <div class="doc-section">
          <h2>Interconnection Prediction (Probability &amp; Duration)</h2>
          <div class="doc-sub">Two XGBoost gradient-boosted models trained on the LBNL Queued Up
          dataset, with per-instance SHAP attribution.</div>

          <h3>Method</h3>
          <p>Two separate XGBoost models share the same feature space:</p>
          <ul>
            <li><b>Classifier</b> — predicts P(project reaches commercial operation). Binary target:
                <code>will_complete = 1</code> if the project's status is "operational",
                <code>0</code> if "withdrawn". Trained on 24,690 completed-or-withdrawn rows.
                Class imbalance handled with <code>scale_pos_weight</code> = n_neg / n_pos.</li>
            <li><b>Regressor</b> — predicts queue duration in months for projects that complete.
                Target: <code>(actual_online_date − queue_date)</code> in months. Trained on
                completed projects only for clean signal; 1,722 rows survive after quality filters.</li>
          </ul>

          <p>Both models use a <b>time-aware split</b> to prevent temporal leakage:
          train on pre-2019 cohorts, validate on 2019–21, test on 2022+. The validation set is
          the appropriate benchmark — the test set's recent filings haven't matured enough for
          a fair completion-rate comparison.</p>

          <h3>Features</h3>
          <p>All 20 features use information observable at or before queue entry to prevent
          data leakage:</p>
          <ul>
            <li><b>Project specs:</b> <code>log_capacity_mw</code>, <code>capacity_bucket</code>
                (small / mid / large / utility), <code>tech_bucket</code> (Solar, Wind, Battery,
                Hybrid, Gas, Fossil, Nuclear, Hydro, Other), <code>is_hybrid</code></li>
            <li><b>Location:</b> <code>iso_region</code>, <code>is_slow_iso</code> (PJM/MISO flag)</li>
            <li><b>Filing intent:</b> <code>service_type</code> (NRIS / ERIS / NRIS+ERIS / Other),
                <code>is_nris</code>, <code>has_proposed_date</code>,
                <code>proposed_lead_years</code></li>
            <li><b>Temporal &amp; policy:</b> <code>queue_year</code>, <code>queue_month</code>,
                <code>queue_quarter</code>, <code>queue_decade</code>, <code>post_ferc_2003</code>,
                <code>post_ferc_2023</code>, <code>itc_active</code>, <code>ptc_bonus_period</code>,
                <code>ira_era</code></li>
            <li><b>Queue dynamics:</b> <code>log_queue_backlog</code> — 3-year rolling project
                count in the same ISO at filing date.
                <i>ERCOT inference uses the current queue size instead.</i></li>
          </ul>

          <h3>Validation performance</h3>
          <table class="perf-table">
            <thead>
              <tr><th>model</th><th>metric</th><th>direction</th>
                  <th>train</th><th>val</th><th>test</th><th>baseline</th></tr>
            </thead>
            <tbody>
              <tr class="highlight">
                <td><b>Classifier</b></td><td>AUC-ROC</td>
                <td style="font-size:11px;color:var(--slate-500)">higher = better ↑</td>
                <td class="num">0.888</td><td class="num">0.851</td><td class="num">0.771</td>
                <td class="num">0.500 (random)</td>
              </tr>
              <tr>
                <td>Classifier</td><td>AUC-PR</td>
                <td style="font-size:11px;color:var(--slate-500)">higher = better ↑</td>
                <td class="num">0.702</td><td class="num">0.384</td><td class="num">0.114</td>
                <td class="num">0.071 (val base rate)</td>
              </tr>
              <tr class="highlight">
                <td><b>Regressor</b></td><td>RMSE (months)</td>
                <td style="font-size:11px;color:var(--slate-500)">lower = better ↓</td>
                <td class="num">13.3</td><td class="num">14.3</td>
                <td class="num">18.7&nbsp;<span style="font-size:10px;color:var(--slate-500)">(n=25)</span></td>
                <td class="num">21.9 / 33.8&nbsp;<span style="font-size:10px;color:var(--slate-500)">(val / test ISO-mean)</span></td>
              </tr>
              <tr>
                <td>Regressor</td><td>R²</td>
                <td style="font-size:11px;color:var(--slate-500)">higher = better ↑</td>
                <td class="num">0.813</td><td class="num">0.344</td>
                <td class="num" style="color:var(--slate-500)">n/a (n too small)</td>
                <td class="num">—</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="doc-section">
          <h3>How to read the metrics</h3>
          <p>Each metric in the table has a <b>direction</b> (higher or lower is better) and a
          <b>baseline</b> (the naive score the model has to clear to be useful). A model that
          can't beat its baseline isn't adding value — but in this case every metric clears
          its baseline by a wide margin.</p>

          <p style="margin-bottom:6px"><b>AUC-ROC · classifier · higher = better ↑</b></p>
          <ul style="margin-top:2px">
            <li>Range 0.5 (random guessing) to 1.0 (perfect ranking).</li>
            <li>Interpretation: <i>"of every random pair of one completed and one withdrawn
                project, in what fraction does the model assign the completed one a higher
                probability?"</i></li>
            <li>Our validation 0.851 means the model gets that ordering right 85% of the time —
                substantially above 0.5 random. Clearing the baseline here is exactly what we
                want; the further above 0.5, the better.</li>
          </ul>

          <p style="margin-bottom:6px"><b>AUC-PR · classifier · higher = better ↑</b></p>
          <ul style="margin-top:2px">
            <li>Range = (positive-class base rate) to 1.0. More sensitive than AUC-ROC when the
                positive class is rare, as it is here (7.1% completion rate on the validation set).</li>
            <li>Baseline 0.071 = the precision you'd get by labelling every project as
                "will complete" (the trivial all-positive predictor). Our 0.384 is <b>5.4× better</b>
                than that baseline.</li>
            <li>AUC-PR drops on the test set (0.114) because the post-2022 cohort hasn't matured —
                most projects that will eventually complete are still "active". Validation is the
                appropriate benchmark.</li>
          </ul>

          <p style="margin-bottom:6px"><b>RMSE · regressor · lower = better ↓</b></p>
          <ul style="margin-top:2px">
            <li>Root Mean Squared Error in months. Measures the typical size of the prediction error.</li>
            <li>Baseline 21.9 months = the error you'd get with a naive "just predict the
                ISO-mean duration" model — i.e., the value of knowing nothing about the project
                beyond its region.</li>
            <li>Our validation 14.3 months is <b style="color:var(--teal-700)">35% lower</b>
                than that baseline, which means the model is finding real signal beyond
                region alone — capacity, technology, planned lead time, etc. all add information.</li>
          </ul>

          <p style="margin-bottom:6px"><b>R² · regressor · higher = better ↑</b></p>
          <ul style="margin-top:2px">
            <li>Range -∞ to 1.0. R² = 1 is perfect prediction; R² = 0 means the model is no better
                than predicting the mean; R² &lt; 0 means worse than predicting the mean.</li>
            <li>Validation R² of 0.344 means the model explains 34% of the variance in actual
                queue durations — moderate, with the rest absorbed by unobservable factors
                (developer-specific delays, surprise FERC reforms, financing shocks).</li>
            <li>R² is unstable for very small samples and we don't report it for the test set
                (n=25, see next note).</li>
          </ul>

          <h3>Why the test-set regressor cells are noisier</h3>
          <p>The test split covers all 2022+ filings — 3,207 projects total. The regressor only
          trains and evaluates on rows where the project has actually completed AND has a
          recorded online date:</p>
          <ul>
            <li>3,207 projects in test → only <b>74 are operational</b> (most are still active,
                ~3 years isn't long enough to clear ERCOT's median ~42-month queue).</li>
            <li>Of those 74, only <b>25 have a recorded actual-online-date</b> in the LBNL data.
                LBNL's known limitation: ~50% of operational projects ship without an end-date
                value, and that fraction is even higher for recent vintages.</li>
            <li>n=25 is enough to compute RMSE meaningfully — 18.7 months, vs an ISO-mean
                baseline of 33.8 months on the same 25 rows (the model still beats baseline by
                <b style="color:var(--teal-700)">45%</b>). It's <i>not</i> enough to give a
                stable R², which is why we mark that cell n/a.</li>
          </ul>

          <div class="doc-callout">
            <b>Bottom line for the classifier baseline question:</b> "Above baseline" for AUC
            metrics means the model is doing real work above random/trivial predictions. It's
            the opposite direction from RMSE, where "below baseline" is the win. The "direction"
            column in the table makes that explicit for each metric.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Figure 1 · Classifier feature importance ───────────────────
    img1, exp1 = st.columns([5, 7], gap="medium")
    with img1:
        st.image("outputs/clf_feature_importance.png", use_container_width=True)
    with exp1:
        st.markdown(
            """
            <div class="figure-block">
              <div class="figure-title">Figure 1 · Classifier feature importance</div>
              <p class="figure-body">
              The chart ranks the 15 features that contributed the most signal during classifier
              training, measured by XGBoost's gain metric. <b>Service type (NRIS/ERIS)</b> is the
              strongest single feature — projects requesting full network resource service
              historically complete at much higher rates than energy-only filings. The
              <b>presence of a proposed online date</b> ranks #6, which is why we expose it as a
              separate actionable input in the form. <b>ISO region</b>, <b>technology bucket</b>,
              and <b>capacity bucket</b> all contribute meaningfully — confirming that
              <i>where</i>, <i>what</i>, and <i>how big</i> each carry independent signal.
              </p>
              <div class="figure-takeaway">
                <b>Actionable levers for you:</b> service type, hybrid flag, capacity tier,
                and whether you file with a proposed COD. Everything else is largely fixed by
                the project's economics or context.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div class="figure-divider"></div>', unsafe_allow_html=True)

    # ── Figure 2 · Regressor feature importance ────────────────────
    img2, exp2 = st.columns([5, 7], gap="medium")
    with img2:
        st.image("outputs/reg_feature_importance.png", use_container_width=True)
    with exp2:
        st.markdown(
            """
            <div class="figure-block">
              <div class="figure-title">Figure 2 · Regressor feature importance</div>
              <p class="figure-body">
              Same construction, ranked by contribution to predicted queue duration.
              The dominant feature is the <b>hybrid flag</b> at 15.6% of total gain, nearly
              double the next contributor — Solar+Battery and Wind+Battery projects take
              measurably longer to clear studies (~15 extra months) even when they complete.
              <b>Proposed lead time</b> is the second-largest contributor: developers who
              plan longer runways correlate near-linearly with longer actual duration, because
              they're filing into more complex study environments.
              </p>
              <div class="figure-takeaway">
                <b>Why this matters for ERCOT:</b> as the pipeline tilts toward BESS hybrids,
                duration buffers grow. If timing matters, consider separating the generator
                and BESS as standalone applications — they can be re-combined commercially
                after IA.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div class="figure-divider"></div>', unsafe_allow_html=True)

    # ── Figure 3 · Regressor actual vs predicted ───────────────────
    img3, exp3 = st.columns([5, 7], gap="medium")
    with img3:
        st.image("outputs/reg_actual_vs_predicted.png", use_container_width=True)
    with exp3:
        st.markdown(
            """
            <div class="figure-block">
              <div class="figure-title">Figure 3 · Regressor calibration · actual vs predicted</div>
              <p class="figure-body">
              Each dot is one validation-set project: actual queue duration on the x-axis,
              the model's prediction on the y-axis. The dashed red line is perfect prediction
              (y = x); dots above are over-predictions, below are under-predictions. The tight
              cluster in the <b>20–60 month range</b> — where most completed projects fall —
              indicates strong calibration for typical project profiles. Predictions become
              less precise for the rarer long-duration projects (80+ months).
              </p>
              <div class="figure-takeaway">
                <b>RMSE = 14.3 months on validation</b>, vs an ISO-mean baseline of 21.9
                months — a <b>35% improvement</b>. If you predict the regional average,
                you'd be off by ~22 months; with the model, ~14 months.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div class="figure-divider"></div>', unsafe_allow_html=True)

    # ── Figure 4 · ISO calibration ─────────────────────────────────
    img4, exp4 = st.columns([5, 7], gap="medium")
    with img4:
        st.image("outputs/clf_completion_rate_by_iso.png", use_container_width=True)
    with exp4:
        st.markdown(
            """
            <div class="figure-block">
              <div class="figure-title">Figure 4 · Classifier calibration by ISO region</div>
              <p class="figure-body">
              Each ISO is plotted as a pair of bars: actual completion rate (blue) vs the
              model's average predicted rate (green). What matters most for relative scoring
              is that <b>the model preserves the ISO ranking</b> — SPP and ISO-NE rank as the
              strongest markets in both actual and predicted, while CAISO, NYISO, and ERCOT
              rank lowest in both. The upward bias in predicted percentages reflects the
              time-aware training: pre-2019 cohorts (where 21.8% completed) applied to a full
              dataset that includes newer still-active projects.
              </p>
              <div class="figure-takeaway">
                <b>How to use this:</b> trust the <i>ranking</i> across markets, not the
                absolute predicted percentages. ERCOT is correctly identified as structurally
                harder than SPP today — which is exactly why Project Clustering exists to
                ground-truth the prediction against the live ERCOT queue.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div class="figure-divider"></div>', unsafe_allow_html=True)

    # ── Figure 5 · SHAP beeswarm ───────────────────────────────────
    img5, exp5 = st.columns([5, 7], gap="medium")
    with img5:
        st.image("outputs/shap_clf_beeswarm.png", use_container_width=True)
    with exp5:
        st.markdown(
            """
            <div class="figure-block">
              <div class="figure-title">Figure 5 · SHAP beeswarm · classifier</div>
              <p class="figure-body">
              Each dot is one validation project. Horizontal position is the feature's SHAP
              value — how much it pushed that project's predicted probability up (right) or
              down (left). Dot colour is the feature value (red = high, blue = low). Three
              patterns: <b>(1)</b> <code>log_capacity_mw</code> red dots cluster on the left
              — large capacity reduces predicted probability. <b>(2)</b> <code>queue_year</code>
              red dots (recent vintage) shift left — recent filings face tougher conditions.
              <b>(3)</b> binary flags like <code>has_proposed_date</code> show clear bimodal
              splits — filing with a COD is a strong commitment signal.
              </p>
              <div class="figure-takeaway">
                <b>Same engine, same logic, per-project:</b> when you submit a form, a
                per-instance SHAP computation tells us which features pushed your specific
                prediction up or down — and the narratives module turns each contribution
                into the plain-English driver cards you see on the Predictor tab.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="doc-section">
          <h3>Per-instance SHAP &amp; the insight cards</h3>
          <p>When the predictor runs on a single user input, a <code>shap.TreeExplainer</code>
          computes the contribution of each feature to that specific prediction. The top
          positive contributors and top negative contributors are surfaced in the UI as
          compact factor rows (with raw SHAP values) and then expanded into plain-English
          insight cards via <code>src/ercot/narratives.py</code>:</p>
          <ul>
            <li>Each feature → label + context-specific narrative
                (completion vs duration use <i>different</i> narratives because the same feature
                can push these in opposite directions)</li>
            <li>Actionable features (capacity, hybrid, projected COD, service type) get a
                tactical suggestion when the direction is sub-optimal</li>
            <li>Non-actionable features (filing year, ISO region, policy era) are tagged
                "context · no action" — explanation only</li>
          </ul>

          <h3>ERCOT-specific mapping</h3>
          <div class="doc-callout">
            The LBNL feature schema uses two concepts ERCOT doesn't:
            <ul style="margin:8px 0 0">
              <li><b>NRIS / ERIS service type</b> — defaulted to <code>NRIS+ERIS</code>
                  (the modal best-case category in LBNL).</li>
              <li><b>3-year rolling backlog</b> — replaced with the <i>current ERCOT queue size</i>
                  (~1,800) so the prediction reflects today's congestion, not the historical average.</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section: Project Clustering ─────────────────────────────────
    st.markdown(
        """
        <div class="doc-section">
          <h2>Project Clustering (Live ERCOT Peers)</h2>
          <div class="doc-sub">A k-Nearest-Neighbours search over the current ERCOT queue,
          using a weighted Gower-style distance over mixed numeric and categorical features.</div>

          <h3>Why KNN and not another model</h3>
          <p>KNN is the natural choice here for three reasons:</p>
          <ul>
            <li><b>Interpretability.</b> The output is concrete reference projects — not an
                abstract score. A developer can read the 5 matches' actual milestone history
                and form their own judgement.</li>
            <li><b>No training required.</b> The "model" is the live queue itself; every
                monthly GIS refresh updates the search corpus automatically.</li>
            <li><b>Cold-start robust.</b> Unlike a learned similarity model, KNN doesn't need
                historical labels for ERCOT projects post-2019 — it just measures distance
                to whatever's in the queue right now.</li>
          </ul>

          <h3>Distance formula</h3>
          <p>The total distance between the user's project <code>q</code> and a candidate
          project <code>p</code> is:</p>
          <p style="margin-left:16px;font-family:'JetBrains Mono',monospace;color:var(--slate-900);
                    font-size:13px">
            d(q, p) = √(Σᵢ (zᵢ(q) − zᵢ(p))²) + Σⱼ wⱼ · 𝟙{cⱼ(q) ≠ cⱼ(p)}
          </p>
          <p>where the first term is standardised Euclidean over numeric features, and the
          second is a weighted mismatch over categorical features.</p>

          <h3>Feature weights</h3>
          <table class="perf-table">
            <thead>
              <tr><th>feature</th><th>type</th><th>weight</th><th>rationale</th></tr>
            </thead>
            <tbody>
              <tr><td><code>log_capacity_mw</code></td><td>numeric (z-scored)</td>
                  <td class="num">1.0</td><td>Larger capacity → different study scope.</td></tr>
              <tr><td><code>vintage_year</code></td><td>numeric (z-scored)</td>
                  <td class="num">1.0</td><td>Newer queue entries face different conditions.</td></tr>
              <tr><td><code>projected_lead_years</code></td><td>numeric (z-scored)</td>
                  <td class="num">1.0</td><td>COD lead time signals commitment + complexity.</td></tr>
              <tr><td><code>fuel_label</code></td><td>categorical</td>
                  <td class="num">2.0</td><td>Single strongest peer signal — Solar peers with Solar.</td></tr>
              <tr><td><code>County</code></td><td>categorical</td>
                  <td class="num">1.5</td><td>Same-county = same transmission node, same permits.</td></tr>
              <tr><td><code>CDR Reporting Zone</code></td><td>categorical</td>
                  <td class="num">1.0</td><td>Same load zone = same study region.</td></tr>
            </tbody>
          </table>

          <h3>From distance to similarity %</h3>
          <p>Distances are converted to a 0–100% similarity score using the queue's maximum
          observed distance as the "completely dissimilar" anchor:</p>
          <p style="margin-left:16px;font-family:'JetBrains Mono',monospace;color:var(--slate-900);
                    font-size:13px">
            similarity(q, p) = (1 − d(q, p) / max d) × 100%
          </p>
          <p>Higher = more similar. A match at 95% means the project is nearly indistinguishable
          from the user's input on the weighted feature space; 60% means it's a moderate peer.</p>

          <h3>Step-by-step example</h3>
          <ol>
            <li>User submits: <code>{capacity_mw: 200, fuel: Battery, county: Crockett,
                zone: WEST, COD: 2027-12}</code></li>
            <li>Engine standardises numeric features against the live queue's distribution.</li>
            <li>For every active ERCOT project, compute d(q, p) using the formula above.</li>
            <li>Sort ascending; take top 5.</li>
            <li>Convert each distance to a similarity %; render the 5 matches with their
                current GIM milestone state from the loader.</li>
          </ol>

          <h3>What the visualisations show</h3>
          <ul>
            <li><b>Mini map.</b> Geographic spread of the 5 matches plus a star at your project's
                county centroid. Same-county matches cluster around your star; cross-county
                matches show how far the peer set has to reach.</li>
            <li><b>Feature-space scatter.</b> Every ERCOT queue project plotted on
                (log-capacity, months-in-queue). Your project sits at month 0 (just filed);
                matches scatter by where they are in the queue today. Cluster tightness around
                your point indicates how representative your peer set is.</li>
            <li><b>Match cards.</b> Per-match: name, INR, fuel, capacity, location, similarity %
                bar, current stage, months in queue, and the 9-milestone strip.</li>
          </ul>

          <h3>Refresh cadence</h3>
          <p>The search corpus is the ERCOT GIS Report, refreshed monthly via a GitHub Actions
          cron (see the <b>Data</b> tab for details). Layer 1 stays frozen; Layer 2 always
          reflects the latest available queue snapshot.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# Tab 3 · Data
# ═══════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown(
        f"""
        <div class="doc-section">
          <h2>Data sources</h2>
          <div class="doc-sub">Two primary datasets, plus a geocoding lookup. Refreshed on different cadences.</div>

          <table class="doc-table">
            <thead><tr>
              <th>source</th><th>scope</th><th>used for</th><th>refresh</th>
            </tr></thead>
            <tbody>
              <tr>
                <td><b>LBNL Queued Up 2024</b></td>
                <td>36,441 projects · all major US ISOs · 1970–2024</td>
                <td>Layer 1 ML training (24,690 completed + withdrawn rows)</td>
                <td>Annual (LBNL release cycle)</td>
              </tr>
              <tr>
                <td><b>ERCOT GIS Report</b></td>
                <td>{n_proj:,} active large-gen projects today · ERCOT only</td>
                <td>Layer 2 similarity + queue overview + map</td>
                <td>Monthly (auto-refreshed)</td>
              </tr>
              <tr>
                <td><b>US Census 2020 county centroids (Texas)</b></td>
                <td>254 Texas counties · mean center of population</td>
                <td>Map coordinates for ERCOT projects</td>
                <td>Static (decennial)</td>
              </tr>
              <tr>
                <td><b>HIFLD Substations</b> (optional)</td>
                <td>~30k US substations · public infrastructure dataset</td>
                <td>Substation-level pinpoint coordinates (upgrade over county centroid)</td>
                <td>Static (user opt-in)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="doc-section">
          <h3>Monthly refresh — how it works</h3>
          <p>A GitHub Actions workflow runs at 14:00 UTC on the 5th of every month:</p>
          <ol>
            <li>Scrapes the ERCOT MIS endpoint
              (<code>mis.ercot.com/misapp/GetReports.do?reportTypeId=15933</code>)
              for the newest GIS Report .xlsx — no authentication required.</li>
            <li>Downloads it to <code>data/ercot/snapshots/YYYY-MM.xlsx</code>.</li>
            <li>Updates the <code>data/ercot/latest.xlsx</code> pointer.</li>
            <li>Commits the new file and pushes — Streamlit Cloud auto-redeploys.</li>
          </ol>
          <p>You can also trigger it manually from the <b>Actions</b> tab of the GitHub repo,
          or run <code>python -m src.ercot.fetcher</code> locally.</p>
        </div>

        <div class="doc-section">
          <h3>What's in the ERCOT GIS Report</h3>
          <p>The "Project Details – Large Gen" sheet (1,800+ rows) is the core feed.
          Each row is one interconnection request with:</p>
          <ul>
            <li><b>Project identifier</b> — INR, project name, interconnecting entity</li>
            <li><b>Location</b> — POI substation name, county, CDR reporting zone</li>
            <li><b>Project specs</b> — fuel code, technology, capacity (MW; negative for net-change repowering)</li>
            <li><b>Milestones</b> — date-stamped columns for Screening Study Started/Complete,
                FIS Requested/Approved, IA Signed, Construction Start/End, Approved for Energization,
                Approved for Synchronization</li>
            <li><b>Permits</b> — Air, GHG, Water Availability (where required)</li>
            <li><b>Planning</b> — meets Guide Section 6.9 requirements</li>
          </ul>
          <p>The loader (<code>src/ercot/loader.py</code>) parses these into a tidy DataFrame and
          computes derived columns: <code>milestones_done</code>, <code>progress_pct</code>,
          <code>current_stage</code>, <code>queue_entry_date</code>, <code>months_in_queue</code>.</p>
        </div>

        <div class="doc-section">
          <h3>Known limitations</h3>
          <ul>
            <li><b>ERCOT post-2019 dynamics underrepresented in Layer 1.</b> LBNL training data ends 2024
                but most recent ERCOT filings haven't yet completed or withdrawn. Layer 2 exists to
                cover this gap.</li>
            <li><b>NRIS/ERIS doesn't apply to ERCOT.</b> The strongest classifier feature in the LBNL
                data isn't a real ERCOT concept; we default it to NRIS+ERIS, which is documented in
                the per-prediction "Mapping assumptions" expander.</li>
            <li><b>Inactive (INA) projects excluded.</b> The GIS Report Large-Gen sheet is restricted
                to projects with a Full Interconnection Study requested. Withdrawn / cancelled
                projects sit on separate sheets (not yet wired into Layer 2 — planned upgrade).</li>
            <li><b>Capacity factor</b> is in the input form but is not a feature in the trained
                LBNL model — it influences Layer 2 similarity weights only.</li>
            <li><b>Coordinates use county centroids by default.</b> Multiple projects in the same
                county render at the centroid with small deterministic jitter. Substation-precision
                coordinates require dropping a HIFLD substations CSV into <code>data/geo/</code>.</li>
            <li><b>Repowering projects have net-change capacity</b> — sometimes negative or zero.
                The map uses absolute value for marker sizing and flags repowering in the tooltip.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top:24px;padding-top:14px;border-top:1px solid var(--slate-200);
                font-size:11px;color:var(--slate-500);
                font-family:'JetBrains Mono',monospace">
      data: ERCOT GIS Report (monthly) + LBNL Queued Up 2024 ·
      layer 1 model frozen · layer 2 reflects live queue ·
      ERCOT post-2019 dynamics underrepresented in layer 1
    </div>
    """,
    unsafe_allow_html=True,
)
