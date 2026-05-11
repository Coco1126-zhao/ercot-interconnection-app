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

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from src.ercot.loader     import load_gis_large_gen, MILESTONE_COLS_LARGE
from src.ercot.geo_utils  import attach_coordinates
from src.ercot.similarity import SimilarityEngine
from src.ercot.predictor  import Predictor

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
}
.metric .label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--slate-500);
}
.metric .value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 34px; font-weight: 600;
  color: var(--teal-700); line-height: 1.05; margin-top: 6px;
}
.metric .unit { font-size: 14px; color: var(--slate-500); margin-left: 4px; }

/* Queue feed list */
.feed { max-height: 360px; overflow-y: auto; padding-right: 4px; }
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


# ── Two-column main layout ─────────────────────────────────────────
LEFT, RIGHT = st.columns([5, 8], gap="large")


# ═══════════════════ LEFT: live ERCOT queue ═══════════════════════
with LEFT:
    # Map card ------------------------------------------------------
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

    # Quick stats ---------------------------------------------------
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

    # Queue feed ----------------------------------------------------
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
            f"    <div class='feed-pct'>{pct:.0f}%</div>"
            f"    <div class='feed-stage'>{r['current_stage']} · {miq_str}</div>"
            f"    <div class='bar-track'><div class='bar-fill' style='width:{pct:.0f}%'></div></div>"
            f"  </div>"
            f"</div>"
        )
    feed_html += "</div>"
    st.markdown(feed_html, unsafe_allow_html=True)


# ═══════════════════ RIGHT: prediction (main function) ═════════════
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

        submitted = st.form_submit_button("⚡  Run prediction",
                                          use_container_width=False)

    # ─── Output area ──────────────────────────────────────────────
    if not submitted:
        st.markdown(
            '<div class="placeholder">Submit a project above to see '
            '<b>Layer 1</b> (LBNL ML estimate + drivers) and '
            '<b>Layer 2</b> (5 most-similar live ERCOT projects with their actual progress).</div>',
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

        L1col, L2col = st.columns([1, 1], gap="medium")

        # ── Layer 1 ───────────────────────────────────────────────
        with L1col:
            st.markdown('<div class="section-label">layer 1 · ML prediction</div>',
                        unsafe_allow_html=True)

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
                        <div class="label">P · reaches operation</div>
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

                st.markdown(
                    '<div class="section-label" style="margin-top:14px">'
                    'drivers · completion probability</div>',
                    unsafe_allow_html=True,
                )
                rows = []
                for n, v in res["clf_top_positive"][:4]:
                    rows.append(
                        f"<div class='factor'>"
                        f"<span class='dot-pos'></span>"
                        f"<span class='name'>{n}</span>"
                        f"<span class='val'>+{v:.3f}</span></div>"
                    )
                for n, v in res["clf_top_negative"][:4]:
                    rows.append(
                        f"<div class='factor'>"
                        f"<span class='dot-neg'></span>"
                        f"<span class='name'>{n}</span>"
                        f"<span class='val'>{v:+.3f}</span></div>"
                    )
                st.markdown("".join(rows), unsafe_allow_html=True)

                st.markdown(
                    '<div class="section-label" style="margin-top:14px">'
                    'drivers · duration  '
                    '<span style="color:var(--teal-700)">teal = shortens</span> · '
                    '<span style="color:var(--rose)">rose = lengthens</span></div>',
                    unsafe_allow_html=True,
                )
                rows = []
                for n, v in res["reg_top_speedup"][:3]:
                    rows.append(
                        f"<div class='factor'>"
                        f"<span class='dot-pos'></span>"
                        f"<span class='name'>{n}</span>"
                        f"<span class='val'>{v:+.2f} mo</span></div>"
                    )
                for n, v in res["reg_top_slowdown"][:3]:
                    rows.append(
                        f"<div class='factor'>"
                        f"<span class='dot-neg'></span>"
                        f"<span class='name'>{n}</span>"
                        f"<span class='val'>{v:+.2f} mo</span></div>"
                    )
                st.markdown("".join(rows), unsafe_allow_html=True)

                with st.expander("Mapping assumptions (ERCOT → LBNL)"):
                    st.json(res["assumptions"])

        # ── Layer 2 ───────────────────────────────────────────────
        with L2col:
            st.markdown('<div class="section-label">layer 2 · similar live projects</div>',
                        unsafe_allow_html=True)
            eng = get_similarity_engine(n_proj)
            sim = eng.find_similar(user_input, n=5)

            cards = []
            for _, r in sim.iterrows():
                mw_str  = fmt_mw(r["capacity_mw"])
                miq     = r["months_in_queue"]
                miq_str = "n/a" if pd.isna(miq) else f"{miq:.0f} mo"
                pct     = r["progress_pct"]
                # mini timeline strip from full ercot_df row (sim rows lack _done_ cols)
                full_row = ercot_df[ercot_df["INR"] == r["INR"]].iloc[0]
                strip = milestone_strip(full_row)
                cards.append(
                    f"""<div class="card card-tight">
                      <div style="display:flex;justify-content:space-between;align-items:start;gap:8px">
                        <div>
                          <div class="feed-name">{r['Project Name']}</div>
                          <div class="feed-meta">
                            <span class="inr">{r['INR']}</span> · {r['fuel_label']} ·
                            {mw_str} MW · {r['County']} · {r['CDR Reporting Zone']}
                          </div>
                        </div>
                        <div style="text-align:right">
                          <div class="feed-pct">{pct:.0f}%</div>
                          <div class="feed-stage">d={r['similarity_distance']:.2f}</div>
                        </div>
                      </div>
                      <div class="feed-meta" style="margin-top:6px">
                        stage: <b style="color:var(--slate-700)">{r['current_stage']}</b> · {miq_str} in queue
                      </div>
                      {strip}
                    </div>"""
                )
            st.markdown("".join(cards), unsafe_allow_html=True)

            st.markdown(
                '<div class="note">Distance: weighted mix of fuel, zone, county, '
                'log-MW, vintage, and projected lead time. Lower = more similar. '
                'The mini strip shows ERCOT GIM milestones (left = early, right = energised).</div>',
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
