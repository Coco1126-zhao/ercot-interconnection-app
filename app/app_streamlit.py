"""
app/app_streamlit.py
====================
ERCOT Interconnection Predictor — interface.

Layout (terminal aesthetic, dark / monospace):
  Left sidebar  : project input form
  Top main      : interactive map of ERCOT queue + commissioning
  Mid main      : scrollable terminal-style queue feed
  Bottom main   : on submit → two side-by-side panels
                  Layer 1 — ML prediction + SHAP factors
                  Layer 2 — top-N similar live projects + timelines

Run:  streamlit run app/app_streamlit.py
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

# ── Page config + theme ─────────────────────────────────────────────
st.set_page_config(
    page_title="ERCOT Interconnection Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

TERMINAL_CSS = """
<style>
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', 'SF Mono', 'Menlo', monospace !important;
}
.stApp { background-color: #0d1117; color: #c9f5c9; }
section[data-testid="stSidebar"] { background-color: #06090d; }
h1, h2, h3, h4 { color: #4ade80 !important; letter-spacing: 0.02em; }
.stButton>button {
    background: #14532d; color: #c9f5c9; border: 1px solid #4ade80;
    border-radius: 0; font-family: monospace; text-transform: uppercase;
}
.stButton>button:hover { background: #166534; border-color: #86efac; }
.metric-card {
    border: 1px solid #166534; padding: 12px 16px; margin-bottom: 8px;
    background: #0a1410;
}
.queue-card {
    border-left: 3px solid #4ade80; padding: 8px 12px; margin: 4px 0;
    background: #0a1410; font-size: 12px; line-height: 1.4;
}
.factor-pos { color: #4ade80; }
.factor-neg { color: #f87171; }
.timeline-row {
    display: flex; align-items: center; gap: 4px; font-size: 11px;
    margin: 2px 0;
}
.tl-done { color: #4ade80; }
.tl-todo { color: #4b5563; }
hr { border-color: #166534; }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


# ── Cached loaders ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading ERCOT GIS report...")
def load_ercot():
    df = load_gis_large_gen("data/ercot/latest.xlsx")
    df = attach_coordinates(df)
    return df


@st.cache_resource(show_spinner="Loading trained models...")
def load_predictor():
    try:
        return Predictor()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def get_similarity_engine(_df_hash: int):  # _df cached implicitly by streamlit
    return SimilarityEngine(load_ercot())


# ── Header ──────────────────────────────────────────────────────────
st.markdown("# > ERCOT_INTERCONNECTION_PREDICTOR")
st.markdown(
    "<span style='color:#86efac'>"
    "[STATUS: ONLINE] · LBNL-trained ML + live ERCOT GIS Report · "
    f"queue snapshot: <b>data/ercot/latest.xlsx</b>"
    "</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

ercot_df = load_ercot()
predictor = load_predictor()

# ── Sidebar — input form ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## > NEW_PROJECT_INPUT")
    st.caption("Enter project parameters to estimate viability and lookup similar live projects.")

    cap = st.number_input("Nameplate capacity (MW)", min_value=1.0, value=200.0, step=10.0)
    fuel = st.selectbox(
        "Fuel / Technology",
        ["Solar", "Wind", "Battery", "Gas", "Hydro", "Biomass", "Coal", "Oil", "Nuclear", "Other"],
        index=2,
    )
    is_hybrid = st.checkbox("Hybrid (e.g., Solar + Battery)", value=False)
    cap_factor = st.slider("Capacity factor (informational, used in Layer 2)",
                           0.05, 0.95, 0.30, 0.05)

    counties = sorted(ercot_df["County"].dropna().unique().tolist())
    county = st.selectbox("County (Texas)", counties,
                          index=counties.index("Crockett") if "Crockett" in counties else 0)
    zones = sorted(ercot_df["CDR Reporting Zone"].dropna().unique().tolist())
    zone = st.selectbox("CDR Reporting Zone", zones,
                        index=zones.index("WEST") if "WEST" in zones else 0)
    poi = st.text_input("POI / Substation (optional)", value="")

    cod = st.date_input("Projected COD",
                        value=pd.Timestamp.today().normalize() + pd.Timedelta(days=730))
    service = st.selectbox("Service request (LBNL feature; default for ERCOT)",
                           ["NRIS+ERIS", "NRIS", "ERIS"], index=0)

    submitted = st.button("> RUN_PREDICTION", use_container_width=True)


# ── Main area ───────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown("## > MAP · ERCOT_LIVE_QUEUE")
    plot_df = ercot_df.dropna(subset=["lat", "lon"]).copy()
    fmap = folium.Map(location=[31.0, -99.5], zoom_start=6,
                      tiles="CartoDB dark_matter")

    color_map = {
        "Approved for Synchronization": "#4ade80",
        "Approved for Energization":    "#86efac",
        "Construction End":             "#bbf7d0",
        "Construction Start":           "#fde68a",
        "IA Signed":                    "#fbbf24",
        "FIS Approved":                 "#f59e0b",
        "FIS Requested":                "#fb923c",
        "Screening Study Complete":     "#f97316",
        "Screening Study Started":      "#ef4444",
        "Pre-Screening":                "#7f1d1d",
    }
    for _, r in plot_df.iterrows():
        c = color_map.get(r["current_stage"], "#7f1d1d")
        # capacity can be NaN or negative (repowering: GIS report notes)
        mw_raw = r["capacity_mw"]
        mw = 0.0 if pd.isna(mw_raw) else abs(float(mw_raw))
        radius = max(3.0, min(10.0, mw ** 0.4))
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=radius,
            color=c, fill=True, fill_color=c, fill_opacity=0.75, weight=1,
            tooltip=(f"<b>{r['Project Name']}</b><br>"
                     f"INR: {r['INR']}<br>"
                     f"{r['fuel_label']} · {mw:.0f} MW"
                     + (" (repower net)" if (not pd.isna(mw_raw) and mw_raw < 0) else "") + "<br>"
                     f"Stage: {r['current_stage']}<br>"
                     f"Progress: {r['progress_pct']:.0f}%"),
        ).add_to(fmap)

    st_folium(fmap, height=440, use_container_width=True,
              returned_objects=[])

with right:
    st.markdown("## > QUEUE_FEED")
    sort_by = st.selectbox("Sort by", ["progress_pct ↓", "capacity_mw ↓",
                                       "months_in_queue ↓", "INR"], index=0)
    key, asc = {
        "progress_pct ↓":   ("progress_pct",   False),
        "capacity_mw ↓":    ("capacity_mw",    False),
        "months_in_queue ↓":("months_in_queue",False),
        "INR":              ("INR",            True),
    }[sort_by]
    feed = ercot_df.sort_values(key, ascending=asc)
    fuel_filter = st.multiselect(
        "Filter fuel", sorted(ercot_df["fuel_label"].dropna().unique()),
        default=[],
    )
    if fuel_filter:
        feed = feed[feed["fuel_label"].isin(fuel_filter)]

    st.caption(f"{len(feed)} projects ▼ scroll")
    feed_html = "<div style='max-height:420px; overflow-y:auto;'>"
    for _, r in feed.head(150).iterrows():
        mw = r["capacity_mw"]
        mw_str = "n/a" if pd.isna(mw) else f"{mw:.0f}"
        miq = r["months_in_queue"]
        miq_str = "n/a" if pd.isna(miq) else f"{miq:.0f}"
        feed_html += (
            f"<div class='queue-card'>"
            f"<b>[{r['INR']}]</b> {r['Project Name']}<br>"
            f"<span style='color:#86efac'>{r['fuel_label']}</span> · "
            f"{mw_str} MW · {r['County']} ({r['CDR Reporting Zone']})<br>"
            f"<span style='color:#fbbf24'>STAGE:</span> {r['current_stage']} · "
            f"{r['progress_pct']:.0f}% · {miq_str} mo in queue"
            f"</div>"
        )
    feed_html += "</div>"
    st.markdown(feed_html, unsafe_allow_html=True)

st.markdown("---")


# ── Helper for milestone strip ──────────────────────────────────────
def _milestone_strip(df: pd.DataFrame, inr: str) -> str:
    """Return inline-HTML mini timeline of milestones for one project."""
    row = df[df["INR"] == inr]
    if row.empty:
        return ""
    row = row.iloc[0]
    cells = []
    for col in MILESTONE_COLS_LARGE:
        done = bool(row.get(f"_done_{col}", False))
        klass = "tl-done" if done else "tl-todo"
        glyph = "■" if done else "·"
        short = "".join(w[0] for w in col.split())
        cells.append(f"<span class='{klass}' title='{col}'>{glyph} {short}</span>")
    return ("<div class='timeline-row'>" + " ".join(cells) + "</div>")


# ── Prediction outputs ──────────────────────────────────────────────
if submitted:
    user_input = {
        "capacity_mw":      float(cap),
        "fuel_label":       fuel,
        "is_hybrid":        int(is_hybrid),
        "county":           county,
        "cdr_zone":         zone,
        "poi":              poi,
        "projected_cod":    str(cod),
        "service_type":     service,
        "capacity_factor":  cap_factor,
    }

    L1, L2 = st.columns(2)

    # ───── Layer 1: ML prediction + SHAP ─────
    with L1:
        st.markdown("### > LAYER_1 · ML_PREDICTION  (LBNL-trained)")
        if predictor is None:
            st.error(
                "Model artifacts not found. Run:\n\n"
                "    python -m src.ercot.train_and_save\n\n"
                "from the project root, then refresh."
            )
        else:
            queue_size = int(len(ercot_df))
            res = predictor.predict(user_input, current_queue_size=queue_size)

            c1, c2 = st.columns(2)
            c1.markdown(
                f"<div class='metric-card'>"
                f"<div style='font-size:11px;color:#86efac'>P(REACHES_OPERATION)</div>"
                f"<div style='font-size:32px;color:#4ade80'>{res['prob_complete']:.1%}</div>"
                f"</div>", unsafe_allow_html=True)
            c2.markdown(
                f"<div class='metric-card'>"
                f"<div style='font-size:11px;color:#86efac'>EXPECTED_DURATION</div>"
                f"<div style='font-size:32px;color:#4ade80'>"
                f"{res['duration_months']:.1f} mo</div>"
                f"</div>", unsafe_allow_html=True)

            st.markdown("**Drivers — completion probability**")
            for n, v in res["clf_top_positive"][:5]:
                st.markdown(f"<span class='factor-pos'>+ {n}</span> "
                            f"<span style='color:#4b5563'>(SHAP {v:+.3f})</span>",
                            unsafe_allow_html=True)
            for n, v in res["clf_top_negative"][:5]:
                st.markdown(f"<span class='factor-neg'>- {n}</span> "
                            f"<span style='color:#4b5563'>(SHAP {v:+.3f})</span>",
                            unsafe_allow_html=True)

            st.markdown("**Drivers — duration**")
            st.caption("- shortens · + lengthens")
            for n, v in res["reg_top_speedup"][:3]:
                st.markdown(f"<span class='factor-pos'>- {n}</span> "
                            f"<span style='color:#4b5563'>(SHAP {v:+.2f} mo)</span>",
                            unsafe_allow_html=True)
            for n, v in res["reg_top_slowdown"][:3]:
                st.markdown(f"<span class='factor-neg'>+ {n}</span> "
                            f"<span style='color:#4b5563'>(SHAP {v:+.2f} mo)</span>",
                            unsafe_allow_html=True)

            with st.expander("Mapping assumptions (ERCOT → LBNL)"):
                st.json(res["assumptions"])

    # ───── Layer 2: similar live ERCOT projects ─────
    with L2:
        st.markdown("### > LAYER_2 · SIMILAR_LIVE_PROJECTS")
        eng = get_similarity_engine(len(ercot_df))
        sim = eng.find_similar(user_input, n=5)

        for _, r in sim.iterrows():
            mw = r["capacity_mw"]
            mw_str = "n/a" if pd.isna(mw) else f"{mw:.0f}"
            miq = r["months_in_queue"]
            miq_str = "n/a" if pd.isna(miq) else f"{miq:.0f}"
            st.markdown(
                f"<div class='queue-card'>"
                f"<b>[{r['INR']}]</b> {r['Project Name']}  "
                f"<span style='color:#4b5563'>(distance {r['similarity_distance']:.2f})</span><br>"
                f"<span style='color:#86efac'>{r['fuel_label']}</span> · "
                f"{mw_str} MW · {r['County']} ({r['CDR Reporting Zone']})<br>"
                f"<span style='color:#fbbf24'>STAGE:</span> {r['current_stage']} · "
                f"{r['progress_pct']:.0f}% · {miq_str} mo in queue<br>"
                + _milestone_strip(ercot_df, r["INR"]) +
                f"</div>", unsafe_allow_html=True
            )
        st.caption(
            "Distance is a weighted Gower-style mix of fuel, zone, county, "
            "log-MW, vintage, and projected lead time. Lower = more similar."
        )


# Footer
st.markdown("---")
st.caption(
    "Data: ERCOT GIS Report (monthly) + LBNL Queued Up 2024. "
    "Layer 1 model frozen; Layer 2 reflects live queue. "
    "ERCOT recent dynamics underrepresented in Layer 1 — see Findings #3 in main README."
)
