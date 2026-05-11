"""
ercot/loader.py
===============
Parse a monthly ERCOT GIS Report xlsx into a clean DataFrame of
queued large-generator projects with milestone-progress columns.

The GIS Report layout (verified against April 2026 file):
  - Sheet "Project Details - Large Gen", header row index = 30
  - Sheet "Project Details - Small Gen", header row index = 14
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

LARGE_GEN_HEADER_ROW = 30
SMALL_GEN_HEADER_ROW = 14

# Milestones in chronological order. Presence of a date = milestone reached.
MILESTONE_COLS_LARGE = [
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

# ERCOT short fuel codes → human-readable
FUEL_MAP = {
    "SOL": "Solar", "WIN": "Wind", "GAS": "Gas",
    "BAT": "Battery", "STO": "Battery", "NUC": "Nuclear",
    "WAT": "Hydro", "BIO": "Biomass", "COA": "Coal",
    "OIL": "Oil",  "OTH": "Other",
}


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def load_gis_large_gen(path: str | Path) -> pd.DataFrame:
    """Return a tidy DataFrame of large-gen ERCOT projects."""
    path = Path(path)
    df = pd.read_excel(
        path, sheet_name="Project Details - Large Gen",
        header=LARGE_GEN_HEADER_ROW,
    )

    # Drop footer / blank rows
    df = df[df["INR"].notna()].copy()

    # Normalize key columns
    df = df.rename(columns={"Capacity (MW)": "capacity_mw"})
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    df["fuel_label"]  = df["Fuel"].astype(str).str.strip().map(FUEL_MAP).fillna(df["Fuel"])
    df["County"]      = df["County"].astype(str).str.strip().str.title()
    df["CDR Reporting Zone"] = df["CDR Reporting Zone"].astype(str).str.strip().str.upper()

    # Parse milestone date columns + boolean "done" flags
    done_cols = []
    for col in MILESTONE_COLS_LARGE:
        if col in df.columns:
            df[col] = _to_dt(df[col])
            d = f"_done_{col}"
            df[d] = df[col].notna()
            done_cols.append(d)

    df["milestones_done"] = df[done_cols].sum(axis=1)
    df["milestones_total"] = len(done_cols)
    df["progress_pct"] = (df["milestones_done"] / len(done_cols)) * 100.0

    # Current stage = latest milestone reached
    def _stage(row):
        for col in reversed(MILESTONE_COLS_LARGE):
            if row.get(f"_done_{col}", False):
                return col
        return "Pre-Screening"
    df["current_stage"] = df.apply(_stage, axis=1)

    # Queue entry proxy: earliest study/FIS request date
    candidates = [c for c in ["Screening Study Started", "FIS Requested"] if c in df.columns]
    df["queue_entry_date"] = df[candidates].min(axis=1)

    today = pd.Timestamp.today().normalize()
    df["months_in_queue"] = ((today - df["queue_entry_date"]).dt.days / 30.4375).round(1)
    df["Projected COD"] = _to_dt(df["Projected COD"])

    df["sheet_source"] = "large_gen"
    return df.reset_index(drop=True)


def load_gis_small_gen(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(
        path, sheet_name="Project Details - Small Gen",
        header=SMALL_GEN_HEADER_ROW,
    )
    df = df[df["INR"].notna()].copy()
    df = df.rename(columns={"Capacity (MW)": "capacity_mw"})
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    df["fuel_label"]  = df["Fuel"].astype(str).str.strip().map(FUEL_MAP).fillna(df["Fuel"])
    df["County"]      = df["County"].astype(str).str.strip().str.title()
    df["CDR Reporting Zone"] = df["CDR Reporting Zone"].astype(str).str.strip().str.upper()
    df["sheet_source"] = "small_gen"
    return df.reset_index(drop=True)


def load_gis_report(path: str | Path) -> pd.DataFrame:
    """Combined large + small generator view."""
    big   = load_gis_large_gen(path)
    small = load_gis_small_gen(path)
    return pd.concat([big, small], ignore_index=True, sort=False)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "data/ercot/latest.xlsx"
    df = load_gis_large_gen(p)
    print(f"Loaded {len(df)} large-gen projects")
    print(df[["INR", "Project Name", "fuel_label", "capacity_mw",
              "current_stage", "progress_pct", "months_in_queue"]].head(10).to_string())
