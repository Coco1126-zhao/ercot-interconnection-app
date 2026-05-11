"""
data_loader.py
==============
Loads and cleans the LBNL Interconnection Queue dataset (thru 2024).
Converts Excel serial dates, computes regression and classification
targets, and returns a modelling-ready DataFrame of completed /
withdrawn projects (active / suspended excluded from supervised targets).

Usage
-----
    from src.data_loader import load_queue_data
    df_raw, df_model = load_queue_data("data/raw/lbnl_ix_queue_data_file_thru2024.xlsx")
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Excel serial-date → Python datetime (Excel epoch = 1899-12-30) ─────────
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def _excel_to_date(series: pd.Series) -> pd.Series:
    """Convert a numeric Excel date column to datetime64."""
    return _EXCEL_EPOCH + pd.to_timedelta(series, unit="D")


# ── Main loader ─────────────────────────────────────────────────────────────
def load_queue_data(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    path : str | Path
        Path to the LBNL xlsx file.

    Returns
    -------
    df_raw : pd.DataFrame
        Full cleaned dataset (all statuses, 36k+ rows).
    df_model : pd.DataFrame
        Modelling subset: only completed ('operational') and withdrawn
        projects with valid queue entry dates and MW capacity.
        Includes computed target columns:
            - queue_duration_months  (regression target)
            - will_complete          (classification target: 1=operational, 0=withdrawn)
    """
    path = Path(path)
    print(f"Loading: {path.name}")

    df = pd.read_excel(
        path,
        sheet_name="03. Complete Queue Data",
        header=1,
    )

    # ── 1. Rename for clarity ────────────────────────────────────────────────
    rename = {
        "q_id":           "project_id",
        "q_status":       "status",
        "q_date":         "queue_date_raw",
        "prop_date":      "proposed_online_date_raw",
        "on_date":        "actual_online_date_raw",
        "wd_date":        "withdrawal_date_raw",
        "ia_date":        "ia_date_raw",
        "IA_status_clean":"ia_status",
        "region":         "iso_region",
        "type_clean":     "tech_type",
        "mw1":            "capacity_mw",
        "service":        "service_type",
        "cluster":        "cluster_study",
        "q_year":         "queue_year",
        "prop_year":      "proposed_online_year",
        "state":          "state",
        "entity":         "entity",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # ── 2. Convert date serials ──────────────────────────────────────────────
    date_cols = {
        "queue_date_raw":            "queue_date",
        "proposed_online_date_raw":  "proposed_online_date",
        "actual_online_date_raw":    "actual_online_date",
        "withdrawal_date_raw":       "withdrawal_date",
        "ia_date_raw":               "ia_date",
    }
    for raw_col, clean_col in date_cols.items():
        if raw_col in df.columns:
            df[clean_col] = _excel_to_date(pd.to_numeric(df[raw_col], errors="coerce"))

    # ── 3. Clean status ──────────────────────────────────────────────────────
    df["status"] = df["status"].str.strip().str.lower()

    # ── 4. Clean capacity ───────────────────────────────────────────────────
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")

    # ── 5. Derived columns on full dataset ──────────────────────────────────
    df["queue_year"] = pd.to_numeric(df["queue_year"], errors="coerce")
    df["is_hybrid"] = df["tech_type"].str.contains(
        r"\+", na=False
    ).astype(int)

    print(f"  Raw shape: {df.shape}")
    print(f"  Status counts:\n{df['status'].value_counts().to_string()}\n")

    # ── 6. Build modelling subset ────────────────────────────────────────────
    df_model = _build_model_df(df)
    print(f"  Modelling subset shape: {df_model.shape}")
    return df, df_model


def _build_model_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subset to completed ('operational') and withdrawn projects only.
    Compute target variables. Enforce basic data quality filters.
    """
    mask = df["status"].isin(["operational", "withdrawn"])
    dm = df[mask].copy()

    # ── Classification target ────────────────────────────────────────────────
    dm["will_complete"] = (dm["status"] == "operational").astype(int)

    # ── Regression target: queue duration ───────────────────────────────────
    # For completed projects: queue_date → actual_online_date
    # For withdrawn projects: queue_date → withdrawal_date
    dm["end_date"] = dm["actual_online_date"].where(
        dm["status"] == "operational", dm["withdrawal_date"]
    )
    dm["queue_duration_months"] = (
        (dm["end_date"] - dm["queue_date"]).dt.days / 30.4375
    )

    # ── Quality filters ──────────────────────────────────────────────────────
    # Must have valid queue entry date
    dm = dm[dm["queue_date"].notna()]
    # Must have positive capacity
    dm = dm[dm["capacity_mw"].gt(0)]
    # Duration must be positive (sanity: some data entry errors exist)
    dm = dm[dm["queue_duration_months"].gt(0) | dm["queue_duration_months"].isna()]
    # Cap extreme outliers (>25 years is almost certainly data error)
    dm = dm[dm["queue_duration_months"].lt(300) | dm["queue_duration_months"].isna()]

    # ── Queue year (fallback if q_year missing) ──────────────────────────────
    dm["queue_year"] = dm["queue_year"].fillna(dm["queue_date"].dt.year)

    return dm.reset_index(drop=True)


# ── Quick diagnostics ────────────────────────────────────────────────────────
def print_summary(df_raw: pd.DataFrame, df_model: pd.DataFrame) -> None:
    print("=" * 55)
    print("DATASET SUMMARY")
    print("=" * 55)

    print(f"\n{'Raw records:':<35} {len(df_raw):>7,}")
    print(f"{'Modelling records:':<35} {len(df_model):>7,}")

    print(f"\n{'Completed (operational):':<35} "
          f"{df_model['will_complete'].sum():>7,} "
          f"({df_model['will_complete'].mean()*100:.1f}%)")
    print(f"{'Withdrawn:':<35} "
          f"{(1-df_model['will_complete']).sum():>7,} "
          f"({(1-df_model['will_complete']).mean()*100:.1f}%)")

    print(f"\n{'Queue year range:':<35} "
          f"{int(df_model['queue_year'].min())} – {int(df_model['queue_year'].max())}")
    print(f"{'Median capacity (MW):':<35} {df_model['capacity_mw'].median():>7.0f}")

    dur = df_model.loc[df_model["will_complete"]==1, "queue_duration_months"]
    print(f"\n{'Median duration (completed, months):':<35} {dur.median():>7.1f}")
    print(f"{'Mean duration  (completed, months):':<35} {dur.mean():>7.1f}")

    print(f"\nRegion breakdown:")
    print(df_model["iso_region"].value_counts().to_string())

    print(f"\nTop tech types:")
    print(df_model["tech_type"].value_counts().head(10).to_string())

    print(f"\nMissing % in modelling set:")
    miss = (df_model.isnull().mean() * 100).round(1)
    print(miss[miss > 0].sort_values(ascending=False).to_string())
    print("=" * 55)


# ── CLI convenience ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    fpath = sys.argv[1] if len(sys.argv) > 1 else "data/raw/lbnl_ix_queue_data_file_thru2024.xlsx"
    df_raw, df_model = load_queue_data(fpath)
    print_summary(df_raw, df_model)
