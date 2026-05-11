"""
features.py
===========
Feature engineering for the Project Viability & Speed-to-Market Predictor.
All features use only information observable at or before queue entry date
to prevent data leakage into the predictive pipeline.

Usage
-----
    from src.features import build_features
    df_feat = build_features(df_model, df_raw)
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ── 1. Feature construction ──────────────────────────────────────────────────

def build_features(df_model: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construct all model-ready features from the cleaned modelling DataFrame.

    Parameters
    ----------
    df_model : pd.DataFrame
        Output of data_loader.load_queue_data() – completed + withdrawn rows.
    df_raw : pd.DataFrame
        Full raw dataset (all statuses) – used to compute queue backlog.

    Returns
    -------
    pd.DataFrame
        df_model with all engineered feature columns appended.
        Target columns (will_complete, queue_duration_months) are preserved.
    """
    df = df_model.copy()

    df = _capacity_features(df)
    df = _tech_type_features(df)
    df = _temporal_features(df)
    df = _iso_features(df)
    df = _service_features(df)
    df = _developer_lead_time(df)
    df = _queue_backlog(df, df_raw)
    df = _policy_features(df)

    print(f"Feature engineering complete. Shape: {df.shape}")
    print(f"New feature columns: {_feature_cols()}")
    return df


# ── 2. Individual feature groups ─────────────────────────────────────────────

def _capacity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform capacity (MW is heavily right-skewed)."""
    df["log_capacity_mw"] = np.log1p(df["capacity_mw"])

    # Size bucket: small (<50 MW), mid (50–200), large (200–500), utility (>500)
    df["capacity_bucket"] = pd.cut(
        df["capacity_mw"],
        bins=[0, 50, 200, 500, np.inf],
        labels=["small", "mid", "large", "utility"],
        right=True,
    ).astype(str)
    return df


def _tech_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and group technology types.
    Hybrids (Solar+Battery etc.) are grouped as 'Hybrid' to avoid
    long-tail sparsity, while keeping the is_hybrid binary flag.
    """
    # is_hybrid already computed in data_loader; recompute defensively
    df["is_hybrid"] = df["tech_type"].str.contains(r"\+", na=False).astype(int)

    # Consolidated tech bucket for modelling
    _TECH_MAP = {
        "Solar":          "Solar",
        "Wind":           "Wind",
        "Offshore Wind":  "Offshore Wind",
        "Battery":        "Battery",
        "Gas":            "Gas",
        "Coal":           "Fossil",
        "Gas+Oil":        "Fossil",
        "Oil":            "Fossil",
        "Diesel":         "Fossil",
        "Nuclear":        "Nuclear",
        "Hydro":          "Hydro",
        "Geothermal":     "Other",
        "Other":          "Other",
    }
    mapped = df["tech_type"].map(_TECH_MAP)
    fallback = np.where(df["is_hybrid"] == 1, "Hybrid", "Other")
    df["tech_bucket"] = mapped.where(mapped.notna(), other=pd.Series(fallback, index=df.index))
    return df


def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Year, month, and quarter of queue entry."""
    df["queue_month"]   = df["queue_date"].dt.month
    df["queue_quarter"] = df["queue_date"].dt.quarter

    # Decade cohort: captures structural shifts in queue dynamics
    df["queue_decade"] = (df["queue_year"] // 10 * 10).astype("Int64").astype(str)

    # Regulatory reform era flags
    # FERC Order 2003 (serial studies mandatory) – effective 2004
    df["post_ferc_2003"] = (df["queue_year"] >= 2004).astype(int)
    # FERC Order 2023 (cluster studies reform) – effective late 2023
    df["post_ferc_2023"] = (df["queue_year"] >= 2023).astype(int)

    return df


def _iso_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Region / ISO dummy. Already present as iso_region; kept as-is for
    one-hot encoding in the preprocessor. Here we add a binary flag for
    historically slow ISOs (PJM, MISO) based on domain knowledge.
    """
    _SLOW_ISOS = {"PJM", "MISO"}
    df["is_slow_iso"] = df["iso_region"].isin(_SLOW_ISOS).astype(int)
    return df


def _service_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Network Resource Interconnection Service (NRIS) vs Energy Resource
    Interconnection Service (ERIS). NRIS requires full network study
    → typically longer queue duration.
    """
    df["is_nris"] = (
        df["service_type"].str.upper().str.contains("NRIS", na=False)
    ).astype(int)
    return df


def _developer_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proposed lead time = proposed online year – queue year.
    Developers who plan further out signal more serious project intent.
    Capped at 20 years; negative values (data errors) set to NaN.
    """
    df["proposed_lead_years"] = df["proposed_online_year"] - df["queue_year"]
    df["proposed_lead_years"] = df["proposed_lead_years"].clip(lower=0, upper=20)
    df["has_proposed_date"]   = df["proposed_online_year"].notna().astype(int)
    return df


def _queue_backlog(df: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Queue backlog at entry: number of projects that entered the same ISO
    region in the 3-year window ending on this project's queue date.

    This is a leading congestion indicator — a high backlog predicts
    longer wait times and higher withdrawal probability.

    Vectorized implementation: sort all entries by ISO + date, then use
    a merge_asof-style rolling count per ISO group.
    """
    from src.data_loader import _excel_to_date  # noqa: keep for potential future use

    # df_raw uses renamed columns from data_loader
    ref = df_raw[["iso_region", "queue_date"]].dropna(subset=["queue_date", "iso_region"]).copy()
    ref = ref.dropna(subset=["queue_date", "iso_region"])
    ref = ref.sort_values("queue_date").reset_index(drop=True)

    WINDOW_DAYS = 3 * 365

    print("  Computing queue backlog (vectorised 3-year rolling window per ISO)...")
    results = {}
    for iso, grp in ref.groupby("iso_region"):
        dates = grp["queue_date"].values.astype("datetime64[D]").astype(np.int64)
        counts = np.searchsorted(dates, dates, side="right") - \
                 np.searchsorted(dates, dates - WINDOW_DAYS, side="left")
        results[iso] = pd.Series(counts, index=grp.index)

    ref["backlog"] = pd.concat(results.values())

    # Map back to df by (iso_region, queue_date) — exact match via merge
    target = df[["iso_region", "queue_date"]].copy()
    target["_idx"] = target.index
    ref_dedup = ref[["iso_region", "queue_date", "backlog"]].drop_duplicates(
        subset=["iso_region", "queue_date"]
    )
    merged = target.merge(ref_dedup, on=["iso_region", "queue_date"], how="left")
    merged = merged.set_index("_idx")["backlog"].reindex(df.index)

    df["queue_backlog_3yr"] = merged
    df["log_queue_backlog"] = np.log1p(df["queue_backlog_3yr"])
    return df


def _policy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flags for major US federal energy policy periods.
    ITC/PTC significantly affect project economics and whether developers
    persist through a long queue or withdraw.

    ITC active at 30%: 2006–2016, then restored at 30% 2022+ (IRA)
    PTC active (wind): continuously, but bonus periods 2009–2013, 2015–2021
    """
    yr = df["queue_year"]

    df["itc_active"] = (
        ((yr >= 2006) & (yr <= 2016)) | (yr >= 2022)
    ).astype(int)

    df["ptc_bonus_period"] = (
        ((yr >= 2009) & (yr <= 2013)) | ((yr >= 2015) & (yr <= 2021))
    ).astype(int)

    # IRA era (Inflation Reduction Act, signed Aug 2022) — major tailwind
    df["ira_era"] = (yr >= 2022).astype(int)

    return df


# ── 3. Feature column manifest ───────────────────────────────────────────────

def _feature_cols() -> dict:
    return {
        "numeric": [
            "log_capacity_mw",
            "log_queue_backlog",
            "proposed_lead_years",
            "queue_month",
            "queue_quarter",
            "queue_year",
            "is_hybrid",
            "is_slow_iso",
            "is_nris",
            "has_proposed_date",
            "post_ferc_2003",
            "post_ferc_2023",
            "itc_active",
            "ptc_bonus_period",
            "ira_era",
        ],
        "categorical": [
            "iso_region",
            "tech_bucket",
            "capacity_bucket",
            "service_type",
            "queue_decade",
        ],
    }


def get_feature_names() -> tuple[list, list]:
    """Return (numeric_features, categorical_features) lists."""
    cols = _feature_cols()
    return cols["numeric"], cols["categorical"]


# ── 4. Sklearn preprocessor ──────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Returns a fitted-ready sklearn ColumnTransformer:
      - Numeric: median imputation → StandardScaler
      - Categorical: most-frequent imputation → OneHotEncoder (drop first)
    """
    numeric_features, categorical_features = get_feature_names()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


# ── 5. Train/val/test split (time-aware) ─────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_cutoff: int = 2019,
    val_cutoff:   int = 2022,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split to prevent temporal leakage.

    Train : queue_year < train_cutoff        (pre-2019)
    Val   : train_cutoff ≤ queue_year < val_cutoff (2019–2021)
    Test  : queue_year ≥ val_cutoff          (2022+)

    Parameters
    ----------
    df            : feature-engineered DataFrame
    train_cutoff  : first year excluded from training set
    val_cutoff    : first year of test set

    Returns
    -------
    train, val, test DataFrames
    """
    yr = df["queue_year"]
    train = df[yr < train_cutoff].copy()
    val   = df[(yr >= train_cutoff) & (yr < val_cutoff)].copy()
    test  = df[yr >= val_cutoff].copy()

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n = len(split)
        comp_rate = split["will_complete"].mean() * 100 if n > 0 else 0
        print(f"  {name:<6}: {n:>6,} rows | "
              f"completion rate {comp_rate:.1f}% | "
              f"years {int(split['queue_year'].min())}–{int(split['queue_year'].max())}")

    return train, val, test


# ── CLI convenience ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from src.data_loader import load_queue_data

    fpath = sys.argv[1] if len(sys.argv) > 1 else "data/raw/lbnl_ix_queue_data_file_thru2024.xlsx"
    df_raw, df_model = load_queue_data(fpath)

    print("\nBuilding features...")
    df_feat = build_features(df_model, df_raw)

    print("\nTime-aware split:")
    train, val, test = time_split(df_feat)

    print(f"\nSample feature row:\n{df_feat[_feature_cols()['numeric']].head(3).to_string()}")
