"""
ercot/similarity.py
===================
Layer 2 — find the most similar projects already in the ERCOT queue
to a user's submitted project, and return their *actual* milestone
timeline as a reference.

Why a separate model: the LBNL training data ends 2024 and ERCOT
recent dynamics are underrepresented. Matching against the live
ERCOT queue gives a reality-check timeline that the LBNL regressor
cannot provide.

Method: KNN with a Gower-style mixed distance.
  - Numeric: log_capacity_mw, vintage_year, projected_lead_years
    → standardized Euclidean
  - Categorical: fuel_label, CDR Reporting Zone
    → mismatch distance (0 if equal, 1 if different)
  - County: bonus weight if same county (treated as categorical)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime


_NUMERIC = ["log_capacity_mw", "vintage_year", "projected_lead_years"]
_CATEGORICAL = ["fuel_label", "CDR Reporting Zone", "County"]
_CAT_WEIGHTS = {"fuel_label": 2.0, "CDR Reporting Zone": 1.0, "County": 1.5}


def _prep_row(d: dict) -> dict:
    """Build numeric + categorical fields from raw input dict."""
    cap = float(d.get("capacity_mw", 0) or 0)
    cod = pd.to_datetime(d.get("projected_cod"), errors="coerce")
    queue = pd.to_datetime(d.get("queue_entry_date", datetime.today()), errors="coerce")
    if pd.isna(queue):
        queue = pd.Timestamp.today()
    lead = (cod - queue).days / 365.25 if pd.notna(cod) else np.nan
    return {
        "log_capacity_mw":      np.log1p(max(cap, 0)),
        "vintage_year":         queue.year,
        "projected_lead_years": lead if pd.notna(lead) else 3.0,  # default
        "fuel_label":           str(d.get("fuel_label", "")).strip(),
        "CDR Reporting Zone":   str(d.get("cdr_zone", "")).strip().upper(),
        "County":               str(d.get("county", "")).strip().title(),
    }


def _frame_to_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["log_capacity_mw"]      = np.log1p(df["capacity_mw"].fillna(0).clip(lower=0))
    f["vintage_year"]         = df["queue_entry_date"].dt.year.fillna(
        df["queue_entry_date"].dt.year.median()
    )
    cod = pd.to_datetime(df["Projected COD"], errors="coerce")
    lead = ((cod - df["queue_entry_date"]).dt.days / 365.25)
    f["projected_lead_years"] = lead.fillna(lead.median() if not lead.isna().all() else 3.0)
    f["fuel_label"]            = df["fuel_label"].astype(str)
    f["CDR Reporting Zone"]    = df["CDR Reporting Zone"].astype(str)
    f["County"]                = df["County"].astype(str)
    return f


class SimilarityEngine:
    def __init__(self, ercot_df: pd.DataFrame):
        # Restrict to projects with usable queue_entry_date
        df = ercot_df[ercot_df["queue_entry_date"].notna()].copy()
        df = df[df["capacity_mw"].fillna(0) > 0].copy()
        self.df = df.reset_index(drop=True)
        self.feat = _frame_to_features(self.df)

        # Standardize numeric for euclidean distance
        self._num_means = self.feat[_NUMERIC].mean()
        self._num_stds  = self.feat[_NUMERIC].std().replace(0, 1.0)

    def _distance(self, q: dict) -> np.ndarray:
        # Numeric distance
        q_num = np.array([q[c] for c in _NUMERIC], dtype=float)
        z_q = (q_num - self._num_means.values) / self._num_stds.values
        z_db = ((self.feat[_NUMERIC].values - self._num_means.values)
                / self._num_stds.values)
        d_num = np.sqrt(((z_db - z_q) ** 2).sum(axis=1))

        # Categorical mismatch distance (weighted)
        d_cat = np.zeros(len(self.feat))
        for c in _CATEGORICAL:
            mismatch = (self.feat[c].astype(str).str.upper() !=
                        str(q[c]).upper()).astype(float)
            d_cat += _CAT_WEIGHTS[c] * mismatch

        return d_num + d_cat

    def find_similar(self, user_input: dict, n: int = 5) -> pd.DataFrame:
        q = _prep_row(user_input)
        d = self._distance(q)
        idx = np.argsort(d)[:n]
        out = self.df.iloc[idx].copy()
        out["similarity_distance"] = d[idx].round(3)
        # Surface useful display columns up front
        cols = [
            "INR", "Project Name", "fuel_label", "capacity_mw",
            "County", "CDR Reporting Zone",
            "queue_entry_date", "Projected COD",
            "current_stage", "progress_pct", "months_in_queue",
            "similarity_distance",
        ]
        return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


if __name__ == "__main__":
    from src.ercot.loader import load_gis_large_gen
    df = load_gis_large_gen("data/ercot/latest.xlsx")
    eng = SimilarityEngine(df)
    sample = {
        "capacity_mw": 200,
        "fuel_label":  "Battery",
        "cdr_zone":    "WEST",
        "county":      "Crockett",
        "projected_cod": "2027-12-01",
    }
    print(eng.find_similar(sample, n=5).to_string())
