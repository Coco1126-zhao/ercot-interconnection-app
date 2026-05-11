"""
ercot/predictor.py
==================
Layer 1 — wraps the trained LBNL XGBoost classifier + regressor for
single-project predictions, with per-instance SHAP attribution.

Maps a user's ERCOT-flavoured input to the LBNL feature schema. This
is necessary because the LBNL training set uses pan-ISO concepts
(NRIS/ERIS service type, multi-region rolling backlog) that ERCOT
inputs don't carry directly.

EXPLICIT MAPPING ASSUMPTIONS — surfaced in the UI as caveats:
  - service_type defaults to "NRIS+ERIS" (the highest-completion modal
    value in LBNL — best-case substitute for ERCOT's GIM concept).
  - is_slow_iso = 0 (ERCOT is not PJM/MISO).
  - log_queue_backlog uses the *current ERCOT queue size*, not the
    LBNL 3-yr rolling backlog, so that the prediction reflects today's
    congestion rather than the historical training-time average.
  - is_nris = 1 (NRIS+ERIS modal default).
"""
from __future__ import annotations
import os, joblib
import numpy as np
import pandas as pd
import shap
from datetime import datetime
from pathlib import Path

ARTIFACTS_DIR = Path("data/artifacts")

_FUEL_TO_TECH_BUCKET = {
    "Solar":   "Solar",
    "Wind":    "Wind",
    "Battery": "Battery",
    "Gas":     "Gas",
    "Coal":    "Fossil",
    "Oil":     "Fossil",
    "Nuclear": "Nuclear",
    "Hydro":   "Hydro",
    "Biomass": "Other",
    "Other":   "Other",
}


def _capacity_bucket(mw: float) -> str:
    if mw <= 50:    return "small"
    if mw <= 200:   return "mid"
    if mw <= 500:   return "large"
    return "utility"


class Predictor:
    """Loads pickled artifacts and predicts on user inputs."""

    def __init__(self, artifacts_dir: str | Path = ARTIFACTS_DIR):
        d = Path(artifacts_dir)
        self.clf_pre   = joblib.load(d / "clf_preprocessor.pkl")
        self.clf       = joblib.load(d / "clf.pkl")
        self.reg_pre   = joblib.load(d / "reg_preprocessor.pkl")
        self.reg       = joblib.load(d / "reg.pkl")
        self.clf_explainer = joblib.load(d / "clf_explainer.pkl")
        self.reg_explainer = joblib.load(d / "reg_explainer.pkl")
        self.calibration   = joblib.load(d / "calibration.pkl")

    # ── Map user input → LBNL feature row ────────────────────────────
    def build_feature_row(self, ui: dict, current_queue_size: int | None = None) -> pd.DataFrame:
        """
        Convert a user's plain-English input dict into a single-row
        DataFrame matching the LBNL model schema.

        Parameters
        ----------
        ui : dict with keys
            capacity_mw, fuel_label, county, cdr_zone, projected_cod,
            queue_entry_date (optional → today),
            is_hybrid (optional → 0), service_type (optional)
        current_queue_size : int | None
            Number of currently active ERCOT queue projects — used to
            compute log_queue_backlog. If None, falls back to LBNL ERCOT
            historical median.
        """
        cap        = float(ui.get("capacity_mw", 0) or 0)
        fuel       = ui.get("fuel_label", "Other")
        is_hybrid  = int(ui.get("is_hybrid", 0))
        service    = ui.get("service_type", "NRIS+ERIS")
        county     = ui.get("county", "")
        cdr_zone   = str(ui.get("cdr_zone", "")).upper()

        queue_dt = pd.to_datetime(ui.get("queue_entry_date", datetime.today()))
        cod_dt   = pd.to_datetime(ui.get("projected_cod", None), errors="coerce")
        lead_yr  = (cod_dt.year - queue_dt.year) if pd.notna(cod_dt) else np.nan
        if pd.notna(lead_yr):
            lead_yr = float(np.clip(lead_yr, 0, 20))

        tech_bucket = "Hybrid" if is_hybrid else _FUEL_TO_TECH_BUCKET.get(fuel, "Other")

        backlog = current_queue_size
        if backlog is None or backlog <= 0:
            backlog = self.calibration.get("ercot_lbnl_median_backlog", 1000)

        row = {
            # Numeric
            "log_capacity_mw":     np.log1p(cap),
            "log_queue_backlog":   np.log1p(backlog),
            "proposed_lead_years": lead_yr if pd.notna(lead_yr) else np.nan,
            "queue_month":         queue_dt.month,
            "queue_quarter":       queue_dt.quarter,
            "queue_year":          queue_dt.year,
            "is_hybrid":           is_hybrid,
            "is_slow_iso":         0,                                # ERCOT
            "is_nris":             1 if "NRIS" in service.upper() else 0,
            "has_proposed_date":   int(pd.notna(cod_dt)),
            "post_ferc_2003":      int(queue_dt.year >= 2004),
            "post_ferc_2023":      int(queue_dt.year >= 2023),
            "itc_active":          int(((queue_dt.year >= 2006) & (queue_dt.year <= 2016))
                                       or (queue_dt.year >= 2022)),
            "ptc_bonus_period":    int(((queue_dt.year >= 2009) & (queue_dt.year <= 2013))
                                       or ((queue_dt.year >= 2015) & (queue_dt.year <= 2021))),
            "ira_era":             int(queue_dt.year >= 2022),
            # Categorical
            "iso_region":      "ERCOT",
            "tech_bucket":     tech_bucket,
            "capacity_bucket": _capacity_bucket(cap),
            "service_type":    service,
            "queue_decade":    str((queue_dt.year // 10) * 10),
        }
        return pd.DataFrame([row])

    # ── Predict + SHAP factors ───────────────────────────────────────
    def predict(self, ui: dict, current_queue_size: int | None = None,
                top_k: int = 5) -> dict:
        x = self.build_feature_row(ui, current_queue_size)

        x_clf = self.clf_pre.transform(x)
        x_reg = self.reg_pre.transform(x)

        prob_complete = float(self.clf.predict_proba(x_clf)[0, 1])
        duration_mo   = float(self.reg.predict(x_reg)[0])

        # SHAP — per-instance attribution
        sv_clf = self.clf_explainer.shap_values(x_clf)
        sv_reg = self.reg_explainer.shap_values(x_reg)

        clf_factors = self._top_factors(self.clf_pre, sv_clf[0], top_k)
        reg_factors = self._top_factors(self.reg_pre, sv_reg[0], top_k)

        return {
            "prob_complete":      prob_complete,
            "duration_months":    duration_mo,
            "clf_top_positive":   clf_factors["positive"],
            "clf_top_negative":   clf_factors["negative"],
            "reg_top_speedup":    reg_factors["negative"],   # negative SHAP = shorter
            "reg_top_slowdown":   reg_factors["positive"],   # positive SHAP = longer
            "feature_row":        x.iloc[0].to_dict(),
            "assumptions": {
                "service_type":     ui.get("service_type", "NRIS+ERIS"),
                "is_slow_iso":      0,
                "queue_backlog":    current_queue_size or
                                    self.calibration.get("ercot_lbnl_median_backlog"),
                "note": ("ERCOT inputs were mapped to LBNL schema. "
                         "Pre-2019 ERCOT bias in training data — read predictions "
                         "as a baseline; consult Layer 2 (similar projects) for "
                         "a live-queue reality check."),
            },
        }

    def _top_factors(self, preprocessor, shap_vals: np.ndarray, k: int):
        """Return top-k positive / negative SHAP contributions with names."""
        from src.features import get_feature_names
        num_features, cat_features = get_feature_names()
        ohe = preprocessor.named_transformers_["cat"]["ohe"]
        cat_names = list(ohe.get_feature_names_out(cat_features))
        names = num_features + cat_names

        n = min(len(names), len(shap_vals))
        s = pd.Series(shap_vals[:n], index=names[:n])

        pos = s.nlargest(k)
        neg = s.nsmallest(k)
        return {
            "positive": [(name, float(val)) for name, val in pos.items()],
            "negative": [(name, float(val)) for name, val in neg.items()],
        }


if __name__ == "__main__":
    p = Predictor()
    out = p.predict({
        "capacity_mw":   200,
        "fuel_label":    "Battery",
        "county":        "Crockett",
        "cdr_zone":      "WEST",
        "projected_cod": "2027-12-01",
        "is_hybrid":     0,
    }, current_queue_size=1844)
    print(f"P(complete): {out['prob_complete']:.1%}")
    print(f"Expected duration: {out['duration_months']:.1f} months")
    print("Top positive (completion):", out["clf_top_positive"][:3])
    print("Top negative (completion):", out["clf_top_negative"][:3])
