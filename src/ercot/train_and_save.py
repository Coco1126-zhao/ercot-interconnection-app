"""
ercot/train_and_save.py
=======================
One-time training step for the LBNL-based two-model system.
Pickles preprocessor + classifier + regressor + SHAP explainers
to data/artifacts/ so the Streamlit app can serve predictions
without retraining on every launch.

NEW FILE — does not modify the existing model code in src/.

Run from project root:
    python -m src.ercot.train_and_save
"""
from __future__ import annotations
import os, sys, joblib
from pathlib import Path

# Make `src.X` imports work when run as `python -m`
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import shap                                          # noqa: E402
from src.data_loader import load_queue_data          # noqa: E402
from src.features    import build_features           # noqa: E402
from src.models      import train_classifier, train_regressor  # noqa: E402
from src.features    import time_split               # noqa: E402

LBNL_PATH      = ROOT / "data" / "raw" / "LBNL_Ix_Queue_Data_File_thru2024.xlsx"
ARTIFACTS_DIR  = ROOT / "data" / "artifacts"


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading LBNL: {LBNL_PATH}")
    df_raw, df_model = load_queue_data(LBNL_PATH)

    print("\nBuilding features...")
    df_feat = build_features(df_model, df_raw)

    print("\nTime split:")
    train, val, _ = time_split(df_feat)

    clf_pre, clf = train_classifier(train, val)
    reg_pre, reg = train_regressor(train, val)

    print("\nFitting SHAP TreeExplainers...")
    clf_explainer = shap.TreeExplainer(clf)
    reg_explainer = shap.TreeExplainer(reg)

    # ERCOT historical median backlog (used as fallback in Predictor)
    ercot_rows = df_feat[df_feat["iso_region"] == "ERCOT"]
    ercot_backlog_median = float(ercot_rows["queue_backlog_3yr"].median()) \
        if len(ercot_rows) else 1000.0

    calibration = {
        "ercot_lbnl_median_backlog": ercot_backlog_median,
        "lbnl_train_years":          (int(train["queue_year"].min()),
                                      int(train["queue_year"].max())),
        "completion_rate_train":     float(train["will_complete"].mean()),
    }

    print(f"\nSaving artifacts to {ARTIFACTS_DIR}/")
    joblib.dump(clf_pre,        ARTIFACTS_DIR / "clf_preprocessor.pkl")
    joblib.dump(clf,            ARTIFACTS_DIR / "clf.pkl")
    joblib.dump(reg_pre,        ARTIFACTS_DIR / "reg_preprocessor.pkl")
    joblib.dump(reg,            ARTIFACTS_DIR / "reg.pkl")
    joblib.dump(clf_explainer,  ARTIFACTS_DIR / "clf_explainer.pkl")
    joblib.dump(reg_explainer,  ARTIFACTS_DIR / "reg_explainer.pkl")
    joblib.dump(calibration,    ARTIFACTS_DIR / "calibration.pkl")

    print("\nDone. Artifacts:")
    for f in sorted(ARTIFACTS_DIR.glob("*.pkl")):
        print(f"  {f.name:30s} {f.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
