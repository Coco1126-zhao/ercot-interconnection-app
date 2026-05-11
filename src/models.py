"""
models.py
=========
Trains and evaluates the two-model Project Viability & Speed-to-Market system:

  Model 1 — XGBoost Classifier  : P(project reaches commercial operation)
  Model 2 — XGBoost Regressor   : Expected queue duration (months) for completed projects

Both models use time-aware train/val/test splits to prevent temporal leakage.
Outputs trained model objects, evaluation metrics, and feature importance DataFrames
ready for SHAP analysis in the next step.

Usage
-----
    from src.models import evaluate_models
    clf, reg, metrics = evaluate_models(df_feat)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report,
    mean_absolute_error, r2_score,
)
from sklearn.metrics import root_mean_squared_error

from src.features import build_preprocessor, get_feature_names, time_split


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_XY(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target y, dropping rows with missing target."""
    numeric, categorical = get_feature_names()
    feature_cols = numeric + categorical
    mask = df[target].notna()
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, target]
    return X, y


def _imbalance_weight(y_train: pd.Series) -> float:
    """Compute scale_pos_weight = n_negative / n_positive."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    ratio = n_neg / n_pos
    print(f"  Class balance  withdrawn: {n_neg:,} | completed: {n_pos:,} "
          f"| scale_pos_weight: {ratio:.2f}")
    return ratio


# ── Model 1: Classifier ───────────────────────────────────────────────────────

def train_classifier(train: pd.DataFrame, val: pd.DataFrame) -> tuple:
    """
    Train XGBoost classifier to predict will_complete (0/1).

    Handles class imbalance via scale_pos_weight = n_neg / n_pos.
    Uses early stopping on validation AUC-PR to prevent overfitting.

    Returns
    -------
    preprocessor : fitted ColumnTransformer
    clf          : fitted XGBClassifier
    """
    print("\n── Model 1: Completion Probability Classifier ──────────────────")

    X_train, y_train = _get_XY(train, "will_complete")
    X_val,   y_val   = _get_XY(val,   "will_complete")

    spw = _imbalance_weight(y_train)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t   = preprocessor.transform(X_val)

    clf = XGBClassifier(
        n_estimators          = 1000,
        learning_rate         = 0.05,
        max_depth             = 5,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 10,
        scale_pos_weight      = spw,
        eval_metric           = "aucpr",
        early_stopping_rounds = 50,
        random_state          = 42,
        verbosity             = 0,
    )

    clf.fit(
        X_train_t, y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
    )

    print(f"  Best iteration: {clf.best_iteration} trees")
    return preprocessor, clf


# ── Model 2: Regressor ────────────────────────────────────────────────────────

def train_regressor(train: pd.DataFrame, val: pd.DataFrame) -> tuple:
    """
    Train XGBoost regressor to predict queue_duration_months.
    Trains only on COMPLETED (operational) projects for clean signal.

    Returns
    -------
    preprocessor : fitted ColumnTransformer
    reg          : fitted XGBRegressor
    """
    print("\n── Model 2: Queue Duration Regressor (completed projects) ──────")

    train_c = train[train["status"] == "operational"].copy()
    val_c   = val[val["status"]   == "operational"].copy()

    X_train, y_train = _get_XY(train_c, "queue_duration_months")
    X_val,   y_val   = _get_XY(val_c,   "queue_duration_months")

    print(f"  Training rows (completed with duration): {len(X_train):,}")
    print(f"  Validation rows:                         {len(X_val):,}")
    print(f"  Target range: {y_train.min():.1f} – {y_train.max():.1f} months "
          f"| median: {y_train.median():.1f}")

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t   = preprocessor.transform(X_val)

    reg = XGBRegressor(
        n_estimators          = 1000,
        learning_rate         = 0.05,
        max_depth             = 5,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 10,
        eval_metric           = "rmse",
        early_stopping_rounds = 50,
        random_state          = 42,
        verbosity             = 0,
    )

    reg.fit(
        X_train_t, y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
    )

    print(f"  Best iteration: {reg.best_iteration} trees")
    return preprocessor, reg


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_classifier(
    preprocessor, clf,
    splits: dict[str, pd.DataFrame],
) -> dict:
    """Evaluate classifier on all splits. Returns metrics dict."""
    print("\n── Classifier Evaluation ───────────────────────────────────────")
    results = {}

    for split_name, df_split in splits.items():
        X, y = _get_XY(df_split, "will_complete")
        if len(y) == 0 or y.nunique() < 2:
            print(f"  {split_name:<6}: skipped (no positive class in split)")
            continue

        X_t    = preprocessor.transform(X)
        y_prob = clf.predict_proba(X_t)[:, 1]

        auc_roc = roc_auc_score(y, y_prob)
        auc_pr  = average_precision_score(y, y_prob)
        results[split_name] = {"AUC-ROC": auc_roc, "AUC-PR": auc_pr, "n": len(y)}

        print(f"  {split_name:<6}: AUC-ROC={auc_roc:.3f}  AUC-PR={auc_pr:.3f}  "
              f"n={len(y):,}  pos_rate={y.mean()*100:.1f}%")

    # Detailed report on validation set
    df_val = splits.get("Val")
    if df_val is not None:
        X_v, y_v = _get_XY(df_val, "will_complete")
        X_vt = preprocessor.transform(X_v)
        y_p  = clf.predict(X_vt)
        print(f"\n  Validation classification report:")
        print(classification_report(y_v, y_p, target_names=["Withdrawn", "Completed"]))

    return results


def evaluate_regressor(
    preprocessor, reg,
    splits: dict[str, pd.DataFrame],
) -> dict:
    """Evaluate regressor (completed projects only) across splits."""
    print("\n── Regressor Evaluation ────────────────────────────────────────")

    # Naive baseline: ISO-mean duration from training set
    train_df = splits.get("Train")
    iso_means = (
        train_df[train_df["status"] == "operational"]
        .groupby("iso_region")["queue_duration_months"]
        .mean()
    ) if train_df is not None else {}
    global_median = (
        train_df.loc[train_df["status"] == "operational", "queue_duration_months"].median()
        if train_df is not None else 40.0
    )

    results = {}
    for split_name, df_split in splits.items():
        df_c = df_split[df_split["status"] == "operational"].copy()
        X, y = _get_XY(df_c, "queue_duration_months")
        if len(y) < 10:
            print(f"  {split_name:<6}: skipped (n={len(y)})")
            continue

        X_t    = preprocessor.transform(X)
        y_pred = reg.predict(X_t)

        rmse = root_mean_squared_error(y, y_pred)
        mae  = mean_absolute_error(y, y_pred)
        r2   = r2_score(y, y_pred)

        # ISO-mean baseline — align to same index as y (duration-notna rows)
        y_baseline = df_c.loc[y.index, "iso_region"].map(iso_means).fillna(global_median)
        baseline_rmse = root_mean_squared_error(y, y_baseline)

        results[split_name] = {
            "RMSE": rmse, "MAE": mae, "R2": r2,
            "Baseline_RMSE": baseline_rmse, "n": len(y)
        }
        print(f"  {split_name:<6}: RMSE={rmse:.1f}mo  MAE={mae:.1f}mo  "
              f"R²={r2:.3f}  ISO-baseline_RMSE={baseline_rmse:.1f}mo  n={len(y):,}")

    return results


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(preprocessor, model, model_name: str) -> pd.DataFrame:
    """Extract XGBoost gain-based feature importances with readable names."""
    num_features, cat_features = get_feature_names()
    ohe       = preprocessor.named_transformers_["cat"]["ohe"]
    cat_names = list(ohe.get_feature_names_out(cat_features))
    all_names = num_features + cat_names

    importances = model.feature_importances_
    n = min(len(all_names), len(importances))

    df_imp = pd.DataFrame({
        "feature":    all_names[:n],
        "importance": importances[:n],
        "model":      model_name,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df_imp


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    df_imp: pd.DataFrame,
    model_name: str,
    top_n: int = 15,
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of top-N feature importances."""
    top = df_imp.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"], color="#2E75B6", edgecolor="white")
    ax.set_xlabel("Feature Importance (XGBoost Gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Features — {model_name}", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_duration_residuals(
    preprocessor, reg,
    val: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Actual vs predicted duration scatter for completed projects in val set."""
    df_c = val[val["status"] == "operational"].copy()
    X, y = _get_XY(df_c, "queue_duration_months")
    if len(y) < 5:
        print("  Not enough completed projects in val set for residual plot.")
        return

    X_t    = preprocessor.transform(X)
    y_pred = reg.predict(X_t)

    rmse = root_mean_squared_error(y, y_pred)
    mae  = mean_absolute_error(y, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, y_pred, alpha=0.3, s=18, color="#2E75B6", edgecolors="none")
    lim_max = max(float(y.max()), float(y_pred.max())) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual Queue Duration (months)", fontsize=11)
    ax.set_ylabel("Predicted Queue Duration (months)", fontsize=11)
    ax.set_title(
        "Regressor: Actual vs Predicted Duration\n(Validation set — completed projects)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.05, 0.92, f"RMSE = {rmse:.1f} mo\nMAE  = {mae:.1f} mo",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_completion_rate_by_iso(
    preprocessor, clf,
    df_feat: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Actual vs predicted completion rate by ISO — validates regional calibration."""
    numeric, categorical = get_feature_names()
    X_all = df_feat[numeric + categorical]
    X_t   = preprocessor.transform(X_all)

    df_plot = df_feat[["iso_region", "will_complete"]].copy()
    df_plot["pred_prob"] = clf.predict_proba(X_t)[:, 1]

    summary = (
        df_plot.groupby("iso_region")
        .agg(actual_rate=("will_complete", "mean"),
             predicted_rate=("pred_prob",  "mean"),
             n=("will_complete", "count"))
        .sort_values("actual_rate", ascending=False)
        .reset_index()
    )

    x = np.arange(len(summary))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, summary["actual_rate"],    w, label="Actual",    color="#2E75B6")
    ax.bar(x + w/2, summary["predicted_rate"], w, label="Predicted", color="#70AD47", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["iso_region"], fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Completion Rate", fontsize=11)
    ax.set_title("Actual vs Predicted Completion Rate by ISO Region",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ── Master runner ─────────────────────────────────────────────────────────────

def evaluate_models(df_feat: pd.DataFrame) -> tuple:
    """
    Full training + evaluation pipeline.

    Parameters
    ----------
    df_feat : pd.DataFrame
        Output of features.build_features()

    Returns
    -------
    clf_pre, clf   : fitted classifier preprocessor + model
    reg_pre, reg   : fitted regressor preprocessor + model
    metrics        : dict with 'clf' and 'reg' evaluation results
    """
    print("=" * 60)
    print("PROJECT VIABILITY & SPEED-TO-MARKET — MODEL TRAINING")
    print("=" * 60)

    train, val, test = time_split(df_feat)
    splits = {"Train": train, "Val": val, "Test": test}

    # Train
    clf_pre, clf = train_classifier(train, val)
    reg_pre, reg = train_regressor(train, val)

    # Evaluate
    clf_metrics = evaluate_classifier(clf_pre, clf, splits)
    reg_metrics = evaluate_regressor(reg_pre, reg, splits)

    # Feature importance tables
    print("\n── Top 10 Features — Classifier ────────────────────────────────")
    clf_imp = get_feature_importance(clf_pre, clf, "Completion Probability")
    print(clf_imp.head(10)[["feature", "importance"]].to_string(index=False))

    print("\n── Top 10 Features — Regressor ─────────────────────────────────")
    reg_imp = get_feature_importance(reg_pre, reg, "Queue Duration")
    print(reg_imp.head(10)[["feature", "importance"]].to_string(index=False))

    # Plots
    plot_feature_importance(clf_imp, "Completion Probability Classifier",
                            save_path="outputs/clf_feature_importance.png")
    plot_feature_importance(reg_imp, "Queue Duration Regressor",
                            save_path="outputs/reg_feature_importance.png")
    plot_duration_residuals(reg_pre, reg, val,
                            save_path="outputs/reg_actual_vs_predicted.png")
    plot_completion_rate_by_iso(clf_pre, clf, df_feat,
                                save_path="outputs/clf_completion_rate_by_iso.png")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — run shap_analysis.py for deep insights")
    print("=" * 60)

    return clf_pre, clf, reg_pre, reg, {"clf": clf_metrics, "reg": reg_metrics}


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, ".")
    os.makedirs("outputs", exist_ok=True)

    from src.data_loader import load_queue_data
    from src.features    import build_features

    data_path = sys.argv[1] if len(sys.argv) > 1 \
        else "data/raw/lbnl_ix_queue_data_file_thru2024.xlsx"

    df_raw, df_model  = load_queue_data(data_path)
    df_feat           = build_features(df_model, df_raw)
    clf_pre, clf, reg_pre, reg, metrics = evaluate_models(df_feat)
