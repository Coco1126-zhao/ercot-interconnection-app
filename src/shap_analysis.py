"""
shap_analysis.py
================
SHAP-based feature attribution for the two-model Project Viability system.

Produces 6 publication-ready plots:
  1. Classifier beeswarm summary  — direction + magnitude of each feature
  2. Regressor beeswarm summary   — same for duration model
  3. Classifier dependence plots  — top 3 features (how prob changes as feature varies)
  4. Regressor dependence plots   — top 3 features
  5. Fast-mover profile           — SHAP waterfall for a high-probability, short-duration project
  6. Hard-case profile            — SHAP waterfall for a low-probability project

Usage
-----
    from src.shap_analysis import run_shap_analysis
    run_shap_analysis(clf_pre, clf, reg_pre, reg, train, val, df_feat)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

import shap

from src.features import get_feature_names


# ── Helpers ──────────────────────────────────────────────────────────────────

def _transform(preprocessor, df: pd.DataFrame) -> np.ndarray:
    """Apply fitted preprocessor to a DataFrame of raw features."""
    numeric, categorical = get_feature_names()
    return preprocessor.transform(df[numeric + categorical])


def _feature_names_out(preprocessor) -> list[str]:
    """Reconstruct full feature name list after OHE expansion."""
    numeric, categorical = get_feature_names()
    ohe       = preprocessor.named_transformers_["cat"]["ohe"]
    cat_names = list(ohe.get_feature_names_out(categorical))
    return numeric + cat_names


def _get_shap_df(
    preprocessor,
    model,
    df_source: pd.DataFrame,
    task: str = "classification",
    sample_n: int = 2000,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute SHAP values for a random sample of rows.

    Returns
    -------
    shap_values : np.ndarray  shape (n_samples, n_features)
    X_display   : np.ndarray  shape (n_samples, n_features)  — preprocessed values
    feature_names : list[str]
    """
    numeric, categorical = get_feature_names()
    feat_cols = numeric + categorical

    df = df_source[feat_cols].dropna(subset=numeric).copy()
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)

    X_transformed = preprocessor.transform(df)
    feature_names  = _feature_names_out(preprocessor)

    explainer   = shap.TreeExplainer(model)
    shap_raw    = explainer.shap_values(X_transformed)

    # For binary classifiers shap_raw may be list [neg_class, pos_class]
    if isinstance(shap_raw, list):
        shap_values = shap_raw[1]
    else:
        shap_values = shap_raw

    return shap_values, X_transformed, feature_names


# ── 1. Beeswarm summary ───────────────────────────────────────────────────────

def plot_beeswarm(
    shap_values:   np.ndarray,
    X_display:     np.ndarray,
    feature_names: list[str],
    title:         str,
    top_n:         int  = 15,
    save_path:     str | None = None,
) -> None:
    """
    SHAP beeswarm (summary) plot — shows direction AND magnitude of each feature.
    Each dot is one project; colour = feature value (red=high, blue=low).
    """
    # Reduce to top_n features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-top_n:]

    sv_top   = shap_values[:, top_idx]
    X_top    = X_display[:,  top_idx]
    names_top = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv_top, X_top,
        feature_names=names_top,
        plot_type="dot",
        show=False,
        plot_size=None,
        color_bar=True,
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ── 2. Dependence plots ───────────────────────────────────────────────────────

def plot_dependence_grid(
    shap_values:   np.ndarray,
    X_display:     np.ndarray,
    feature_names: list[str],
    title_prefix:  str,
    top_n:         int  = 3,
    save_path:     str | None = None,
) -> None:
    """
    SHAP dependence plots for the top-N features by mean |SHAP|.
    Shows how the SHAP value (model impact) changes as the feature value varies.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-top_n:][::-1]

    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 5), sharey=False)
    if top_n == 1:
        axes = [axes]

    for ax, idx in zip(axes, top_idx):
        fname = feature_names[idx]
        x_vals = X_display[:, idx]
        s_vals = shap_values[:, idx]

        sc = ax.scatter(x_vals, s_vals, alpha=0.3, s=12,
                        c=s_vals, cmap="coolwarm", edgecolors="none")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(fname, fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.set_title(f"{title_prefix}\n{fname}", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ── 3. Waterfall case studies ─────────────────────────────────────────────────

def _pick_case(
    df_source:     pd.DataFrame,
    clf_pre,
    clf,
    reg_pre,
    reg,
    mode:          str,  # "fast_mover" or "hard_case"
) -> pd.Series:
    """
    Select a single representative project for a waterfall case study.

    fast_mover : high completion probability AND short predicted duration
    hard_case  : low completion probability (clear withdrawal signal)
    """
    numeric, categorical = get_feature_names()
    feat_cols = numeric + categorical
    df = df_source[feat_cols].dropna(subset=numeric).copy()
    df = df.reset_index(drop=True)

    X_t      = clf_pre.transform(df)
    clf_prob = clf.predict_proba(X_t)[:, 1]

    if mode == "fast_mover":
        # Must have proposed date; pick high-prob project
        has_date = df["has_proposed_date"] == 1
        candidates = df[has_date].copy()
        c_prob = clf_prob[has_date.values]

        X_reg = reg_pre.transform(candidates[feat_cols])
        reg_pred = reg.predict(X_reg)

        # Score = high prob + short duration
        score = c_prob - (reg_pred / reg_pred.max()) * 0.3
        best_idx = score.argmax()
        print(f"  Fast-mover: P(complete)={c_prob[best_idx]:.2f}  "
              f"Predicted duration={reg_pred[best_idx]:.1f} mo")
        return candidates.iloc[best_idx]

    else:  # hard_case
        # Low completion probability
        worst_idx = clf_prob.argmin()
        print(f"  Hard-case:  P(complete)={clf_prob[worst_idx]:.2f}")
        return df.iloc[worst_idx]


def plot_waterfall(
    preprocessor,
    model,
    project_row:  pd.Series,
    feature_names: list[str],
    title:        str,
    task:         str  = "classification",
    save_path:    str | None = None,
) -> None:
    """
    SHAP waterfall for a single project — shows which features push the
    prediction up or down from the model's baseline (expected value).
    """
    numeric, categorical = get_feature_names()
    feat_cols = numeric + categorical

    X_single  = project_row[feat_cols].values.reshape(1, -1)
    X_t       = preprocessor.transform(
        pd.DataFrame([project_row[feat_cols]], columns=feat_cols)
    )

    explainer  = shap.TreeExplainer(model)
    shap_raw   = explainer(X_t)

    # For binary classifier take class-1 explanation
    if task == "classification" and len(shap_raw.values.shape) == 3:
        sv   = shap_raw.values[0, :, 1]
        base = shap_raw.base_values[0, 1]
    else:
        sv   = shap_raw.values[0]
        base = shap_raw.base_values[0]

    # Top 12 by |SHAP| for readability
    top_n  = 12
    order  = np.argsort(np.abs(sv))[-top_n:][::-1]
    sv_top = sv[order]
    nm_top = [feature_names[i] for i in order]

    # Manual waterfall bar chart (shap.waterfall_plot has display issues in Colab)
    cumulative = base
    y_pos      = np.arange(len(sv_top))
    colors     = ["#d73027" if v > 0 else "#4575b4" for v in sv_top]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (val, col) in enumerate(zip(sv_top[::-1], colors[::-1])):
        ax.barh(i, val, left=cumulative, color=col, edgecolor="white", height=0.6)
        cumulative += val

    ax.axvline(base, color="gray", linestyle="--", linewidth=0.8, label=f"Base = {base:.2f}")
    ax.axvline(cumulative, color="black", linestyle="-", linewidth=1.2,
               label=f"Prediction = {cumulative:.2f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nm_top[::-1], fontsize=9)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    red_patch  = mpatches.Patch(color="#d73027", label="Increases prediction")
    blue_patch = mpatches.Patch(color="#4575b4", label="Decreases prediction")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc="lower right")

    plt.tight_layout()
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ── 4. Fast-mover profile summary ────────────────────────────────────────────

def plot_fast_mover_profile(
    clf_pre, clf,
    reg_pre, reg,
    df_feat:    pd.DataFrame,
    save_path:  str | None = None,
) -> pd.DataFrame:
    """
    Identify the top-decile 'fast movers': projects with both high
    completion probability AND short predicted duration.

    Produces a bar chart showing the tech type + ISO distribution
    of fast movers vs the rest of the dataset — the headline
    commercial insight for the README.
    """
    numeric, categorical = get_feature_names()
    feat_cols = numeric + categorical

    extra_cols = [c for c in ["iso_region", "tech_bucket", "queue_year"] if c not in feat_cols]
    df = df_feat[feat_cols + extra_cols].copy()
    df = df.dropna(subset=numeric).reset_index(drop=True)

    # Score all projects
    X_clf = clf_pre.transform(df[feat_cols])
    X_reg = reg_pre.transform(df[feat_cols])

    df["p_complete"]      = clf.predict_proba(X_clf)[:, 1]
    df["pred_duration"]   = reg.predict(X_reg)

    # Normalise and combine into viability score (higher = better)
    df["viability_score"] = (
        df["p_complete"] - 0.3 * (df["pred_duration"] / df["pred_duration"].max())
    )

    threshold = df["viability_score"].quantile(0.90)
    df["is_fast_mover"] = df["viability_score"] >= threshold

    # ISO breakdown: fast movers vs rest
    iso_comp = (
        df.groupby("iso_region")["is_fast_mover"]
        .mean()
        .sort_values(ascending=False)
    )

    tech_comp = (
        df.groupby("tech_bucket")["is_fast_mover"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ISO chart
    ax = axes[0]
    ax.bar(iso_comp.index, iso_comp.values, color="#2E75B6", edgecolor="white")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.set_title("Fast-Mover Rate by ISO Region\n(top-decile viability score)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Share of projects in top decile", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)

    # Tech chart
    ax = axes[1]
    ax.bar(tech_comp.index, tech_comp.values, color="#70AD47", edgecolor="white")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.set_title("Fast-Mover Rate by Technology Type\n(top-decile viability score)",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Project Viability Profile: Fast Movers vs Full Pipeline",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()

    print("\n  Fast-mover ISO ranking:")
    print(iso_comp.apply(lambda x: f"{x*100:.1f}%").to_string())
    print("\n  Fast-mover tech ranking:")
    print(tech_comp.apply(lambda x: f"{x*100:.1f}%").to_string())

    return df[["iso_region", "tech_bucket", "queue_year",
               "p_complete", "pred_duration", "viability_score", "is_fast_mover"]]


# ── Master runner ─────────────────────────────────────────────────────────────

def run_shap_analysis(
    clf_pre, clf,
    reg_pre, reg,
    train:   pd.DataFrame,
    val:     pd.DataFrame,
    df_feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run full SHAP analysis pipeline. Generates all 6 output plots.

    Parameters
    ----------
    clf_pre, clf : fitted classifier preprocessor + XGBClassifier
    reg_pre, reg : fitted regressor preprocessor + XGBRegressor
    train        : training split DataFrame
    val          : validation split DataFrame
    df_feat      : full feature-engineered DataFrame (for fast-mover profiling)

    Returns
    -------
    viability_df : DataFrame with per-project scores (p_complete, pred_duration,
                   viability_score, is_fast_mover)
    """
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("SHAP FEATURE ATTRIBUTION ANALYSIS")
    print("=" * 60)

    # ── Compute SHAP values ───────────────────────────────────────────────────
    print("\nComputing classifier SHAP values (sample=2000)...")
    clf_shap, clf_X, feat_names = _get_shap_df(
        clf_pre, clf, train, task="classification", sample_n=2000
    )

    print("Computing regressor SHAP values...")
    train_c = train[train["status"] == "operational"].copy()
    reg_shap, reg_X, _ = _get_shap_df(
        reg_pre, reg, train_c, task="regression", sample_n=min(1500, len(train_c))
    )

    # ── Plot 1 & 2: Beeswarm summaries ───────────────────────────────────────
    print("\n[1/6] Classifier beeswarm summary...")
    plot_beeswarm(
        clf_shap, clf_X, feat_names,
        title="SHAP Summary — Completion Probability Classifier\n"
              "(red = high feature value  |  blue = low feature value)",
        save_path="outputs/shap_clf_beeswarm.png",
    )

    print("[2/6] Regressor beeswarm summary...")
    plot_beeswarm(
        reg_shap, reg_X, feat_names,
        title="SHAP Summary — Queue Duration Regressor\n"
              "(red = high feature value  |  blue = low feature value)",
        save_path="outputs/shap_reg_beeswarm.png",
    )

    # ── Plot 3 & 4: Dependence plots ─────────────────────────────────────────
    print("[3/6] Classifier dependence plots (top 3 features)...")
    plot_dependence_grid(
        clf_shap, clf_X, feat_names,
        title_prefix="Classifier",
        top_n=3,
        save_path="outputs/shap_clf_dependence.png",
    )

    print("[4/6] Regressor dependence plots (top 3 features)...")
    plot_dependence_grid(
        reg_shap, reg_X, feat_names,
        title_prefix="Regressor",
        top_n=3,
        save_path="outputs/shap_reg_dependence.png",
    )

    # ── Plot 5: Fast-mover waterfall ──────────────────────────────────────────
    print("[5/6] Fast-mover waterfall case study...")
    fast_project = _pick_case(val, clf_pre, clf, reg_pre, reg, mode="fast_mover")
    plot_waterfall(
        clf_pre, clf, fast_project, feat_names,
        title="SHAP Waterfall — Fast-Mover Project\n"
              "(high completion probability, short predicted duration)",
        task="classification",
        save_path="outputs/shap_waterfall_fast_mover.png",
    )

    # ── Plot 6: Hard-case waterfall ───────────────────────────────────────────
    print("[6/6] Hard-case waterfall case study...")
    hard_project = _pick_case(val, clf_pre, clf, reg_pre, reg, mode="hard_case")
    plot_waterfall(
        clf_pre, clf, hard_project, feat_names,
        title="SHAP Waterfall — High-Risk Project\n"
              "(low completion probability)",
        task="classification",
        save_path="outputs/shap_waterfall_hard_case.png",
    )

    # ── Fast-mover profile chart ──────────────────────────────────────────────
    print("\nGenerating fast-mover portfolio profile...")
    viability_df = plot_fast_mover_profile(
        clf_pre, clf, reg_pre, reg, df_feat,
        save_path="outputs/fast_mover_profile.png",
    )

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print(f"  7 plots saved to outputs/")
    print("=" * 60)

    return viability_df
