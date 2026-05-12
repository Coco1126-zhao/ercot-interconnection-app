"""
ercot/narratives.py
===================
Maps each LBNL feature (raw or one-hot expanded) to plain-English narrative
text for two contexts:
  - 'completion' — positive SHAP = boosts P(operational)
  - 'duration'   — positive SHAP = lengthens queue time (bad if you want speed)

Each narrative entry has:
  - label       : human title
  - positive    : what this means when SHAP is positive
  - negative    : what this means when SHAP is negative
  - suggest_*   : optional tactical recommendation
  - actionable  : whether the developer can influence this feature

Used by the Streamlit UI's drivers panels to turn SHAP values into useful
guidance instead of opaque feature names.
"""
from __future__ import annotations


# ── COMPLETION narratives (positive SHAP = boosts probability = good) ──
COMPLETION_NUM: dict[str, dict] = {
    "log_capacity_mw": {
        "label": "Project capacity",
        "actionable": True,
        "positive": "Capacity is below the historical median for completed projects, "
                    "which reduces study complexity and capital hurdles.",
        "negative": "Capacity is above the historical median. Large projects trigger "
                    "more extensive network-upgrade studies and face higher capital hurdles.",
        "suggest_negative": "Consider phasing into smaller blocks — e.g., split a 600 MW "
                            "project into 2 × 300 MW. Project clustering will surface live peers at that size.",
    },
    "log_queue_backlog": {
        "label": "Queue congestion at filing",
        "actionable": True,
        "positive": "Lower-than-average queue congestion would help studies move faster.",
        "negative": "ERCOT's queue is at historic highs (~1,800 active large-gen projects). "
                    "Competing for limited transmission-upgrade capacity reduces completion odds.",
        "suggest_negative": "Use project clustering to identify which live peers are still progressing — "
                            "their POIs and zones are the ones with realistic timelines today.",
    },
    "queue_year": {
        "label": "Year of queue entry",
        "actionable": False,
        "positive": "Older vintage filings benefited from a less congested queue environment.",
        "negative": "Recent queue years (2022+) have lower completion rates in the training data. "
                    "This is structural and applies equally to every developer filing today.",
    },
    "proposed_lead_years": {
        "label": "Planned lead time (COD − filing year)",
        "actionable": True,
        "positive": "A reasonable lead time signals committed planning and is associated with "
                    "higher completion rates — developers who plan longer runways file more serious projects.",
        "negative": "An aggressive COD relative to filing date may signal an under-planned timeline, "
                    "which correlates with higher withdrawal rates.",
        "suggest_negative": "Extend the projected COD by 12–18 months. ERCOT's median completed-project "
                            "duration is ~42 months; aligning to that timeline reads as a more serious filing.",
    },
    "is_nris": {
        "label": "Network resource service (NRIS) flag",
        "actionable": True,
        "positive": "Requesting NRIS (vs energy-only ERIS) is the strongest completion signal in the "
                    "LBNL data — it implies committed capital and a real off-take story.",
        "negative": "Energy-only (ERIS) requests historically have lower completion rates because "
                    "they suggest less firm contractual backing.",
        "suggest_negative": "If a PPA or firm off-take is in development, request NRIS at filing. "
                            "Note: ERCOT does not formally use NRIS/ERIS — this flag is mapped from your input.",
    },
    "is_hybrid": {
        "label": "Hybrid (multi-technology) project",
        "actionable": True,
        "positive": "Hybrid configuration aligns with current ERCOT pipeline trends (BESS pairing).",
        "negative": "Standalone projects historically clear studies more often than hybrids.",
        "suggest_negative": "If filing approval is the bottleneck, consider applying for the generator "
                            "and BESS separately — they can be combined commercially after IA.",
    },
    "has_proposed_date": {
        "label": "Proposed COD filed at submission",
        "actionable": True,
        "positive": "Filing with a target COD lifts the viability score — it signals real project intent.",
        "negative": "No proposed COD on file. Speculative filings historically withdraw at higher rates.",
        "suggest_negative": "Set a realistic projected COD at filing, even if it later slips. It's a "
                            "strong commitment signal for both the model and ERCOT planners.",
    },
    "is_slow_iso": {
        "label": "Slow-ISO flag (PJM / MISO)",
        "actionable": False,
        "positive": "Your project is in ERCOT, not PJM/MISO — fewer structural queue bottlenecks at the ISO level.",
        "negative": "Not applicable for ERCOT.",
    },
    "post_ferc_2003": {"label": "Post-FERC Order 2003 era", "actionable": False,
                      "positive": "Filing benefits from post-2004 standardized study procedures.",
                      "negative": "Pre-2004 filing — historical record only."},
    "post_ferc_2023": {"label": "Post-FERC Order 2023 (cluster studies)", "actionable": False,
                      "positive": "Filing falls into the cluster-study reform era.",
                      "negative": "Pre-cluster-study reform era."},
    "itc_active":      {"label": "Investment Tax Credit active", "actionable": False,
                      "positive": "ITC is active at filing year — improves project economics.",
                      "negative": "Outside ITC window — less federal economic support."},
    "ptc_bonus_period":{"label": "PTC bonus period", "actionable": False,
                      "positive": "Filing year falls within a Production Tax Credit bonus period.",
                      "negative": "Outside a PTC bonus period — minor effect for non-wind tech."},
    "ira_era":         {"label": "Post-IRA (2022+) era", "actionable": False,
                      "positive": "Post-IRA filings benefit from federal clean-energy tailwinds.",
                      "negative": "Pre-IRA era — historical record only."},
    "queue_month":     {"label": "Filing month",   "actionable": False,
                      "positive": "Seasonal timing slightly favorable.",
                      "negative": "Seasonal timing slightly unfavorable. Minor effect overall."},
    "queue_quarter":   {"label": "Filing quarter", "actionable": False,
                      "positive": "Quarterly timing slightly favorable.",
                      "negative": "Quarterly timing slightly unfavorable. Minor effect overall."},
}

COMPLETION_CAT: dict[str, dict] = {
    "iso_region":      {"label": "ISO region", "actionable": False,
        "positive": "Your ISO region ({v}) historically has higher-than-average completion rates.",
        "negative": "Your ISO region ({v}) has structural challenges — queue backlog and transmission limits depress completion."},
    "tech_bucket":     {"label": "Technology",  "actionable": False,
        "positive": "{v} projects historically complete more often than the dataset average.",
        "negative": "{v} projects historically complete less often in the LBNL data."},
    "capacity_bucket": {"label": "Capacity tier", "actionable": True,
        "positive": "Your project is in the '{v}' size tier, which clears studies more often.",
        "negative": "Your project is in the '{v}' size tier; larger tiers face longer studies and more upgrade requirements.",
        "suggest_negative": "Consider phasing down a tier — project clustering will surface peers at that size."},
    "service_type":    {"label": "Service type", "actionable": True,
        "positive": "Service type '{v}' is associated with higher completion rates.",
        "negative": "Service type '{v}' is associated with lower completion rates.",
        "suggest_negative": "Upgrade to NRIS+ERIS if a firm off-take is in place."},
    "queue_decade":    {"label": "Queue decade",  "actionable": False,
        "positive": "Filings in the {v}s benefited from less competitive queues.",
        "negative": "Filings in the {v}s face heavy queue competition — structural, not project-specific."},
}


# ── DURATION narratives (positive SHAP = lengthens = bad if you want speed) ──
DURATION_NUM: dict[str, dict] = {
    "log_capacity_mw": {
        "label": "Project capacity",
        "actionable": True,
        "positive": "Larger capacity drives longer studies and more network-upgrade scope, "
                    "which lengthens queue duration.",
        "negative": "Smaller capacity reduces study complexity and shortens expected duration.",
        "suggest_positive": "Phasing into smaller blocks can cut months off study timelines.",
    },
    "log_queue_backlog": {
        "label": "Queue congestion",
        "actionable": True,
        "positive": "High queue congestion at filing extends duration — limited transmission-study throughput.",
        "negative": "Low queue congestion shortens expected duration.",
        "suggest_positive": "Use project clustering to find live peers at less-congested POIs — their "
                            "stage timing is your realistic anchor.",
    },
    "queue_year": {
        "label": "Year of queue entry",
        "actionable": False,
        "positive": "Recent vintage filings face longer durations on average than historical norms.",
        "negative": "Older vintage filings benefited from faster study cycles.",
    },
    "proposed_lead_years": {
        "label": "Planned lead time (COD − filing year)",
        "actionable": True,
        "positive": "Longer planned lead times correlate near-linearly with longer actual queue duration "
                    "(every extra year of lead time ≈ 10–15 extra months of queue duration in the LBNL data).",
        "negative": "Short planned lead time correlates with shorter actual duration when the project does complete.",
        "suggest_positive": "Validate that your projected COD reflects real construction/financing milestones — "
                            "overly conservative leads also predict longer durations.",
    },
    "is_nris": {
        "label": "Network resource service (NRIS) flag",
        "actionable": False,
        "positive": "NRIS requests require fuller network studies, which adds time.",
        "negative": "Energy-only (ERIS) requests skip parts of the network study — shorter duration on average.",
    },
    "is_hybrid": {
        "label": "Hybrid (multi-technology) project",
        "actionable": True,
        "positive": "Hybrid projects historically take ~15 months longer than single-tech projects even when they complete — "
                    "the dominant duration predictor in the regressor (15.6% importance).",
        "negative": "Standalone projects clear studies faster than hybrids.",
        "suggest_positive": "If speed is the priority, consider filing the generator and BESS as separate "
                            "applications; combine commercially after IA.",
    },
    "has_proposed_date": {
        "label": "Proposed COD filed at submission",
        "actionable": False,
        "positive": "Having a proposed COD correlates with longer expected duration (more serious projects "
                    "tend to be larger and more complex).",
        "negative": "No proposed COD — fewer milestones to track.",
    },
    "is_slow_iso":     {"label": "Slow-ISO flag (PJM / MISO)", "actionable": False,
                      "positive": "Not applicable for ERCOT.",
                      "negative": "ERCOT is faster than PJM/MISO at the ISO level."},
    "post_ferc_2003":  {"label": "Post-FERC Order 2003 era", "actionable": False,
                      "positive": "Post-2004 standardization has minor duration effects.",
                      "negative": "Pre-2004 — historical record only."},
    "post_ferc_2023":  {"label": "Post-FERC Order 2023 (cluster studies)", "actionable": False,
                      "positive": "Cluster studies may extend per-project duration but consolidate work.",
                      "negative": "Pre-cluster-study reform era — minor effect."},
    "itc_active":      {"label": "ITC active",        "actionable": False,
                      "positive": "ITC-driven demand can extend duration via crowding effects.",
                      "negative": "Outside ITC window."},
    "ptc_bonus_period":{"label": "PTC bonus period",  "actionable": False,
                      "positive": "PTC-driven wind demand can extend duration via crowding.",
                      "negative": "Outside a PTC bonus period."},
    "ira_era":         {"label": "Post-IRA era",      "actionable": False,
                      "positive": "IRA-era demand surge has extended queue durations.",
                      "negative": "Pre-IRA era — historical record only."},
    "queue_month":     {"label": "Filing month",     "actionable": False,
                      "positive": "Seasonal timing extends duration slightly.",
                      "negative": "Seasonal timing shortens duration slightly."},
    "queue_quarter":   {"label": "Filing quarter",   "actionable": False,
                      "positive": "Quarterly timing extends duration slightly.",
                      "negative": "Quarterly timing shortens duration slightly."},
}

DURATION_CAT: dict[str, dict] = {
    "iso_region":      {"label": "ISO region", "actionable": False,
        "positive": "{v} historically has longer queue durations than the dataset average.",
        "negative": "{v} historically has faster queue durations."},
    "tech_bucket":     {"label": "Technology", "actionable": False,
        "positive": "{v} projects historically have longer queue durations.",
        "negative": "{v} projects historically have shorter queue durations."},
    "capacity_bucket": {"label": "Capacity tier", "actionable": True,
        "positive": "The '{v}' tier faces longer studies on average.",
        "negative": "The '{v}' tier clears studies faster.",
        "suggest_positive": "Phasing down a tier can cut months off the timeline."},
    "service_type":    {"label": "Service type", "actionable": False,
        "positive": "Service type '{v}' is associated with longer durations.",
        "negative": "Service type '{v}' is associated with shorter durations."},
    "queue_decade":    {"label": "Queue decade", "actionable": False,
        "positive": "Filings in the {v}s have faced longer durations.",
        "negative": "Filings in the {v}s saw faster durations."},
}


def _explain(feature_name: str, shap_value: float,
             num_dict: dict, cat_dict: dict,
             positive_is_bad: bool) -> dict:
    """Internal: same logic for completion / duration, parameterised by direction."""
    direction = "positive" if shap_value > 0 else "negative"

    # Numeric / binary direct match
    if feature_name in num_dict:
        n = num_dict[feature_name]
        suggestion = n.get(f"suggest_{direction}")
        return {
            "label":      n["label"],
            "value":      float(shap_value),
            "direction":  direction,
            "narrative":  n.get(direction, ""),
            "actionable": bool(n.get("actionable", False)),
            "suggestion": suggestion,
            "is_bad":     (direction == "positive") if positive_is_bad else (direction == "negative"),
        }

    # Categorical one-hot match
    for base, n in cat_dict.items():
        prefix = base + "_"
        if feature_name.startswith(prefix):
            value = feature_name[len(prefix):]
            template = n.get(direction, "")
            suggestion_template = n.get(f"suggest_{direction}")
            return {
                "label":      f"{n['label']}: {value}",
                "value":      float(shap_value),
                "direction":  direction,
                "narrative":  template.format(v=value) if template else "",
                "actionable": bool(n.get("actionable", False)),
                "suggestion": suggestion_template.format(v=value) if suggestion_template else None,
                "is_bad":     (direction == "positive") if positive_is_bad else (direction == "negative"),
            }

    # Fallback
    return {
        "label": feature_name, "value": float(shap_value), "direction": direction,
        "narrative": f"This feature contributes {shap_value:+.3f} to the score — no plain-English narrative configured.",
        "actionable": False, "suggestion": None,
        "is_bad": (direction == "positive") if positive_is_bad else (direction == "negative"),
    }


def explain_driver(feature_name: str, shap_value: float,
                   context: str = "completion") -> dict:
    """context = 'completion' (positive SHAP = good) or 'duration' (positive SHAP = bad)."""
    if context == "duration":
        return _explain(feature_name, shap_value,
                        DURATION_NUM, DURATION_CAT, positive_is_bad=True)
    return _explain(feature_name, shap_value,
                    COMPLETION_NUM, COMPLETION_CAT, positive_is_bad=False)
