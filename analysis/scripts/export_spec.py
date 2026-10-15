#!/usr/bin/env python
"""Turn the analysis outputs into analysis/output/fa_feature_spec.json.

The spec is what a real-time scorer needs: preprocessing, channel groups (Muse names),
window, the features with their band limits, the sign that means "more on-task", a
default weight per feature and alternative weight sets per expertise profile, plus
calibration guidance. Weights are standardised logistic-regression coefficients from
the leave-one-subject-out model fitted on per-subject z-scored features (analyze.py),
re-signed so that positive = more on-task, averaged over datasets per expertise and
normalised to unit L1 norm. Features whose sign disagrees between datasets within an
expertise group get a reduced weight and are flagged.
"""
from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import pandas as pd

from common import BANDS, DERIV_DIR, OUTPUT_DIR, log

CORE = [
    ("frontal", "theta_log"), ("frontal", "alpha_log"), ("frontal", "beta_log"), ("frontal", "delta_log"),
    ("temporal", "theta_log"), ("temporal", "alpha_log"), ("temporal", "beta_log"), ("temporal", "delta_log"),
    ("frontal", "theta_alpha_ratio"), ("temporal", "theta_alpha_ratio"),
    ("frontal", "ap_exponent"), ("temporal", "ap_exponent"), ("temporal", "iaf"), ("frontal", "iaf"),
]
MUSE_GROUPS = {
    "frontal": {"muse_channels": ["AF7", "AF8"], "reference": "Fpz (device reference)",
                "source_proxies": {"ds001787": ["Fp1", "Fp2", "AF7", "AF8"], "osf3uszv": ["F7", "F8"]}},
    "temporal": {"muse_channels": ["TP9", "TP10"], "reference": "Fpz (device reference)",
                 "source_proxies": {"ds001787": ["T7", "T8", "TP7", "TP8", "P7", "P8", "P9", "P10"],
                                    "osf3uszv": ["T3", "T4", "T5", "T6", "A1", "A2"]}},
}
FEATURE_DEF = {
    "delta_log": {"type": "log10_bandpower", "band_hz": BANDS["delta"]},
    "theta_log": {"type": "log10_bandpower", "band_hz": BANDS["theta"]},
    "alpha_log": {"type": "log10_bandpower", "band_hz": BANDS["alpha"]},
    "beta_log": {"type": "log10_bandpower", "band_hz": BANDS["beta"]},
    "theta_alpha_ratio": {"type": "log10_ratio", "numerator_hz": BANDS["theta"], "denominator_hz": BANDS["alpha"]},
    "ap_exponent": {"type": "aperiodic_exponent", "fit_range_hz": [2.0, 30.0], "method": "specparam/fooof fixed mode"},
    "iaf": {"type": "individual_alpha_frequency", "search_range_hz": [7.0, 13.0], "method": "specparam peak CF"},
}


def evaluate_spec_score(features: list[dict]) -> dict:
    """AUC of the plain weighted z-score the spec describes, on the 60-s fpz features."""
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(DERIV_DIR / "features.csv")
    df = df[(df.reference == "fpz") & (df.window_s == 60.0) & df.label.isin(["on_task", "mind_wandering"])].copy()
    df["y"] = (df.label == "on_task").astype(int)
    df["subj"] = df.dataset + ":" + df.subject
    names = [ft["name"] for ft in features]
    g = df.groupby("subj")[names]
    z = ((df[names] - g.transform("mean")) / (g.transform("std") + 1e-9)).fillna(0.0)
    out = {}
    for (ds, exp), idx in df.groupby(["dataset", "group"]).groups.items():
        for profile in ("expert", "novice", "default"):
            w = np.array([ft["default_weight_on_task"] if profile == "default" else ft["by_expertise"][profile]["weight_on_task"]
                          for ft in features])
            s = z.loc[idx].to_numpy() @ w
            y = df.loc[idx, "y"].to_numpy()
            per_subj = []
            for sj in df.loc[idx, "subj"].unique():
                m = (df.loc[idx, "subj"] == sj).to_numpy()
                if y[m].sum() >= 3 and (1 - y[m]).sum() >= 3:
                    per_subj.append(roc_auc_score(y[m], s[m]))
            out[f"{ds}:{exp}:weights={profile}"] = {
                "auc_pooled_zscored": round(float(roc_auc_score(y, s)), 3),
                "auc_within_subject_mean": round(float(np.mean(per_subj)), 3) if per_subj else None,
                "n_subjects_within": len(per_subj),
            }
    return out


def main() -> int:
    eff = pd.read_csv(OUTPUT_DIR / "effects.csv")
    cls = pd.read_csv(OUTPUT_DIR / "classification.csv")
    w = pd.read_csv(OUTPUT_DIR / "weights.csv")
    eff = eff[(eff.reference == "fpz") & (eff.window_s == 60.0)]

    features = []
    for g, f in CORE:
        name = f"{g}_{f}"
        e = eff[eff.feature == name]
        ww = w[w.feature == name]
        by_exp = {}
        for exp in ("expert", "novice"):
            ee = e[e.expertise == exp]
            wexp = ww[ww.expertise == exp]
            ds_signs = {r.dataset: float(np.sign(r.mean_diff_mw_minus_ot)) for r in ee.itertuples()}
            consistent = len(set(ds_signs.values())) <= 1
            d_mean = float(ee.d_between_subject.mean()) if len(ee) else np.nan
            # weight: on-task-positive = -(mind-wandering weight)
            wt = float(-wexp.weight_mw.mean()) if len(wexp) else 0.0
            by_exp[exp] = {
                "on_task_direction": "higher" if d_mean < 0 else "lower",  # d is MW - on-task
                "cohens_d_mw_minus_on_task_mean": None if np.isnan(d_mean) else round(d_mean, 3),
                "per_dataset_d": {r.dataset: round(r.d_between_subject, 3) for r in ee.itertuples()},
                "per_dataset_p": {r.dataset: round(r.p, 4) for r in ee.itertuples()},
                "sign_consistent_across_datasets": consistent,
                "logreg_weight_on_task": round(wt, 4),
                "_d_mean": d_mean,
            }
        features.append({"name": name, "channel_group": g, **FEATURE_DEF[f], "by_expertise": by_exp})

    # Scorer weights come from the univariate effect sizes (robust to the strong collinearity
    # of band powers), on-task-positive, shrunk x0.5 when the sign disagrees between the two
    # datasets, zeroed when |d| < 0.15, and normalised to unit L1 per profile.
    for exp in ("expert", "novice"):
        d = np.array([ft["by_expertise"][exp]["_d_mean"] for ft in features])
        cons = np.array([ft["by_expertise"][exp]["sign_consistent_across_datasets"] for ft in features])
        wt = -d * np.where(cons, 1.0, 0.5)
        wt[np.abs(d) < 0.15] = 0.0
        wt = wt / (np.abs(wt).sum() + 1e-12)
        for ft, a in zip(features, wt):
            ft["by_expertise"][exp]["weight_on_task"] = round(float(a), 4)
            del ft["by_expertise"][exp]["_d_mean"]
    for ft in features:
        ft["default_weight_on_task"] = round(float(np.mean([ft["by_expertise"][e]["weight_on_task"] for e in ("expert", "novice")])), 4)

    spec_eval = evaluate_spec_score(features)

    c = cls[(cls.reference == "fpz") & (cls.window_s == 60.0) & (cls.classifier == "logreg")]
    acc = {}
    for (ds, exp), cc in c.groupby(["dataset", "expertise"]):
        best = cc.sort_values("loso_cal_auc_pooled", ascending=False).iloc[0]
        acc[f"{ds}:{exp}"] = {
            "best_feature_set_loso": best.feature_set,
            "loso_auc_per_subject_zscored": round(float(best.loso_cal_auc_pooled), 3),
            "loso_balanced_accuracy_per_subject_zscored": round(float(best.loso_cal_bacc_pooled), 3),
            "loso_auc_raw_features": round(float(best.loso_raw_auc_pooled), 3),
            "within_subject_auc_mean": None if pd.isna(best.within_auc) else round(float(best.within_auc), 3),
            "within_subject_balanced_accuracy_mean": None if pd.isna(best.within_bacc) else round(float(best.within_bacc), 3),
            "n_epochs": int(best.n_epochs),
        }

    spec = {
        "name": "fa_on_task_score",
        "version": f"0.1.0-{date.today().isoformat()}",
        "description": "Focused-attention (breath focus) vs mind-wandering score for Muse Athena EEG, "
                       "derived from OpenNeuro ds001787 and OSF 3uszv restricted to Muse-proximal channels.",
        "target": {"positive_class": "on_task", "negative_class": "mind_wandering",
                   "labels_source": "experience-sampling probes (ds001787: mind-wandering rating <=1 vs >=2; "
                                    "osf3uszv: 'focusing on breath' vs 'thinking about something else')"},
        "sampling_rate_hz": 256,
        "preprocessing": {
            "bandpass_hz": [0.5, 45.0], "notch_hz": [50, 60],
            "reference": "as recorded by Muse (Fpz); do not re-reference",
            "segment_s": 2.0, "segment_overlap": 0.5, "psd": "Welch, Hann, 2-s segments -> 0.5 Hz resolution",
            "artifact_rejection": {"per_channel_peak_to_peak_max_uv": 150, "flat_std_min_uv": 0.5,
                                   "min_fraction_segments_kept_per_channel": 0.5},
            "group_aggregation": "mean PSD across the group's channels, then features",
        },
        "window": {"length_s": 60, "hop_s": 5,
                   "note": "Accuracy is flat from 30 to 60 s (see window_curve.csv); 30 s is acceptable when latency matters. "
                           "Discard the first 10-15 s after any stimulus change."},
        "channel_groups": MUSE_GROUPS,
        "features": features,
        "score": {
            "formula": "score = sum_i weight_i * z_i, where z_i is the feature z-scored against the wearer's own "
                       "running baseline (median/MAD over the last >= 5 min, or a calibration sit); "
                       "positive = more on-task. Weights: default_weight_on_task, or by_expertise[profile].weight_on_task.",
            "profile_selection": "Use 'novice' unless the wearer has > ~1000 h of practice; the alpha and 1/f terms flip sign between profiles.",
        },
        "calibration": {
            "required": True,
            "why": "LOSO AUC with raw features is close to chance; per-subject z-scoring recovers most of the "
                   "within-subject separability (see expected_performance).",
            "procedure": "1) 10-15 min of normal meditation with ~8 probes (on-task vs MW self-report) to estimate "
                         "per-feature mean/SD and, if possible, the wearer's sign for alpha; 2) refit weights with "
                         "the pooled model as a prior once >= 20 labelled probes exist.",
        },
        "expected_performance": {
            "note": "AUC/balanced accuracy for 60-s pre-probe windows, Muse-proximal channels, Fpz reference. "
                    "'loso' = leave-one-subject-out; 'within_subject' = stratified CV inside each subject with >= 5 "
                    "probes per class. Chance = 0.5.",
            "logistic_regression_by_group": acc,
            "spec_weighted_score": {
                "note": "Plain weighted sum of per-subject z-scored features with the weights in this file. "
                        "In-sample (weights derived from the same subjects' effect sizes): optimistic by a few "
                        "AUC points; the cross-profile rows (expert weights on novices and vice versa) show the "
                        "cost of using the wrong alpha sign.",
                **spec_eval,
            },
        },
        "recommended_core_features": [
            {"name": "temporal_alpha_log", "why": "largest effect; sign flips with expertise (experts: alpha higher on-task; novices: alpha higher during mind-wandering) - must be calibrated per wearer"},
            {"name": "temporal_ap_exponent", "why": "only feature with the same sign in all four groups (steeper 1/f during mind-wandering)"},
            {"name": "temporal_delta_log", "why": "broadband low-frequency rise during mind-wandering in 3/4 groups (drowsiness-like)"},
            {"name": "frontal_beta_log", "why": "rises during mind-wandering in 3/4 groups (arousal/EMG); drop if the wearer's frontal channels are noisy"},
            {"name": "temporal_theta_alpha_ratio", "why": "expertise-dependent, mirrors alpha; useful once the wearer's alpha sign is known"},
        ],
        "provenance": {
            "datasets": {
                "ds001787": "OpenNeuro ds001787 v1.1.1 (CC0): 12 expert + 12 novice meditators, 64-ch BioSemi, 1-3 sessions, probes every ~2.5 min",
                "osf3uszv": "OSF 3uszv/v8msp (CC0): 29 meditators + 29 novices, 19-ch 10-20 + A1/A2, probes every ~1 min",
            },
            "scripts": "analysis/scripts/*.py (download -> prepare -> extract_features -> analyze -> export_spec)",
        },
    }
    with open(OUTPUT_DIR / "fa_feature_spec.json", "w") as f:
        json.dump(spec, f, indent=1)
    log(f"wrote {OUTPUT_DIR / 'fa_feature_spec.json'} with {len(features)} features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
