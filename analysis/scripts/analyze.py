#!/usr/bin/env python
"""Statistics, classifiers and figures for on-task vs mind-wandering separability.

Reads derivatives/features.csv (from extract_features.py) and writes to analysis/output/:
  effects.csv                 per dataset x expertise x reference x window x feature:
                              paired within-subject effect (MW - on-task), Cohen's d, t, p
  classification.csv          within-subject CV, leave-one-subject-out (LOSO) with and
                              without per-subject standardisation, per feature set / window
  window_curve.csv            LOSO AUC vs window length
  weights.csv                 standardised logistic-regression weights (calibrated LOSO model)
  fig_effects.png, fig_accuracy.png, fig_alpha_sign.png
  summary.json                headline numbers consumed by export_spec.py
"""
from __future__ import annotations

import json
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import DERIV_DIR, OUTPUT_DIR, log  # noqa: E402

warnings.filterwarnings("ignore")
RNG = 0

BAND_FEATS = ["delta_log", "theta_log", "alpha_log", "beta_log"]
REL_FEATS = ["delta_rel", "theta_rel", "alpha_rel", "beta_rel"]
RATIO_FEATS = ["theta_alpha_ratio", "alpha_beta_ratio", "alpha_asym"]
SPEC_FEATS = ["iaf", "alpha_peak_power", "ap_exponent", "ap_offset"]
ALL_FEATS = BAND_FEATS + REL_FEATS + RATIO_FEATS + SPEC_FEATS

FEATURE_SETS = {
    "frontal_bands": [f"frontal_{f}" for f in BAND_FEATS],
    "temporal_bands": [f"temporal_{f}" for f in BAND_FEATS],
    "both_bands": [f"{g}_{f}" for g in ("frontal", "temporal") for f in BAND_FEATS],
    "both_bands_rel_ratios": [f"{g}_{f}" for g in ("frontal", "temporal") for f in BAND_FEATS + REL_FEATS + RATIO_FEATS],
    "both_all": [f"{g}_{f}" for g in ("frontal", "temporal") for f in ALL_FEATS],
    "frontal_all": [f"frontal_{f}" for f in ALL_FEATS],
    "temporal_all": [f"temporal_{f}" for f in ALL_FEATS],
}
MIN_PER_CLASS_WITHIN = 5


def load() -> pd.DataFrame:
    df = pd.read_csv(DERIV_DIR / "features.csv")
    df = df[df.label.isin(["on_task", "mind_wandering"])].copy()
    df["y"] = (df.label == "mind_wandering").astype(int)
    df["expertise"] = df.group.map({"expert": "expert", "novice": "novice"})
    df["subj"] = df.dataset + ":" + df.subject
    return df


def subject_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("subj")[cols]
    out[cols] = (out[cols] - g.transform("mean")) / (g.transform("std") + 1e-9)
    return out


# ----------------------------------------------------------------------------- effects
def effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feats = [c for c in df.columns if any(c.endswith(f) for f in ALL_FEATS) and c.split("_")[0] in ("frontal", "temporal")]
    for (ds, exp, ref, W), d in df.groupby(["dataset", "expertise", "reference", "window_s"]):
        for f in feats:
            per_subj = []
            for s, ds_ in d.groupby("subj"):
                a, b = ds_.loc[ds_.y == 1, f].dropna(), ds_.loc[ds_.y == 0, f].dropna()
                if len(a) >= 3 and len(b) >= 3:
                    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
                    per_subj.append({"subj": s, "diff": a.mean() - b.mean(), "d_within": (a.mean() - b.mean()) / sd})
            if len(per_subj) < 4:
                continue
            ps = pd.DataFrame(per_subj)
            t, p = stats.ttest_1samp(ps["diff"], 0.0)
            w = stats.wilcoxon(ps["diff"]).pvalue if (ps["diff"] != 0).any() else np.nan
            rows.append({
                "dataset": ds, "expertise": exp, "reference": ref, "window_s": W, "feature": f,
                "n_subjects": len(ps), "mean_diff_mw_minus_ot": ps["diff"].mean(),
                "d_between_subject": ps["diff"].mean() / (ps["diff"].std(ddof=1) + 1e-12),
                "mean_d_within": ps["d_within"].mean(), "frac_subjects_positive": (ps["diff"] > 0).mean(),
                "t": t, "p": p, "p_wilcoxon": w,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- classifiers
def make_clf(kind: str):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(), StandardScaler(),
                             LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
    return make_pipeline(SimpleImputer(), StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def safe_auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def within_subject(df: pd.DataFrame, cols: list[str], kind: str) -> dict:
    aucs, baccs, n = [], [], 0
    for s, d in df.groupby("subj"):
        if d.y.sum() < MIN_PER_CLASS_WITHIN or (1 - d.y).sum() < MIN_PER_CLASS_WITHIN:
            continue
        X, y = d[cols].to_numpy(), d.y.to_numpy()
        k = int(min(5, d.y.sum(), (1 - d.y).sum()))
        scores = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=RNG).split(X, y):
            clf = make_clf(kind).fit(X[tr], y[tr])
            scores[te] = clf.decision_function(X[te])
        aucs.append(safe_auc(y, scores))
        baccs.append(balanced_accuracy_score(y, scores > 0))
        n += 1
    return {"within_auc": np.nanmean(aucs) if aucs else np.nan, "within_bacc": np.nanmean(baccs) if baccs else np.nan,
            "within_auc_sd": np.nanstd(aucs) if aucs else np.nan, "within_n_subjects": n}


def loso(df: pd.DataFrame, cols: list[str], kind: str, calibrated: bool) -> dict:
    d = subject_zscore(df, cols) if calibrated else df
    X, y, g = d[cols].to_numpy(), d.y.to_numpy(), d.subj.to_numpy()
    scores = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        clf = make_clf(kind).fit(X[tr], y[tr])
        scores[te] = clf.decision_function(X[te])
    per_subj_auc = [safe_auc(y[g == s], scores[g == s]) for s in np.unique(g)]
    per_subj_bacc = [balanced_accuracy_score(y[g == s], scores[g == s] > 0) for s in np.unique(g)
                     if len(np.unique(y[g == s])) == 2]
    tag = "loso_cal" if calibrated else "loso_raw"
    return {f"{tag}_auc_pooled": safe_auc(y, scores), f"{tag}_bacc_pooled": balanced_accuracy_score(y, scores > 0),
            f"{tag}_auc_subjmean": np.nanmean(per_subj_auc), f"{tag}_bacc_subjmean": np.nanmean(per_subj_bacc),
            f"{tag}_n_subjects": len(np.unique(g)), "n_epochs": len(y)}


def classification(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, exp, ref, W), d in df.groupby(["dataset", "expertise", "reference", "window_s"]):
        if W not in (30.0, 60.0) and ref != "fpz":
            continue
        for fs_name, cols in FEATURE_SETS.items():
            dd = d.dropna(subset=cols, how="all")
            if dd.subj.nunique() < 4 or dd.y.nunique() < 2:
                continue
            for kind in ("logreg", "lda"):
                rec = {"dataset": ds, "expertise": exp, "reference": ref, "window_s": W, "feature_set": fs_name,
                       "classifier": kind}
                rec.update(within_subject(dd, cols, kind))
                rec.update(loso(dd, cols, kind, calibrated=True))
                rec.update(loso(dd, cols, kind, calibrated=False))
                rows.append(rec)
                log(f"  {ds} {exp} {ref} W={W:.0f} {fs_name} {kind}: within AUC {rec['within_auc']:.3f}, "
                    f"LOSO cal AUC {rec['loso_cal_auc_pooled']:.3f}, raw {rec['loso_raw_auc_pooled']:.3f}")
    # pooled across expertise within dataset (does one model fit everyone?)
    for (ds, ref, W), d in df.groupby(["dataset", "reference", "window_s"]):
        if W != 60.0:
            continue
        for fs_name in ("both_bands", "both_all"):
            cols = FEATURE_SETS[fs_name]
            dd = d.dropna(subset=cols, how="all")
            rec = {"dataset": ds, "expertise": "all", "reference": ref, "window_s": W, "feature_set": fs_name,
                   "classifier": "logreg"}
            rec.update(within_subject(dd, cols, "logreg"))
            rec.update(loso(dd, cols, "logreg", True))
            rec.update(loso(dd, cols, "logreg", False))
            rows.append(rec)
    return pd.DataFrame(rows)


def weights(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Standardised logistic weights (sign: + means higher value -> more mind-wandering)."""
    rows = []
    for (ds, exp), d in df.groupby(["dataset", "expertise"]):
        dd = subject_zscore(d.dropna(subset=cols, how="all"), cols)
        X, y = dd[cols].to_numpy(), dd.y.to_numpy()
        clf = make_clf("logreg").fit(X, y)
        coef = clf[-1].coef_[0]
        for c, w in zip(cols, coef):
            rows.append({"dataset": ds, "expertise": exp, "feature": c, "weight_mw": w})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def fig_effects(eff: pd.DataFrame, ref: str, W: float):
    e = eff[(eff.reference == ref) & (eff.window_s == W)]
    feats = [f"{g}_{f}" for g in ("frontal", "temporal") for f in BAND_FEATS + ["theta_alpha_ratio", "iaf", "ap_exponent"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    combos = [("ds001787", "expert"), ("ds001787", "novice"), ("osf3uszv", "expert"), ("osf3uszv", "novice")]
    colors = ["#1f77b4", "#aec7e8", "#d62728", "#ff9896"]
    for k, ((ds, exp), col) in enumerate(zip(combos, colors)):
        ee = e[(e.dataset == ds) & (e.expertise == exp)].set_index("feature").reindex(feats)
        ypos = np.arange(len(feats)) + (k - 1.5) * 0.18
        ax.barh(ypos, ee.d_between_subject, height=0.17, color=col, label=f"{ds} {exp} (n={int(ee.n_subjects.max())})")
        for yy, (d_, p_) in zip(ypos, zip(ee.d_between_subject, ee.p)):
            if np.isfinite(p_) and p_ < 0.05:
                ax.text(d_ + (0.03 if d_ >= 0 else -0.03), yy, "*", va="center", ha="left" if d_ >= 0 else "right", fontsize=9)
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels(feats)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Cohen's d of within-subject difference (mind-wandering - on-task); * p<0.05 uncorrected")
    ax.set_title(f"Muse-proximal features, {W:.0f}-s pre-probe windows, reference={ref}")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_effects.png", dpi=150)
    plt.close(fig)


def fig_accuracy(cls: pd.DataFrame, curve: pd.DataFrame, ref: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    c = cls[(cls.reference == ref) & (cls.window_s == 60.0) & (cls.classifier == "logreg") & (cls.expertise != "all")]
    sets = ["frontal_bands", "temporal_bands", "both_bands", "both_all"]
    combos = [("ds001787", "expert"), ("ds001787", "novice"), ("osf3uszv", "expert"), ("osf3uszv", "novice")]
    x = np.arange(len(sets))
    ax = axes[0]
    for k, (ds, exp) in enumerate(combos):
        cc = c[(c.dataset == ds) & (c.expertise == exp)].set_index("feature_set").reindex(sets)
        ax.plot(x + k * 0.05, cc.within_auc, "o", color=f"C{k}", label=f"{ds} {exp}: within-subject")
        ax.plot(x + k * 0.05, cc.loso_cal_auc_pooled, "s", mfc="none", color=f"C{k}", label=f"{ds} {exp}: LOSO, per-subject z-scored")
        ax.plot(x + k * 0.05, cc.loso_raw_auc_pooled, "x", color=f"C{k}", label=f"{ds} {exp}: LOSO, raw")
    ax.axhline(0.5, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sets, rotation=20)
    ax.set_ylabel("ROC AUC (60-s windows)")
    ax.set_title("Single-trial separability by feature set")
    ax.legend(fontsize=6, ncol=2)
    ax = axes[1]
    for k, (ds, exp) in enumerate(combos):
        cc = curve[(curve.dataset == ds) & (curve.expertise == exp) & (curve.reference == ref)]
        ax.plot(cc.window_s, cc.loso_cal_auc_pooled, "s-", color=f"C{k}", label=f"{ds} {exp}: LOSO z-scored")
        ax.plot(cc.window_s, cc.within_auc, "o--", color=f"C{k}", alpha=0.6, label=f"{ds} {exp}: within-subject")
    ax.axhline(0.5, color="k", lw=0.8)
    ax.set_xlabel("pre-probe window length (s)")
    ax.set_ylabel("ROC AUC (both groups, all features)")
    ax.set_title("Effect of window length")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_accuracy.png", dpi=150)
    plt.close(fig)


def fig_alpha_sign(df: pd.DataFrame, ref: str, W: float):
    d = df[(df.reference == ref) & (df.window_s == W)]
    feats = ["frontal_alpha_log", "temporal_alpha_log", "frontal_theta_log", "temporal_theta_log", "temporal_iaf", "temporal_ap_exponent"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    combos = [("ds001787", "expert"), ("ds001787", "novice"), ("osf3uszv", "expert"), ("osf3uszv", "novice")]
    for ax, f in zip(axes.ravel(), feats):
        data, labels = [], []
        for ds, exp in combos:
            diffs = []
            for s, dd in d[(d.dataset == ds) & (d.expertise == exp)].groupby("subj"):
                a, b = dd.loc[dd.y == 1, f].dropna(), dd.loc[dd.y == 0, f].dropna()
                if len(a) >= 3 and len(b) >= 3:
                    diffs.append(a.mean() - b.mean())
            data.append(diffs)
            labels.append(f"{'ds001787' if ds == 'ds001787' else 'OSF'}\n{exp}\n(n={len(diffs)})")
        ax.boxplot(data, showfliers=False)
        for i, dd_ in enumerate(data):
            ax.scatter(np.random.normal(i + 1, 0.05, len(dd_)), dd_, s=10, alpha=0.6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f)
        ax.set_ylabel("MW - on-task (per subject)")
    fig.suptitle(f"Sign of the mind-wandering effect per subject ({W:.0f}-s windows, ref={ref})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_alpha_sign.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> int:
    df = load()
    log(f"{len(df)} labelled feature rows; subjects per dataset: {df.groupby('dataset').subj.nunique().to_dict()}")
    log(df[(df.window_s == 60) & (df.reference == 'fpz')].groupby(['dataset', 'expertise', 'label']).size())

    eff = effects(df)
    eff.to_csv(OUTPUT_DIR / "effects.csv", index=False)

    cls = classification(df)
    cls.to_csv(OUTPUT_DIR / "classification.csv", index=False)

    curve = cls[(cls.feature_set == "both_all") & (cls.classifier == "logreg") & (cls.expertise != "all")][
        ["dataset", "expertise", "reference", "window_s", "within_auc", "loso_cal_auc_pooled", "loso_raw_auc_pooled"]]
    curve.to_csv(OUTPUT_DIR / "window_curve.csv", index=False)

    d60 = df[(df.window_s == 60.0) & (df.reference == "fpz")]
    w = weights(d60, FEATURE_SETS["both_all"])
    w.to_csv(OUTPUT_DIR / "weights.csv", index=False)

    fig_effects(eff, "fpz", 60.0)
    fig_accuracy(cls, curve, "fpz")
    fig_alpha_sign(df, "fpz", 60.0)

    summary = {
        "n_subjects": df.groupby("dataset").subj.nunique().to_dict(),
        "n_epochs_60s_fpz": d60.groupby(["dataset", "expertise", "label"]).size().reset_index().to_dict("records"),
        "effects_60s_fpz": eff[(eff.window_s == 60) & (eff.reference == "fpz")].to_dict("records"),
        "classification_60s_fpz": cls[(cls.window_s == 60) & (cls.reference == "fpz")].to_dict("records"),
        "window_curve_fpz": curve[curve.reference == "fpz"].to_dict("records"),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
