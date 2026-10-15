#!/usr/bin/env python
"""Compute Muse-proximal spectral features for the pre-probe windows of both datasets.

For every probe (row of derivatives/<dataset>/probes.csv), window length W in
WINDOWS_S and reference scheme in {fpz, avg} this computes, per channel group
(frontal ~ AF7/AF8, temporal ~ TP9/TP10):

  log10 absolute band power   delta/theta/alpha/beta   (Welch, 2-s Hann, mean PSD in band)
  relative band power         band / total 1-30 Hz
  theta_alpha_ratio, alpha_beta_ratio (log10 differences)
  alpha asymmetry             log10 alpha(right) - log10 alpha(left)
  iaf, alpha_peak_power       individual alpha peak (7-13 Hz) from specparam/fooof
  ap_exponent, ap_offset      aperiodic 1/f fit (fooof, 2-30 Hz, fixed mode)

The window ends 0.5 s before the probe and starts at onset-W, but never before the
previous probe's last button press + 3 s. 2-s segments with peak-to-peak > 150 uV or
near-flat signal are dropped per channel before Welch; channels keeping < 50 % of their
segments are dropped, and windows with < max(W/2, min(10 s, W)) of data are NaN.

Reference schemes
  fpz  Muse-faithful: proxies minus Fpz (ds001787) / minus mean(FP1, FP2) (OSF, where the
       frontal group is then F7/F8 only because FP1/FP2 become pure asymmetry).
  avg  common average of all scalp channels (what the source papers used).

Output: derivatives/features.csv (one row per probe x window x reference).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from common import BANDS, DERIV_DIR, MUSE_PROXIES, WINDOWS_S, log

warnings.filterwarnings("ignore")

SEG_S = 2.0
P2P_MAX_V = 150e-6
FLAT_V = 0.5e-6
REFERENCES = ("fpz", "avg")
GROUPS = ("frontal", "temporal")


def group_channels(dataset: str, ref: str) -> dict[str, list[str]]:
    prox = MUSE_PROXIES[dataset]
    frontal = list(prox["frontal"])
    if dataset == "osf3uszv" and ref == "fpz":
        frontal = ["F7", "F8"]
    return {"frontal": frontal, "temporal": list(prox["temporal"])}


def reference_signal(raw: mne.io.BaseRaw, dataset: str, ref: str) -> np.ndarray:
    if ref == "avg":
        return raw.get_data(picks=["AVG"])[0]
    if dataset == "ds001787":
        return raw.get_data(picks=["Fpz"])[0]
    return raw.get_data(picks=["FP1", "FP2"]).mean(axis=0)


def fooof_features(freqs: np.ndarray, psd: np.ndarray) -> dict:
    from fooof import FOOOF

    out = {"iaf": np.nan, "alpha_peak_power": 0.0, "ap_exponent": np.nan, "ap_offset": np.nan, "fooof_r2": np.nan}
    try:
        fm = FOOOF(peak_width_limits=(1.0, 8.0), max_n_peaks=6, min_peak_height=0.05, aperiodic_mode="fixed",
                   verbose=False)
        fm.fit(freqs, psd, freq_range=(2.0, 30.0))
        out["ap_offset"], out["ap_exponent"] = fm.aperiodic_params_
        out["fooof_r2"] = fm.r_squared_
        peaks = fm.peak_params_
        if len(peaks):
            a = peaks[(peaks[:, 0] >= 7) & (peaks[:, 0] <= 13)]
            if len(a):
                best = a[np.argmax(a[:, 1])]
                out["iaf"], out["alpha_peak_power"] = best[0], best[1]
    except Exception:
        pass
    return out


def band_features(freqs: np.ndarray, psd_ch: np.ndarray, chans: list[str]) -> dict:
    """psd_ch: (n_ch, n_freq) PSD in V^2/Hz for the channels of one group."""
    psd = psd_ch.mean(axis=0)
    feats = {}
    total = psd[(freqs >= 1) & (freqs <= 30)].mean()
    for b, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        feats[f"{b}_log"] = np.log10(psd[m].mean())
        feats[f"{b}_rel"] = psd[m].mean() / total
    feats["theta_alpha_ratio"] = feats["theta_log"] - feats["alpha_log"]
    feats["alpha_beta_ratio"] = feats["alpha_log"] - feats["beta_log"]
    # hemispheric asymmetry: odd-numbered / left-named channels vs right
    left = [i for i, c in enumerate(chans) if c[-1] in "13579" or c.endswith("1")]
    right = [i for i, c in enumerate(chans) if c[-1] in "02468" and not c.endswith("z")]
    m = (freqs >= BANDS["alpha"][0]) & (freqs < BANDS["alpha"][1])
    if left and right:
        feats["alpha_asym"] = np.log10(psd_ch[right][:, m].mean()) - np.log10(psd_ch[left][:, m].mean())
    else:
        feats["alpha_asym"] = np.nan
    feats.update(fooof_features(freqs, psd))
    return feats


def window_features(x: np.ndarray, sfreq: float, chans: list[str]) -> tuple[dict, float, int]:
    """x: (n_ch, n_samples) referenced data for one group.

    Segments are rejected per channel (a blink on one proxy channel does not discard the
    others); a channel is dropped if it keeps < 50 % of its segments. Returns
    (features, mean fraction of segments kept, n_segments).
    """
    n_seg = int(SEG_S * sfreq)
    hop = n_seg // 2
    starts = list(range(0, x.shape[1] - n_seg + 1, hop))
    n_total = len(starts)
    if n_total == 0:
        return {}, 0.0, 0
    segs = np.stack([x[:, s:s + n_seg] for s in starts], axis=0)  # (n_seg, n_ch, n_samp)
    p2p = segs.max(axis=2) - segs.min(axis=2)
    good = (p2p <= P2P_MAX_V) & (segs.std(axis=2) >= FLAT_V)  # (n_seg, n_ch)
    f, p = welch(segs, fs=sfreq, nperseg=n_seg, noverlap=0, window="hann", axis=2)  # (n_seg, n_ch, n_f)
    frac = good.mean(axis=0)
    keep_ch = frac >= 0.5
    if not keep_ch.any():
        return {}, float(frac.mean()), n_total
    psd_ch = np.full((x.shape[0], p.shape[2]), np.nan)
    for c in np.flatnonzero(keep_ch):
        psd_ch[c] = p[good[:, c], c].mean(axis=0)
    chans_kept = [c for c, k in zip(chans, keep_ch) if k]
    return band_features(f, psd_ch[keep_ch], chans_kept), float(frac.mean()), n_total


def process_file(args) -> list[dict]:
    dataset, fif, rows = args
    raw = mne.io.read_raw_fif(DERIV_DIR / dataset / fif, preload=True, verbose="error")
    sf = raw.info["sfreq"]
    out = []
    for ref in REFERENCES:
        refsig = reference_signal(raw, dataset, ref)
        groups = group_channels(dataset, ref)
        gdata = {g: raw.get_data(picks=ch) - refsig for g, ch in groups.items()}
        for r in rows:
            for W in WINDOWS_S:
                end = r["onset_s"] - 0.5
                start = max(end - W, r["prev_end_s"] + 3.0, 1.0)
                avail = end - start
                rec = {**r, "window_s": W, "reference": ref, "avail_s": avail}
                if avail < max(0.5 * W, min(10.0, W)):
                    out.append(rec)
                    continue
                i0, i1 = int(start * sf), int(end * sf)
                ok = True
                for g, ch in groups.items():
                    feats, kept_frac, tot = window_features(gdata[g][:, i0:i1], sf, ch)
                    rec[f"{g}_segments_kept_frac"] = kept_frac
                    rec[f"{g}_segments_total"] = tot
                    if not feats:
                        ok = False
                        continue
                    rec.update({f"{g}_{k}": v for k, v in feats.items()})
                rec["complete"] = ok
                out.append(rec)
    log(f"  {dataset}/{fif}: {len(out)} feature rows")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=["ds001787", "osf3uszv"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N files per dataset (debug)")
    args = ap.parse_args()

    jobs = []
    for ds in args.datasets:
        probes = pd.read_csv(DERIV_DIR / ds / "probes.csv")
        probes["probe_idx"] = probes.groupby(["subject", "session"]).cumcount()
        for i, (fif, grp) in enumerate(probes.groupby("fif")):
            if args.limit and i >= args.limit:
                break
            jobs.append((ds, fif, grp.to_dict("records")))
    log(f"{len(jobs)} recordings")
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(process_file, jobs):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(DERIV_DIR / "features.csv", index=False)
    log(df.groupby(["dataset", "reference", "window_s"]).complete.mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
