#!/usr/bin/env python
"""Prepare ds001787: Muse-proximal channels + reference signals, probe labels.

For every session this writes
  derivatives/ds001787/<sub>_<ses>_raw.fif   256 Hz, 0.5-45 Hz band-pass, 50 Hz notch;
                                             channels = Muse proxies + Fpz + AVG (scalp mean)
and one table derivatives/ds001787/probes.csv with one row per experience-sampling
probe: onset, the ratings (depth of meditation, mind wandering, tiredness; 0-3) and
`label` (on_task if mind-wandering rating <= 1, mind_wandering if >= 2).

Ratings come from the behavioural logs in code/MW_Current_TextFileBIDS.zip because the
EEG Status channel never recorded key "0" (parallel-port bit 0), so the EEG events alone
are ambiguous. Logs are aligned to the EEG probe triggers (value 128) by the mode of the
pairwise onset differences; the session whose log aligns best is used.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from common import BIOSEMI64_AB_TO_1010, DATA_DIR, DERIV_DIR, MUSE_PROXIES, TARGET_SFREQ, log

SRC = DATA_DIR / "ds001787"
OUT = DERIV_DIR / "ds001787"
PROX = MUSE_PROXIES["ds001787"]
KEEP = PROX["frontal"] + PROX["temporal"] + ["Fpz"]


def parse_log(text: str) -> list[dict]:
    """Return probes [{t, depth, mw, tired}] from a *_info.txt behavioural log."""
    probes = []
    for line in text.splitlines():
        m = re.match(r"MW question asked at time\s+([\d.]+)", line)
        if m:
            probes.append({"t": float(m.group(1)), "keys": []})
            continue
        m = re.match(r"Key (\S+) \(code (-?\d+)\) pressed at time\s+([\d.]+) seconds, status (\d)", line)
        if m and probes:
            key, status, t = m.group(1), int(m.group(4)), float(m.group(3))
            if key in "0123" and status in (1, 2, 3):
                probes[-1]["keys"].append((status, int(key), t))
    out = []
    for p in probes:
        by_status = {s: k for s, k, _ in p["keys"]}
        times = [t for _, _, t in p["keys"]]
        out.append({
            "t": p["t"],
            "depth": by_status.get(1, np.nan),
            "mw": by_status.get(2, np.nan),
            "tired": by_status.get(3, np.nan),
            "t_last_response": max(times) if times else np.nan,
        })
    return out


def load_logs() -> dict[str, list[dict]]:
    zpath = SRC / "code" / "MW_Current_TextFileBIDS.zip"
    logs = {}
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            base = Path(name).name
            if not base.endswith("_info.txt") and not base.endswith("_info_old.txt"):
                continue
            logs[base] = parse_log(z.read(name).decode("utf-8", "replace"))
    return logs


def align(eeg_onsets: np.ndarray, log_probes: list[dict], tol: float = 1.0):
    """Find the constant offset mapping log times to EEG times. Returns (offset, n_matched)."""
    if not log_probes or len(eeg_onsets) == 0:
        return None, 0
    lt = np.array([p["t"] for p in log_probes])
    diffs = (eeg_onsets[:, None] - lt[None, :]).ravel()
    # mode of the differences at 0.5-s resolution
    hist, edges = np.histogram(diffs, bins=np.arange(diffs.min() - 1, diffs.max() + 1, 0.5))
    i = np.argmax(hist)
    cand = diffs[(diffs >= edges[i] - 0.5) & (diffs < edges[i + 1] + 0.5)]
    offset = float(np.median(cand))
    matched = sum(np.min(np.abs(eeg_onsets - (p["t"] + offset))) < tol for p in log_probes)
    return offset, int(matched)


def process_session(bdf: Path, logs: dict, participants: pd.DataFrame) -> list[dict]:
    sub = bdf.name.split("_")[0]  # sub-001
    ses = bdf.name.split("_")[1]  # ses-01
    group = participants.set_index("participant_id").loc[sub, "group"]
    raw = mne.io.read_raw_bdf(bdf, preload=True, verbose="error")
    raw.rename_channels({k: v for k, v in BIOSEMI64_AB_TO_1010.items() if k in raw.ch_names})
    scalp = [ch for ch in BIOSEMI64_AB_TO_1010.values() if ch in raw.ch_names]
    raw.pick(scalp)
    raw.set_channel_types({ch: "eeg" for ch in scalp})
    raw.notch_filter(50.0, verbose="error")
    raw.filter(0.5, 45.0, verbose="error")
    avg = raw.get_data().mean(axis=0, keepdims=True)
    avg_info = mne.create_info(["AVG"], raw.info["sfreq"], ["misc"])
    raw.add_channels([mne.io.RawArray(avg, avg_info, verbose="error")], force_update_info=True)
    raw.pick(KEEP + ["AVG"])
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose="error")

    events = pd.read_csv(bdf.with_name(bdf.name.replace("_eeg.bdf", "_events.tsv")), sep="\t")
    stim = events[(events.trial_type == "stimulus") & (events.value == 128)].onset.to_numpy()

    # Pick the behavioural log whose probe-interval pattern aligns best with this
    # session's probe triggers. Matching is done over *all* logs because the log
    # numbering is not identical to the BIDS numbering (e.g. sub-013 <-> sub14_info.txt).
    best = None
    for k in logs:
        off, n = align(stim, logs[k])
        if off is not None and (best is None or n > best[2]):
            best = (k, off, n)
    rows = []
    if best is None or best[2] < max(5, 0.8 * len(logs[best[0]])):
        log(f"  {sub} {ses}: no behavioural log aligned (best={best}); skipping labels")
        return rows
    k, off, n = best
    log(f"  {sub} {ses}: log {k} offset {off:.1f}s matched {n}/{len(logs[k])} probes (EEG has {len(stim)})")
    prev_end = -np.inf
    for p in logs[k]:
        t_eeg = p["t"] + off
        j = np.argmin(np.abs(stim - t_eeg))
        if abs(stim[j] - t_eeg) > 1.0:
            continue
        onset = float(stim[j])
        mw = p["mw"]
        if np.isnan(mw):
            label = None
        else:
            label = "on_task" if mw <= 1 else "mind_wandering"
        rows.append({
            "dataset": "ds001787", "subject": sub, "session": ses, "group": group,
            "onset_s": onset,
            "prev_end_s": prev_end if np.isfinite(prev_end) else 0.0,
            "depth": p["depth"], "mw": mw, "tired": p["tired"], "label": label,
            "fif": f"{sub}_{ses}_raw.fif",
        })
        if np.isfinite(p["t_last_response"]):
            prev_end = p["t_last_response"] + off
        else:
            prev_end = onset + 15.0
    raw.save(OUT / f"{sub}_{ses}_raw.fif", overwrite=True, verbose="error")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", help="subset like 001 002")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(SRC / "participants.tsv", sep="\t")
    logs = load_logs()
    bdfs = sorted(SRC.glob("sub-*/ses-*/eeg/*_eeg.bdf"))
    if args.subjects:
        bdfs = [b for b in bdfs if b.name.split("_")[0][4:] in args.subjects]
    rows = []
    for bdf in bdfs:
        log(f"processing {bdf.name}")
        rows.extend(process_session(bdf, logs, participants))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probes.csv", index=False)
    log(df.groupby(["group", "label"]).size())
    return 0


if __name__ == "__main__":
    sys.exit(main())
