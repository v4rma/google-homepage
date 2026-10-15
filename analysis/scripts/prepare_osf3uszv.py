#!/usr/bin/env python
"""Prepare OSF 3uszv task recordings: Muse-proximal channels + probe labels.

Each `*_task_*.txt.cpgz` is a gzip-compressed cpio archive wrapping one CSV with 23
numeric columns (19 scalp 10-20 sites, A1/A2 ear electrodes, 2 EOG) and an `Events`
column with RS232 trigger strings. The CSV has no sampling-rate field; the published
MATLAB code assumes 512 Hz, but the 50 Hz mains line and the ~10 Hz alpha peak only
appear at 2048 Hz, and durations (task ~55 min, rest 5 min) agree with 2048 Hz, so
2048 Hz is used here.

Experience-sampling probes (trigger 100) are followed by the answers
  answer      1 = focusing on the breath, 2 = thinking about something else (mind
              wandering), 3 = distracted by a sensation
  confidence, sleepiness, then focus (answer 1) / engagement, visual, auditory (answer 2)
  / sensation (answer 3)
exactly as in the authors' preprocessing.m.

Outputs derivatives/osf3uszv/<file>_raw.fif (256 Hz, 0.5-45 Hz, 50 Hz notch; proxies +
AVG reference signal) and derivatives/osf3uszv/probes.csv.
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from common import DATA_DIR, DERIV_DIR, MUSE_PROXIES, TARGET_SFREQ, log

RAW_DIR = DATA_DIR / "osf3uszv" / "raw"
OUT = DERIV_DIR / "osf3uszv"
PROX = MUSE_PROXIES["osf3uszv"]
SFREQ_IN = 2048.0
COLS = ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "PZ",
        "P4", "T6", "O1", "O2", "A1", "A2", "SensorCEOG", "SensorDEOG", "Events"]
SCALP19 = COLS[:19]
KEEP = PROX["frontal"] + PROX["temporal"]
TRIG_RE = re.compile(r"RS232 Trigger: (\d+)\(")


def open_cpio_csv(path: Path):
    """Return a file object positioned at the CSV inside the gzip'd cpio archive."""
    f = gzip.open(path, "rb")
    hdr = f.read(76)
    if hdr[:6] != b"070707":
        f.seek(0)  # plain gzip'd CSV
        return f
    namesize = int(hdr[59:65], 8)
    f.read(namesize)
    return f


def read_task_file(path: Path):
    """Stream the CSV; return (data[23, n] float32 at 2048 Hz, events[(sample, code)])."""
    f = open_cpio_csv(path)
    chunks, events, n0 = [], [], 0
    reader = pd.read_csv(f, header=0, names=COLS, usecols=COLS, chunksize=1_000_000, engine="c",
                         on_bad_lines="skip", encoding="latin-1")
    for chunk in reader:
        ev = chunk["Events"]
        mask = ev.notna()
        for idx, s in zip(np.flatnonzero(mask.to_numpy()), ev[mask]):
            m = TRIG_RE.search(str(s))
            if m:
                events.append((n0 + int(idx), int(m.group(1))))
        x = chunk[COLS[:-1]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        good = ~np.isnan(x).any(axis=1)
        chunks.append(x[good])
        n0 += len(chunk)
    data = np.concatenate(chunks).T
    return data, events


def parse_probes(events: list[tuple[int, int]]) -> list[dict]:
    ev = [(s, c) for s, c in events if c != 0]  # trigger 0 is a known glitch
    codes = [c for _, c in ev]
    probes = []
    for t, (sample, code) in enumerate(ev):
        if code != 100:
            continue

        def get(k):
            return codes[t + k] if t + k < len(codes) and codes[t + k] != 100 else np.nan

        answer = get(1)
        p = {"sample": sample, "answer": answer, "confidence": get(2), "sleepiness": get(3),
             "focus": np.nan, "engagement": np.nan, "visual": np.nan, "auditory": np.nan, "sensation": np.nan}
        if answer == 1:
            p["focus"], n_resp = get(4), 4
        elif answer == 2:
            p["engagement"], p["visual"], p["auditory"], n_resp = get(4), get(5), get(6), 6
        elif answer == 3:
            p["sensation"], n_resp = get(4), 4
        else:
            n_resp = 3
        last = t + n_resp
        while last > t and (last >= len(ev) or ev[last][1] == 100):
            last -= 1
        p["last_response_sample"] = ev[last][0] if last > t else sample
        probes.append(p)
    return probes


def process_file(path: Path) -> list[dict]:
    m = re.match(r"^([mc])_task_(n?\d+)\.txt", path.name)
    group = {"m": "expert", "c": "novice"}[m.group(1)]
    subject = f"{m.group(1)}{m.group(2)}"
    data, events = read_task_file(path)
    # anti-aliased 8x decimation (2048 -> 256 Hz) before building the Raw keeps memory low
    q = int(round(SFREQ_IN / TARGET_SFREQ))
    data = resample_poly(data[:21], 1, q, axis=1).astype(np.float64) * 1e-6  # uV -> V
    info = mne.create_info(COLS[:21], TARGET_SFREQ, ["eeg"] * 21)
    raw = mne.io.RawArray(data, info, verbose="error")
    del data
    raw.notch_filter(50.0, verbose="error")
    raw.filter(0.5, 45.0, verbose="error")
    avg = raw.get_data(picks=SCALP19).mean(axis=0, keepdims=True)
    raw.add_channels([mne.io.RawArray(avg, mne.create_info(["AVG"], TARGET_SFREQ, ["misc"]), verbose="error")],
                     force_update_info=True)
    raw.pick(KEEP + ["AVG"])
    fif = f"{path.name.split('.')[0]}_raw.fif"
    raw.save(OUT / fif, overwrite=True, verbose="error")

    rows, prev_end = [], 0.0
    for p in parse_probes(events):
        onset = p["sample"] / SFREQ_IN
        label = {1: "on_task", 2: "mind_wandering", 3: "distracted_sensation"}.get(p["answer"])
        rows.append({
            "dataset": "osf3uszv", "subject": subject, "session": "ses-01", "group": group,
            "onset_s": onset, "prev_end_s": prev_end,
            "answer": p["answer"], "confidence": p["confidence"], "sleepiness": p["sleepiness"],
            "focus": p["focus"], "engagement": p["engagement"], "label": label, "fif": fif,
        })
        prev_end = p["last_response_sample"] / SFREQ_IN
    log(f"  {path.name}: {raw.n_times / TARGET_SFREQ / 60:.1f} min, {len(rows)} probes, "
        f"labels={pd.Series([r['label'] for r in rows]).value_counts().to_dict()}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2, help="parallel files (each needs ~2.5 GB RAM)")
    ap.add_argument("--files", nargs="*", help="subset of file names")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*_task_*.txt.cpgz"))
    if args.files:
        files = [f for f in files if f.name in args.files]
    log(f"{len(files)} task files")
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(process_file, files):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probes.csv", index=False)
    log(df.groupby(["group", "label"]).size())
    return 0


if __name__ == "__main__":
    sys.exit(main())
