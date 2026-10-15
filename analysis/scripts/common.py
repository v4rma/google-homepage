"""Shared paths, channel maps and helpers for the FA feature analysis."""
from __future__ import annotations

import os
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FA_DATA_DIR", ANALYSIS_DIR / "data"))
DERIV_DIR = Path(os.environ.get("FA_DERIV_DIR", ANALYSIS_DIR / "derivatives"))
OUTPUT_DIR = ANALYSIS_DIR / "output"

for _d in (DATA_DIR, DERIV_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TARGET_SFREQ = 256.0

# Muse Athena electrode sites and the closest sites available in each source montage.
# Frontal  = AF7/AF8 on Muse.
# Temporal = TP9/TP10 on Muse (behind the ear, roughly over the mastoid).
MUSE_PROXIES = {
    "ds001787": {  # BioSemi 64 (10-10 names)
        "frontal": ["Fp1", "Fp2", "AF7", "AF8"],
        "temporal": ["T7", "T8", "TP7", "TP8", "P7", "P8", "P9", "P10"],
        "reference": "Fpz",
    },
    "osf3uszv": {  # 19-channel 10-20 + ear electrodes A1/A2
        "frontal": ["FP1", "FP2", "F7", "F8"],
        "temporal": ["T3", "T4", "T5", "T6", "A1", "A2"],
        "reference": None,  # no Fpz recorded; approximated as mean(FP1, FP2)
    },
}

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Pre-probe window lengths (seconds) evaluated in the analysis.
WINDOWS_S = (5.0, 15.0, 30.0, 60.0)

# BioSemi 64 default A/B labels -> 10-10 names (used only if the BDF lacks real labels).
BIOSEMI64_AB_TO_1010 = {
    "A1": "Fp1", "A2": "AF7", "A3": "AF3", "A4": "F1", "A5": "F3", "A6": "F5", "A7": "F7",
    "A8": "FT7", "A9": "FC5", "A10": "FC3", "A11": "FC1", "A12": "C1", "A13": "C3", "A14": "C5",
    "A15": "T7", "A16": "TP7", "A17": "CP5", "A18": "CP3", "A19": "CP1", "A20": "P1", "A21": "P3",
    "A22": "P5", "A23": "P7", "A24": "P9", "A25": "PO7", "A26": "PO3", "A27": "O1", "A28": "Iz",
    "A29": "Oz", "A30": "POz", "A31": "Pz", "A32": "CPz", "B1": "Fpz", "B2": "Fp2", "B3": "AF8",
    "B4": "AF4", "B5": "AFz", "B6": "Fz", "B7": "F2", "B8": "F4", "B9": "F6", "B10": "F8",
    "B11": "FT8", "B12": "FC6", "B13": "FC4", "B14": "FC2", "B15": "FCz", "B16": "Cz", "B17": "C2",
    "B18": "C4", "B19": "C6", "B20": "T8", "B21": "TP8", "B22": "CP6", "B23": "CP4", "B24": "CP2",
    "B25": "P2", "B26": "P4", "B27": "P6", "B28": "P8", "B29": "P10", "B30": "PO8", "B31": "PO4",
    "B32": "O2",
}


def log(*args):
    print(*args, flush=True)
