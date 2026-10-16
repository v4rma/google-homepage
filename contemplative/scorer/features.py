"""Feature spec loading and spectral feature computation.

The spec format is the one exported by the data analysis
(``spec/fa_on_task_score.json``; see docs/fa-feature-analysis.md in the
project store).  Key fields we consume::

    sampling_rate_hz
    preprocessing.segment_s / segment_overlap        2-s segments, 50 % overlap
    preprocessing.artifact_rejection.*                per-channel ptp / flat / (optional) EMG thresholds,
                                                      min_fraction_segments_kept_per_channel
    window.length_s / window.hop_s                    scoring window (60 s / 5 s in the analysis spec)
    channel_groups.<group>.muse_channels              frontal = AF7/AF8, temporal = TP9/TP10
    channel_groups.<group>.max_fit                    (optional, ours) worst acceptable horseshoe value
    features[]: name, channel_group, type, band_hz | numerator_hz+denominator_hz | fit_range_hz | search_range_hz,
                default_weight_on_task, by_expertise.<profile>.weight_on_task, calibrate_sign (optional, ours)

Feature types: ``log10_bandpower``, ``log10_ratio``, ``aperiodic_exponent``
(log-log linear fit; exponent = -slope, so steeper 1/f = larger value) and
``individual_alpha_frequency`` (peak of the spectrum above the aperiodic fit).

A profile ("novice" | "expert" | "default") selects the weights.  Features
whose on-task direction differs between the two expertise profiles (alpha and
theta/alpha ratios) have their sign chosen by calibration; this can be forced
with ``calibrate_sign``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import welch

SPEC_DIR = Path(__file__).parent / "spec"
DEFAULT_SPEC = SPEC_DIR / "fa_on_task_score.json"
FALLBACK_SPEC = SPEC_DIR / "fa_frontal_theta_alpha_v1.json"
CHANNELS = ("TP9", "AF7", "AF8", "TP10")
PROFILES = ("novice", "expert", "default")


@dataclass
class FeatureDef:
    name: str
    group: str
    type: str
    params: dict[str, Any]
    weight: float
    calibrate_sign: bool
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def sign(self) -> float:
        return 1.0 if self.weight >= 0 else -1.0


class FeatureSpec:
    def __init__(self, raw: dict[str, Any], profile: str = "novice") -> None:
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        self.raw = raw
        self.profile = profile
        self.name: str = raw["name"]
        self.version: str = raw.get("version", "")
        self.fs = float(raw.get("sampling_rate_hz", 256))
        pre = raw.get("preprocessing", {})
        self.segment_s = float(pre.get("segment_s", 2.0))
        self.segment_overlap = float(pre.get("segment_overlap", 0.5))
        self.artifact: dict[str, Any] = dict(pre.get("artifact_rejection", {}))
        win = raw.get("window", {})
        self.window_s = float(win.get("length_s", 60.0))
        self.hop_s = float(win.get("hop_s", 5.0))
        self.groups: dict[str, list[int]] = {}
        self.group_max_fit: dict[str, float] = {}
        for g, cfg in raw.get("channel_groups", {}).items():
            self.groups[g] = [CHANNELS.index(c) for c in cfg["muse_channels"]]
            # temporal electrodes must be well seated (horseshoe 1); frontal may be medium (2)
            self.group_max_fit[g] = float(cfg.get("max_fit", 1 if g == "temporal" else 2))
        self.features: list[FeatureDef] = [self._parse_feature(f) for f in raw["features"]]
        self.tolerant = False

    def apply_tolerant(self, ptp_uv: float = 300.0) -> "FeatureSpec":
        """Dry-electrode mode (``--fit-tolerant``): accept medium horseshoe fit on every
        group and raise the peak-to-peak rejection threshold."""
        for g in self.group_max_fit:
            self.group_max_fit[g] = max(self.group_max_fit[g], 2.0)
        self.artifact["per_channel_peak_to_peak_max_uv"] = max(float(self.artifact.get("per_channel_peak_to_peak_max_uv", 150.0)), ptp_uv)
        self.tolerant = True
        return self

    # -- parsing -------------------------------------------------------------------
    def _parse_feature(self, f: dict[str, Any]) -> FeatureDef:
        by = f.get("by_expertise", {})
        if self.profile in by and "weight_on_task" in by[self.profile]:
            w = float(by[self.profile]["weight_on_task"])
        else:
            w = float(f.get("default_weight_on_task", f.get("weight", 1.0)))
        dirs = {p: by[p].get("on_task_direction") for p in by}
        differs = len({d for d in dirs.values() if d}) > 1
        cal_sign = bool(f.get("calibrate_sign", differs))
        t = f["type"]
        if t == "log10_bandpower":
            params = {"band": tuple(map(float, f["band_hz"]))}
        elif t == "log10_ratio":
            params = {"num": tuple(map(float, f["numerator_hz"])), "den": tuple(map(float, f["denominator_hz"]))}
        elif t == "aperiodic_exponent":
            params = {"range": tuple(map(float, f.get("fit_range_hz", [2.0, 30.0])))}
        elif t == "individual_alpha_frequency":
            params = {"range": tuple(map(float, f.get("search_range_hz", [7.0, 13.0]))), "fit_range": (2.0, 30.0)}
        else:
            raise ValueError(f"unknown feature type {t!r} in spec {self.name}")
        return FeatureDef(f["name"], f["channel_group"], t, params, w, cal_sign, f)

    # -- convenience -----------------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def weights(self) -> dict[str, float]:
        return {f.name: f.weight for f in self.features}

    @property
    def active_features(self) -> list[FeatureDef]:
        return [f for f in self.features if f.weight != 0.0]

    def with_profile(self, profile: str) -> "FeatureSpec":
        other = FeatureSpec(self.raw, profile)
        other.window_s, other.hop_s = self.window_s, self.hop_s
        if self.tolerant:
            other.apply_tolerant(float(self.artifact["per_channel_peak_to_peak_max_uv"]))
        return other

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "profile": self.profile,
            "window_s": self.window_s,
            "hop_s": self.hop_s,
            "groups": {g: [CHANNELS[i] for i in idx] for g, idx in self.groups.items()},
            "max_fit": self.group_max_fit,
            "ptp_max_uv": self.artifact.get("per_channel_peak_to_peak_max_uv", 150.0),
            "tolerant": self.tolerant,
            "features": [{"name": f.name, "group": f.group, "type": f.type, "weight": f.weight, "calibrate_sign": f.calibrate_sign} for f in self.features],
        }


def load_spec(path: str | Path | None = None, profile: str = "novice", allow_fallback: bool = True) -> FeatureSpec:
    """Load a spec; with no path, the shipped analysis spec, falling back to the simple frontal spec."""
    p = Path(path) if path else DEFAULT_SPEC
    try:
        with open(p) as f:
            return FeatureSpec(json.load(f), profile)
    except (OSError, ValueError, KeyError) as e:
        if path is None and allow_fallback and p != FALLBACK_SPEC:
            print(f"[scorer] default spec unusable ({e!r}); using fallback {FALLBACK_SPEC.name}")
            with open(FALLBACK_SPEC) as f:
                return FeatureSpec(json.load(f), profile)
        raise


# -- spectra -----------------------------------------------------------------------------


def segment_psd(x: np.ndarray, fs: float, nperseg_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (uV^2/Hz) of a (n_channels, n_samples) segment; Hann, 50 % overlap, linear detrend."""
    x = np.atleast_2d(x)
    nperseg = int(min(x.shape[1], round(nperseg_s * fs)))
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear", window="hann", axis=-1)
    return f, pxx


def band_mean(f: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    m = (f >= band[0]) & (f < band[1])
    return float(np.mean(psd[m])) if m.any() else float("nan")


def aperiodic_fit(f: np.ndarray, psd: np.ndarray, rng: tuple[float, float]) -> tuple[float, float]:
    """Return (exponent, offset) of log10(psd) = offset - exponent * log10(f) over ``rng``."""
    m = (f >= rng[0]) & (f <= rng[1]) & (f > 0) & np.isfinite(psd) & (psd > 0)
    if m.sum() < 4:
        return float("nan"), float("nan")
    lf, lp = np.log10(f[m]), np.log10(psd[m])
    slope, offset = np.polyfit(lf, lp, 1)
    return float(-slope), float(offset)


def individual_alpha_frequency(f: np.ndarray, psd: np.ndarray, rng: tuple[float, float], fit_range: tuple[float, float]) -> float:
    """Frequency of the largest peak above the aperiodic fit inside ``rng``; NaN if nothing sticks out."""
    exp_, off = aperiodic_fit(f, psd, fit_range)
    if not math.isfinite(exp_):
        return float("nan")
    m = (f >= rng[0]) & (f <= rng[1]) & (f > 0)
    if not m.any():
        return float("nan")
    resid = np.log10(np.maximum(psd[m], 1e-12)) - (off - exp_ * np.log10(f[m]))
    i = int(np.argmax(resid))
    return float(f[m][i]) if resid[i] > 0.05 else float("nan")


def feature_value(fd: FeatureDef, f: np.ndarray, psd: np.ndarray) -> float:
    if fd.type == "log10_bandpower":
        return float(np.log10(max(band_mean(f, psd, fd.params["band"]), 1e-12)))
    if fd.type == "log10_ratio":
        return float(np.log10(max(band_mean(f, psd, fd.params["num"]), 1e-12) / max(band_mean(f, psd, fd.params["den"]), 1e-12)))
    if fd.type == "aperiodic_exponent":
        return aperiodic_fit(f, psd, fd.params["range"])[0]
    if fd.type == "individual_alpha_frequency":
        return individual_alpha_frequency(f, psd, fd.params["range"], fd.params["fit_range"])
    raise ValueError(fd.type)


def compute_features(spec: FeatureSpec, f: np.ndarray, group_psd: dict[str, np.ndarray]) -> dict[str, float]:
    """Features for every spec feature whose group has a PSD; missing groups give NaN."""
    out: dict[str, float] = {}
    for fd in spec.features:
        psd = group_psd.get(fd.group)
        out[fd.name] = feature_value(fd, f, psd) if psd is not None else float("nan")
    return out
