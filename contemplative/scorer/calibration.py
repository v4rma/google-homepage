"""Per-wearer calibration: feature scale (z-scoring), sign selection and profile.

Protocol: eyes-open fixation on the uniform isoluminant field (2 min, the
"mind-wandering-like" reference) then breath focus (2 min), repeated.  From
the window features of both conditions we fit, per feature:

    mean / std   over all calibration windows               -> z-scoring (the per-feature scale)
    d            Cohen's d of breath - fixation               -> quality / sign evidence
    sign         for sign-ambiguous features (alpha, theta/alpha ratios; those whose
                 on-task direction differs between expert and novice profiles) the
                 sign of d when |d| >= min_abs_d, else the profile's sign;
                 for the others the profile's sign (the analysis found them
                 expertise-independent); a strong contradiction is reported in ``notes``.

Profile: ``novice`` by default (analysis recommendation), ``expert`` if the
wearer asks, or ``auto`` = whichever profile's weights separate the two
calibration conditions better.

Score = sum_i |w_i| * sign_i * z_i / sum_i |w_i| over the features available
in a window, then standardised so fixation windows have mean 0 / sd 1.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .features import FeatureSpec

# Minimum per-feature scale used for z-scoring, by feature type.
MIN_STD = {"individual_alpha_frequency": 0.25, "aperiodic_exponent": 0.02, "log10_bandpower": 0.02, "log10_ratio": 0.02, "default": 0.02}


@dataclass
class Calibration:
    spec_name: str
    spec_version: str
    profile: str
    wearer: str
    created: float
    feature_mean: dict[str, float]
    feature_std: dict[str, float]
    feature_sign: dict[str, float]
    feature_d: dict[str, float]
    weights: dict[str, float]  # |w| used in the combination
    score_mean: float
    score_std: float
    n_fixation: int
    n_breath: int
    separation_d: float
    window_s: float
    notes: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def raw_score(self, feats: dict[str, float]) -> float | None:
        num = 0.0
        den = 0.0
        for name, w in self.weights.items():
            v = feats.get(name)
            if v is None or not math.isfinite(v) or w == 0.0:
                continue
            z = (v - self.feature_mean[name]) / max(self.feature_std[name], 1e-9)
            num += abs(w) * self.feature_sign[name] * z
            den += abs(w)
        return num / den if den > 0 else None

    def score(self, feats: dict[str, float]) -> float | None:
        r = self.raw_score(feats)
        return None if r is None else (r - self.score_mean) / max(self.score_std, 1e-9)

    def summary(self) -> dict:
        return {
            "created": self.created,
            "profile": self.profile,
            "separation_d": round(self.separation_d, 3),
            "signs": self.feature_sign,
            "d": {k: round(v, 3) for k, v in self.feature_d.items()},
            "n": [self.n_fixation, self.n_breath],
            "window_s": self.window_s,
            "notes": self.notes,
        }


class CalibrationFitter:
    def __init__(self, spec: FeatureSpec, wearer: str = "default", min_abs_d: float = 0.2, profile: str = "novice") -> None:
        self.spec = spec
        self.wearer = wearer
        self.min_abs_d = min_abs_d
        self.profile = profile  # "novice" | "expert" | "default" | "auto"
        self.fixation: list[dict[str, float]] = []
        self.breath: list[dict[str, float]] = []

    def add(self, condition: str, feats: dict[str, float]) -> None:
        if condition == "calibration_fixation":
            self.fixation.append(feats)
        elif condition == "calibration_breath":
            self.breath.append(feats)
        else:
            raise ValueError(condition)

    @property
    def n(self) -> tuple[int, int]:
        return len(self.fixation), len(self.breath)

    def fit(self) -> Calibration:
        if len(self.fixation) < 5 or len(self.breath) < 5:
            raise ValueError(f"not enough valid calibration windows (fixation={len(self.fixation)}, breath={len(self.breath)})")
        if self.profile == "auto":
            cands = [self._fit_profile(p) for p in ("novice", "expert")]
            best = max(cands, key=lambda c: c.separation_d)
            best.notes.append("profile chosen automatically: " + ", ".join(f"{c.profile} d={c.separation_d:.2f}" for c in cands))
            return best
        return self._fit_profile(self.profile)

    def _fit_profile(self, profile: str) -> Calibration:
        spec = self.spec if self.spec.profile == profile else self.spec.with_profile(profile)
        names = spec.feature_names
        F = np.array([[e.get(n, np.nan) for n in names] for e in self.fixation], dtype=float)
        B = np.array([[e.get(n, np.nan) for n in names] for e in self.breath], dtype=float)
        A = np.vstack([F, B])
        # a group that was unusable throughout calibration (e.g. temporal in frontal-only
        # fallback) gives all-NaN columns; those features get the floor scale and d = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(A, axis=0)
            std = np.nanstd(A, axis=0, ddof=1)
            # floor the scale so a feature that barely moved in calibration (e.g. a
            # quantised alpha frequency) cannot explode into huge z-scores later
            floor = np.array([MIN_STD.get(fd.type, MIN_STD["default"]) for fd in spec.features])
            std = np.where(np.isfinite(std), np.maximum(std, floor), floor)
            pooled = np.sqrt((np.nanvar(F, axis=0, ddof=1) + np.nanvar(B, axis=0, ddof=1)) / 2) + 1e-9
            d = (np.nanmean(B, axis=0) - np.nanmean(F, axis=0)) / pooled
            d = np.where(np.isfinite(d), d, 0.0)
        notes: list[str] = []
        sign = np.zeros(len(names))
        weights: dict[str, float] = {}
        for i, fd in enumerate(spec.features):
            weights[fd.name] = abs(fd.weight)
            if fd.weight == 0.0:
                sign[i] = fd.sign
                continue
            if fd.calibrate_sign:
                if abs(d[i]) >= self.min_abs_d:
                    sign[i] = math.copysign(1.0, d[i])
                    if sign[i] != fd.sign:
                        notes.append(f"{fd.name}: calibration sign {sign[i]:+.0f} overrides profile sign {fd.sign:+.0f} (d={d[i]:+.2f})")
                else:
                    sign[i] = fd.sign
                    notes.append(f"{fd.name}: weak contrast (d={d[i]:+.2f}); keeping profile sign {fd.sign:+.0f}")
            else:
                sign[i] = fd.sign
                if abs(d[i]) >= 0.5 and math.copysign(1.0, d[i]) != fd.sign:
                    notes.append(f"{fd.name}: calibration contrast (d={d[i]:+.2f}) contradicts the fixed sign {fd.sign:+.0f}; kept")
        cal = Calibration(
            spec_name=spec.name,
            spec_version=spec.version,
            profile=profile,
            wearer=self.wearer,
            created=time.time(),
            feature_mean={n: float(mean[i]) if np.isfinite(mean[i]) else 0.0 for i, n in enumerate(names)},
            feature_std={n: float(std[i]) for i, n in enumerate(names)},
            feature_sign={n: float(sign[i]) for i, n in enumerate(names)},
            feature_d={n: float(d[i]) for i, n in enumerate(names)},
            weights=weights,
            score_mean=0.0,
            score_std=1.0,
            n_fixation=len(F),
            n_breath=len(B),
            separation_d=0.0,
            window_s=spec.window_s,
            notes=notes,
        )
        raw_F = np.array([v for v in (cal.raw_score(e) for e in self.fixation) if v is not None])
        raw_B = np.array([v for v in (cal.raw_score(e) for e in self.breath) if v is not None])
        cal.score_mean = float(raw_F.mean())
        cal.score_std = float(max(raw_F.std(ddof=1), 1e-9))
        cal.separation_d = float((raw_B.mean() - raw_F.mean()) / (np.sqrt((raw_F.var(ddof=1) + raw_B.var(ddof=1)) / 2) + 1e-9))
        return cal
