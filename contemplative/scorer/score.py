"""Real-time focused-attention scorer.

* Feeds on packets, builds 2-s segment PSDs (:class:`SegmentBuffer`).
* Every ``spec.hop_s`` emits a display score over the last ``spec.window_s``.
* :meth:`interval_score` scores an arbitrary interval (used for trial rewards
  and calibration blocks), so the reward uses one long window over the scored
  part of a phase rather than a median of short noisy windows.
* Without a calibration it uses the profile signs and a running z-score so the
  dashboard still shows something meaningful.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..ingest.base import Packet
from .calibration import Calibration
from .features import FeatureSpec
from .windows import SegmentBuffer, WindowFeatures


@dataclass
class ScoreSample:
    t: float
    score: float | None  # None when the window was unusable
    raw: float | None
    valid: bool
    reasons: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    fit: list[float] = field(default_factory=list)


class FocusScorer:
    def __init__(self, spec: FeatureSpec, calibration: Calibration | None = None) -> None:
        self.spec = spec
        self.calibration = calibration
        self.buffer = SegmentBuffer(spec)
        self._next_eval: float | None = None
        self._fallback_hist: deque[float] = deque(maxlen=200)
        self.n_valid = 0
        self.n_rejected = 0
        self.last_quality: dict = {}

    def set_calibration(self, cal: Calibration | None) -> None:
        self.calibration = cal

    @property
    def fit(self) -> list[float]:
        return self.buffer.fit

    # -- streaming -------------------------------------------------------------------
    def push(self, packets: list[Packet]) -> list[ScoreSample]:
        out: list[ScoreSample] = []
        for seg in self.buffer.push(packets):
            if self._next_eval is None:
                self._next_eval = seg.t_end + max(self.spec.window_s - self.spec.segment_s, 0.0)
            if seg.t_end >= self._next_eval:
                out.append(self._evaluate(seg.t_end))
                self._next_eval += self.spec.hop_s
                if seg.t_end - self._next_eval > 2 * self.spec.hop_s:
                    self._next_eval = seg.t_end + self.spec.hop_s
        return out

    def _evaluate(self, t: float) -> ScoreSample:
        wf = self.buffer.window(t - self.spec.window_s, t)
        return self._to_sample(t, wf)

    def interval_score(self, t0: float, t1: float) -> ScoreSample:
        """One score over all segments ending in (t0, t1] (trial rewards, calibration)."""
        return self._to_sample(t1, self.buffer.window(t0, t1))

    def interval_features(self, t0: float, t1: float) -> WindowFeatures | None:
        return self.buffer.window(t0, t1)

    # -- scoring ---------------------------------------------------------------------------
    def _to_sample(self, t: float, wf: WindowFeatures | None) -> ScoreSample:
        if wf is None:
            return ScoreSample(t, None, None, False, ["no_segments"], fit=self.fit)
        self.last_quality = wf.quality()
        if not wf.valid:
            self.n_rejected += 1
            return ScoreSample(t, None, None, False, wf.reasons or ["invalid"], wf.features, wf.quality(), wf.fit)
        if self.calibration is not None:
            raw = self.calibration.raw_score(wf.features)
            score = self.calibration.score(wf.features)
        else:
            raw = self._profile_raw(wf.features)
            score = None
            if raw is not None:
                self._fallback_hist.append(raw)
                if len(self._fallback_hist) >= 10:
                    h = np.array(self._fallback_hist)
                    score = float((raw - h.mean()) / max(h.std(), 1e-9))
                else:
                    score = 0.0
        if raw is None or score is None or not math.isfinite(score):
            self.n_rejected += 1
            return ScoreSample(t, None, None, False, wf.reasons + ["no_features"], wf.features, wf.quality(), wf.fit)
        self.n_valid += 1
        return ScoreSample(t, float(score), float(raw), True, wf.reasons, wf.features, wf.quality(), wf.fit)

    def _profile_raw(self, feats: dict[str, float]) -> float | None:
        """Uncalibrated: weighted sum with the profile's signs; features are not scaled, so this is only indicative."""
        num = den = 0.0
        for fd in self.spec.active_features:
            v = feats.get(fd.name)
            if v is None or not math.isfinite(v):
                continue
            num += fd.weight * v
            den += abs(fd.weight)
        return num / den if den > 0 else None
