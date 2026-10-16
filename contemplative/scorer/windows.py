"""Rolling buffer of per-segment PSDs and window aggregation.

Every ``hop`` (= segment_s * (1 - overlap), 1 s for 2-s segments at 50 %) the
latest segment is checked per channel and its PSD stored.  A window is the
set of segments whose end time falls in ``(t0, t1]``; the group PSD is the
mean over kept (channel, segment) pairs of the group's usable channels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..ingest.base import Packet
from .artifacts import check_segment, usable_channels
from .features import CHANNELS, FeatureSpec, compute_features, segment_psd


@dataclass
class Segment:
    t_end: float
    psd: np.ndarray  # (4, F)
    keep: np.ndarray  # (4,) bool
    fit: list[float]
    reasons: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class WindowFeatures:
    t0: float
    t1: float
    n_segments: int
    features: dict[str, float]
    groups_used: list[str]
    kept_fraction: dict[str, float]  # per channel name
    fit: list[float] | None  # latest horseshoe; None when no horseshoe message has arrived
    fallback_frontal_only: bool
    reasons: list[str] = field(default_factory=list)
    raw_offset: list[float] | None = None
    raw_scale: float = 1.0

    @property
    def valid(self) -> bool:
        return bool(self.groups_used) and any(np.isfinite(v) for v in self.features.values())

    def quality(self) -> dict:
        tp = [self.kept_fraction.get("TP9", 0.0), self.kept_fraction.get("TP10", 0.0)]
        af = [self.kept_fraction.get("AF7", 0.0), self.kept_fraction.get("AF8", 0.0)]
        return {
            "groups_used": self.groups_used,
            "fallback_frontal_only": self.fallback_frontal_only,
            "kept_fraction": {k: round(v, 2) for k, v in self.kept_fraction.items()},
            "temporal_kept": round(float(np.mean(tp)), 2),
            "frontal_kept": round(float(np.mean(af)), 2),
            "fit": self.fit,
            "fit_known": self.fit is not None,
            "n_segments": self.n_segments,
            "raw_offset": None if self.raw_offset is None else [round(v, 1) for v in self.raw_offset],
            "raw_scale": self.raw_scale,
        }


class SegmentBuffer:
    def __init__(self, spec: FeatureSpec, max_history_s: float = 900.0) -> None:
        self.spec = spec
        self.seg_n = int(round(spec.segment_s * spec.fs))
        self.hop_s = spec.segment_s * (1.0 - spec.segment_overlap)
        self._raw: deque[list[float]] = deque(maxlen=self.seg_n + 8)
        self._t: deque[float] = deque(maxlen=self.seg_n + 8)
        self._next_seg: float | None = None
        self._fit: list[float] | None = None  # unknown until a horseshoe message arrives
        self.n_horseshoe = 0
        self._markers: deque[tuple[float, str]] = deque(maxlen=128)
        self.segments: deque[Segment] = deque(maxlen=int(max_history_s / self.hop_s) + 1)
        self.f: np.ndarray | None = None
        self.n_segments = 0
        # raw-unit handling: Mind Monitor sends microvolts offset by ~800; if the
        # stream is clearly in volts (MAD < 0.01) it is rescaled by 1e6.  Both the
        # estimated offset and the scale are reported, never assumed.
        self.raw_offset: list[float] | None = None
        self.raw_scale: float = 1.0
        self.scale_note: str | None = None

    @property
    def fit(self) -> list[float] | None:
        return None if self._fit is None else list(self._fit)

    @property
    def fit_known(self) -> bool:
        return self._fit is not None

    def push(self, packets: list[Packet]) -> list[Segment]:
        """Consume packets; return newly completed segments."""
        new: list[Segment] = []
        for p in packets:
            if p.kind == "eeg":
                self._raw.append(p.data[:4])
                self._t.append(p.t)
                if self._next_seg is None:
                    self._next_seg = p.t + self.spec.segment_s
                elif p.t >= self._next_seg:
                    seg = self._make_segment(p.t)
                    if seg is not None:
                        new.append(seg)
                    self._next_seg += self.hop_s
                    if p.t - self._next_seg > 2 * self.hop_s:  # dropped packets / replay jump
                        self._next_seg = p.t + self.hop_s
            elif p.kind == "horseshoe" and len(p.data) >= 4:
                self._fit = [float(v) for v in p.data[:4]]
                self.n_horseshoe += 1
            elif p.kind in ("blink", "jaw_clench"):
                self._markers.append((p.t, p.kind))
        return new

    def _make_segment(self, t_end: float) -> Segment | None:
        if len(self._raw) < self.seg_n:
            return None
        x = np.array(list(self._raw)[-self.seg_n :], dtype=float).T
        if self.raw_offset is None:
            self._detect_units(x)
        x = x * self.raw_scale
        t_start = self._t[-self.seg_n]
        pad = 0.25
        marks = [m for (tm, m) in self._markers if t_start - pad <= tm <= t_end + pad]
        f, psd = segment_psd(x, self.spec.fs)
        self.f = f
        chk = check_segment(x, self.spec, f, psd, self._fit, marks)
        seg = Segment(t_end, psd, chk.keep, self.fit, chk.reasons)
        self.segments.append(seg)
        self.n_segments += 1
        return seg

    def _detect_units(self, x: np.ndarray) -> None:
        med = np.nanmedian(x, axis=1)
        mad = float(np.nanmedian(np.abs(x - med[:, None])))
        self.raw_offset = [float(v) for v in med]
        if np.isfinite(mad) and 0.0 < mad < 0.01:
            self.raw_scale = 1e6
            self.scale_note = f"raw values look like volts (median abs deviation {mad:.2e}); rescaled by 1e6 to microvolts"
        elif np.isfinite(mad) and mad > 5000:
            self.scale_note = f"raw values have an unusually large spread (median abs deviation {mad:.0f}); check units"
        elif not (400 < float(np.nanmedian(med)) < 1200):
            self.scale_note = f"raw EEG is not centred near Mind Monitor's ~800 uV (median {float(np.nanmedian(med)):.1f}); offset is removed per segment, so this is informational"

    def window(self, t0: float, t1: float) -> WindowFeatures | None:
        """Aggregate segments ending in (t0, t1]."""
        segs = [s for s in self.segments if t0 < s.t_end <= t1]
        if not segs or self.f is None:
            return None
        keep = np.array([s.keep for s in segs])  # (S, 4)
        frac = keep.mean(axis=0)
        usable = usable_channels(frac, self.spec)
        group_psd: dict[str, np.ndarray] = {}
        for g, idx in usable.items():
            acc = []
            for s in segs:
                for i in idx:
                    if s.keep[i]:
                        acc.append(s.psd[i])
            if acc:
                group_psd[g] = np.mean(acc, axis=0)
        feats = compute_features(self.spec, self.f, group_psd)
        groups_used = sorted(group_psd)
        fallback = "temporal" in self.spec.groups and "temporal" not in group_psd and "frontal" in group_psd
        reasons: list[str] = []
        if not group_psd:
            reasons.append("no_usable_channels")
        if fallback:
            reasons.append("temporal_unusable")
        return WindowFeatures(
            t0=t0,
            t1=t1,
            n_segments=len(segs),
            features=feats,
            groups_used=groups_used,
            kept_fraction={CHANNELS[i]: float(frac[i]) for i in range(4)},
            fit=segs[-1].fit,
            fallback_frontal_only=fallback,
            reasons=reasons,
            raw_offset=self.raw_offset,
            raw_scale=self.raw_scale,
        )
