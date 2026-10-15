"""Per-segment, per-channel artifact rejection and window-level channel/group selection.

Segment level (2 s, every channel independently), following the analysis
spec's ``artifact_rejection``:

* peak-to-peak > ``per_channel_peak_to_peak_max_uv`` (blinks, movement),
* std < ``flat_std_min_uv`` (electrode off, saturation),
* optional EMG: log10 power in ``emg_band_hz`` above ``emg_log10_power_max``
  (jaw / temporalis; the analysis warns TP9/TP10 may be EMG-dominated on Muse),
* device markers: ``blink`` drops the frontal channels of the segment,
  ``jaw_clench`` drops all channels,
* horseshoe fit worse than the group's ``max_fit`` drops the channel.

Window level: a channel is usable if at least
``min_fraction_segments_kept_per_channel`` of its segments survived; a group
is usable if any of its channels is.  When the temporal group is unusable the
scorer falls back to frontal-only features and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import FeatureSpec, band_mean


@dataclass
class SegmentCheck:
    keep: np.ndarray  # (n_channels,) bool
    reasons: dict[int, list[str]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def check_segment(
    x: np.ndarray,
    spec: FeatureSpec,
    f: np.ndarray | None = None,
    psd: np.ndarray | None = None,
    fit: list[float] | None = None,
    markers: list[str] | None = None,
) -> SegmentCheck:
    """``x`` is (4, n) raw EEG with any DC offset; ``psd`` (4, F) if already computed."""
    cfg = spec.artifact
    n_ch = x.shape[0]
    keep = np.ones(n_ch, dtype=bool)
    reasons: dict[int, list[str]] = {i: [] for i in range(n_ch)}
    xd = x - x.mean(axis=1, keepdims=True)
    ptp = xd.max(axis=1) - xd.min(axis=1)
    std = xd.std(axis=1)
    ptp_max = float(cfg.get("per_channel_peak_to_peak_max_uv", 150.0))
    flat_min = float(cfg.get("flat_std_min_uv", 0.5))
    for i in range(n_ch):
        if ptp[i] > ptp_max:
            keep[i] = False
            reasons[i].append("ptp")
        if std[i] < flat_min:
            keep[i] = False
            reasons[i].append("flat")
    emg_band = cfg.get("emg_band_hz")
    emg_max = cfg.get("emg_log10_power_max")
    metrics: dict[str, float] = {"ptp_max": float(ptp.max()), "std_min": float(std.min())}
    if emg_band and emg_max is not None and f is not None and psd is not None:
        for i in range(n_ch):
            e = np.log10(max(band_mean(f, psd[i], tuple(emg_band)), 1e-12))
            if e > float(emg_max):
                keep[i] = False
                reasons[i].append("emg")
        metrics["emg_log10_max"] = float(max(np.log10(max(band_mean(f, psd[i], tuple(emg_band)), 1e-12)) for i in range(n_ch)))
    frontal = spec.groups.get("frontal", [])
    for m in markers or []:
        if m == "blink":
            for i in frontal:
                keep[i] = False
                reasons[i].append("blink")
        elif m == "jaw_clench":
            for i in range(n_ch):
                keep[i] = False
                reasons[i].append("jaw")
    if fit is not None and len(fit) >= n_ch:
        for g, idx in spec.groups.items():
            for i in idx:
                if fit[i] > spec.group_max_fit[g]:
                    keep[i] = False
                    reasons[i].append("fit")
    return SegmentCheck(keep, {i: r for i, r in reasons.items() if r}, metrics)


def usable_channels(kept_fraction: np.ndarray, spec: FeatureSpec) -> dict[str, list[int]]:
    """Channels per group whose kept fraction over the window clears the spec minimum."""
    min_frac = float(spec.artifact.get("min_fraction_segments_kept_per_channel", 0.5))
    out: dict[str, list[int]] = {}
    for g, idx in spec.groups.items():
        ok = [i for i in idx if kept_fraction[i] >= min_frac]
        if ok:
            out[g] = ok
    return out
