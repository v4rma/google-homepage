"""Offline check that the backend's parameter schedule contains no 3-30 Hz
content (report Section 2: avoid anything periodic in the photic-driving /
SSVEP band).

Two things can put temporal structure on screen:

1. what the backend sends: step changes of the parameter vector and of the
   baseline/stimulus mix.  The page never applies these directly; it morphs
   with a smoothstep over ``morph_s`` (>= 10 s).  This module reconstructs
   that schedule for a whole trial sequence, samples it at display rate and
   measures its spectrum.
2. what the shader does on its own: a 0.1 Hz pulsation and a slowly
   advancing 3-D noise field.  That part is measured on the real rendering by
   the page's ``?diag=1`` self-check (readPixels luminance spectrum), see
   docs/temporal-spectrum-check.md.

``check_morph_schedule`` covers 1.  A smoothstep of duration T is C1, so its
spectrum falls off as f^-3 beyond ~1/T; with T = 10 s the 3-30 Hz band is more
than 100 dB below the <1 Hz content.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .params import AXES, LITERATURE_PRIOR, StimulusParams, to_physical


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def schedule(
    morph_s: float = 10.0,
    baseline_s: float = 60.0,
    stimulus_s: float = 75.0,
    n_trials: int = 4,
    fs: float = 120.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, mix, params[n, 6]) for a sequence of baseline/stimulus trials
    exactly as the visuals page would morph them (initial state: uniform field)."""
    rng = np.random.default_rng(seed)
    morph_s = max(float(morph_s), 10.0)
    total = n_trials * (baseline_s + stimulus_s) + baseline_s
    t = np.arange(0.0, total, 1.0 / fs)
    mix = np.ones_like(t)  # 1 = uniform baseline field
    cur = np.array([LITERATURE_PRIOR[a] for a in AXES])
    params = np.tile(cur, (t.size, 1))
    events = []  # (time, mix_target, param_target)
    for k in range(n_trials):
        t_stim = baseline_s + k * (baseline_s + stimulus_s)
        x = cur if k == 0 else rng.uniform(0.0, 1.0, len(AXES))
        events.append((t_stim, 0.0, x))
        events.append((t_stim + stimulus_s, 1.0, x))
    mix_from, par_from = 1.0, cur.copy()
    for t_ev, mix_to, par_to in events:
        u = _smoothstep((t - t_ev) / morph_s)
        sel = t >= t_ev
        mix[sel] = mix_from + (mix_to - mix_from) * u[sel]
        params[sel] = par_from + (par_to - par_from) * u[sel, None]
        mix_from, par_from = mix_to, np.asarray(par_to, float)
    return t, mix, params


def band_shares(x: np.ndarray, fs: float) -> dict[str, float]:
    f, p = welch(x - x.mean(), fs=fs, nperseg=int(fs * 40), detrend=False)
    tot = float(np.trapezoid(p, f)) + 1e-30
    band = (f >= 3.0) & (f <= 30.0)
    low = f < 1.0
    p_band = float(np.trapezoid(p[band], f[band])) if band.any() else 0.0
    p_low = float(np.trapezoid(p[low], f[low]))
    peak_i = int(np.argmax(np.where(band, p, -np.inf))) if band.any() else 0
    return {
        "share_3_30hz": p_band / tot,
        "share_below_1hz": p_low / tot,
        "band_to_low_db": 10.0 * np.log10((p_band + 1e-30) / (p_low + 1e-30)),
        "peak_in_band_hz": float(f[peak_i]),
        "peak_in_band_rel_db": 10.0 * np.log10((p[peak_i] + 1e-30) / (p.max() + 1e-30)),
    }


def check_morph_schedule(morph_s: float = 10.0, max_band_share: float = 1e-6, min_margin_db: float = 60.0, **kw) -> dict:
    """Spectrum of the morph schedule the backend drives; ``ok`` when the
    3-30 Hz share of the fluctuation power is below ``max_band_share`` and at
    least ``min_margin_db`` below the <1 Hz content for the baseline/stimulus mix
    and every parameter axis.  Also reports the breath pulsation (pure 0.1 Hz)."""
    fs = float(kw.pop("fs", 120.0))
    t, mix, params = schedule(morph_s=morph_s, fs=fs, **kw)
    out: dict = {"morph_s": max(float(morph_s), 10.0), "fs": fs, "signals": {}}
    ok = True
    for name, x in [("mix", mix)] + [(a, params[:, i]) for i, a in enumerate(AXES)]:
        if np.ptp(x) < 1e-9:
            continue
        s = band_shares(x, fs)
        s["ok"] = bool(s["share_3_30hz"] <= max_band_share and s["band_to_low_db"] <= -min_margin_db)
        ok &= s["ok"]
        out["signals"][name] = s
    # breath pulsation as rendered (5 % radial zoom at full amplitude, 0.1 Hz)
    pulse = 1.0 + 0.05 * np.sin(2 * np.pi * 0.1 * t)
    s = band_shares(pulse, fs)
    s["ok"] = bool(s["share_3_30hz"] <= max_band_share)
    ok &= s["ok"]
    out["signals"]["breath_pulse"] = s
    out["ok"] = bool(ok)
    out["criteria"] = {"max_band_share": max_band_share, "min_margin_db": min_margin_db}
    out["physical_example"] = to_physical(StimulusParams.from_dict(LITERATURE_PRIOR))
    return out
