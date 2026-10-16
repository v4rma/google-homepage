"""Synthetic Muse-like EEG whose spectral signature follows the FA analysis.

The source models a wearer whose "on-task" state ``s`` is

* 0 during the uniform-field baseline / calibration fixation (mind-wandering-like),
* 1 during calibration breath focus,
* ``GroundTruth(x)`` while a stimulus with parameters ``x`` is shown (reached
  through a ~5 s lag so there is an orienting transient to discard),

plus slow Ornstein-Uhlenbeck drift.  Following docs/fa-feature-analysis.md the
state is expressed mostly on the TEMPORAL channels (TP9/TP10):

* temporal 1/f is steeper during mind-wandering: a white component grows with ``s``
  so the aperiodic exponent falls when on-task;
* temporal delta rises during mind-wandering (amplitude ~ 1 + delta_gain * (1 - s));
* alpha (temporal and frontal) scales with ``1 + alpha_gain * s`` where the sign of
  ``alpha_gain`` depends on the profile: novices have more alpha while
  mind-wandering (negative), experts more alpha on task (positive);
* frontal beta rises during mind-wandering;
* frontal theta carries almost nothing (the textbook marker that did not replicate).

Blinks, jaw clenches and intermittent bad fit on the temporal electrodes are
added; horseshoe, PPG, heart-rate and dummy fNIRS packets are emitted so the
whole recording path is exercised.  The ground-truth function is hidden from
the optimizer; tests read it from ``SyntheticSource.ground_truth``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from ..visuals.params import AXES
from .base import FS, EEGSource, Packet

# Paul Kellet-style 1/f ("pinking") IIR filter.
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])

PROFILE_ALPHA_GAIN = {"novice": -0.45, "expert": 0.6}

# Stream defects the synthetic source can imitate (``defects=`` / ``--synthetic-defect``),
# used to exercise the diagnostics and the fail-soft paths without hardware.
DEFECTS = {
    "silent": "emit nothing at all (wrong OSC target / firewall)",
    "bands_only": "no raw /muse/eeg, only band-power messages (Mind Monitor raw streaming off)",
    "no_horseshoe": "never send horseshoe fit values",
    "no_offset": "raw EEG centred on 0 uV instead of Mind Monitor's ~800",
    "volts": "raw EEG in volts (1e-6 scale) instead of microvolts",
    "medium_fit": "horseshoe 2 (medium) on every channel, signal otherwise clean",
    "bad_fit": "temporal electrodes off (horseshoe 4 + noise on TP9/TP10)",
    "all_bad_fit": "every electrode off (horseshoe 4 + noise everywhere)",
    "dry_ptp": "slow electrode drift giving 150-300 uV peak-to-peak on all channels",
    "low_rate": "only every other sample arrives (~128 Hz)",
}


@dataclass
class GroundTruth:
    """Additive Gaussian bumps per axis, normalised so the optimum has value 1.

    Weights encode which axes matter (speed and fractal dimension most,
    colour and texture least).  The optimum is deliberately offset from the
    literature prior (f(prior) ~ 0.56) so the optimizer has to move.
    """

    centre: dict[str, float] = field(
        default_factory=lambda: {
            "speed": 0.60,
            "fractal": 0.75,  # D = 1.1 + 0.6 * 0.75 = 1.55
            "symmetry": 0.80,
            "color": 0.50,
            "breath": 0.20,
            "texture": 0.40,
        }
    )
    width: dict[str, float] = field(
        default_factory=lambda: {"speed": 0.31, "fractal": 0.28, "symmetry": 0.42, "color": 0.70, "breath": 0.42, "texture": 0.70}
    )
    weight: dict[str, float] = field(
        default_factory=lambda: {"speed": 1.0, "fractal": 1.0, "symmetry": 0.6, "color": 0.15, "breath": 0.5, "texture": 0.15}
    )

    @property
    def optimum(self) -> np.ndarray:
        return np.array([self.centre[a] for a in AXES])

    def __call__(self, x) -> float:
        if isinstance(x, dict):
            x = np.array([x[a] for a in AXES], dtype=float)
        x = np.asarray(x, dtype=float)
        total = 0.0
        wsum = 0.0
        for i, a in enumerate(AXES):
            d = (x[i] - self.centre[a]) / self.width[a]
            total += self.weight[a] * math.exp(-0.5 * d * d)
            wsum += self.weight[a]
        return total / wsum

    def relevant_axes(self, min_weight: float = 0.4) -> list[int]:
        return [i for i, a in enumerate(AXES) if self.weight[a] >= min_weight]

    def main_axes(self) -> list[int]:
        """Axes with the largest weight (the ones a 20-trial sit can be expected to localise)."""
        wmax = max(self.weight.values())
        return [i for i, a in enumerate(AXES) if self.weight[a] >= wmax - 1e-9]


class SyntheticSource(EEGSource):
    name = "synthetic"

    def __init__(
        self,
        seed: int = 0,
        profile: str = "novice",
        effect_scale: float = 1.0,
        alpha_gain: float | None = None,
        aperiodic_gain: float = 0.9,
        delta_gain: float = 0.75,
        beta_gain: float = 0.6,
        theta_gain: float = 0.05,
        state_noise: float = 0.05,
        drift_tau_s: float = 30.0,
        burstiness: float = 0.2,
        ground_truth: GroundTruth | None = None,
        blink_rate_hz: float = 0.12,
        jaw_rate_hz: float = 1 / 40,
        bad_fit_rate_hz: float = 1 / 60,
        fs: float = FS,
        defects: set[str] | list[str] | None = None,
        defect_until_s: float | None = None,
    ) -> None:
        super().__init__()
        self.defects = set(defects or ())
        unknown = self.defects - set(DEFECTS)
        if unknown:
            raise ValueError(f"unknown synthetic defects {sorted(unknown)}; known: {sorted(DEFECTS)}")
        self.defect_until_s = defect_until_s  # defects active only before this source time (None = always)
        self._decim = 0
        self._band_buf: deque[list[float]] = deque(maxlen=int(fs))
        self._next_band = 0.0
        self.rng = np.random.default_rng(seed)
        self.fs = fs
        self.profile = profile
        self.effect_scale = effect_scale
        self.alpha_gain = (PROFILE_ALPHA_GAIN[profile] if alpha_gain is None else alpha_gain) * effect_scale
        self.aperiodic_gain = aperiodic_gain * effect_scale
        self.delta_gain = delta_gain * effect_scale
        self.beta_gain = beta_gain * effect_scale
        self.theta_gain = theta_gain * effect_scale
        self.state_noise = state_noise
        self.drift_tau_s = drift_tau_s
        self.burstiness = burstiness
        self.ground_truth = ground_truth or GroundTruth()
        self.blink_rate_hz = blink_rate_hz
        self.jaw_rate_hz = jaw_rate_hz
        self.bad_fit_rate_hz = bad_fit_rate_hz

        self._t_start: float | None = None
        self._t_gen: float | None = None  # anchored to the first tick (virtual clocks start at 0, real ones do not)
        self._target_state = 0.0
        self._mode = "idle"
        self._state_lag = math.exp(-1.0 / (fs * 5.0))  # 5 s time constant
        self._s = 0.0
        self._ou = 0.0
        self._ou_rho = math.exp(-1.0 / (fs * drift_tau_s))

        self._pink_zi = [lfilter_zi(_PINK_B, _PINK_A) * 0.0 for _ in range(4)]
        # oscillators: delta, theta, alpha, beta
        self._osc_f0 = np.array([2.5, 6.0, 10.0, 20.0])
        self._osc_jit = np.array([0.4, 0.3, 0.3, 1.0])
        self._phase = np.zeros(4)
        self._env_zi = np.zeros(4)
        self._env_rho = math.exp(-1.0 / (fs * 1.5))

        self._blink_until = -1.0
        self._blink_start = -1.0
        self._jaw_until = -1.0
        self._bad_fit_until = [-1.0, -1.0, -1.0, -1.0]
        self._next_horseshoe = 0.0
        self._next_hr = 0.0
        self._next_fnirs = 0.0
        self._ppg_t = 0.0
        self._ppg_fs = 64.0
        self._hr_bpm = 66.0

    # -- orchestrator hooks ---------------------------------------------------
    def set_stimulus(self, params: dict[str, float] | None, mode: str) -> None:
        self._mode = mode
        if mode == "stimulus" and params is not None:
            self._target_state = float(self.ground_truth(params))
        elif mode == "calibration_breath":
            self._target_state = 1.0
        else:  # baseline, calibration_fixation, idle
            self._target_state = 0.0

    def describe(self):
        return {
            "name": self.name,
            "profile": self.profile,
            "effect_scale": self.effect_scale,
            "alpha_gain": self.alpha_gain,
            "aperiodic_gain": self.aperiodic_gain,
            "state_noise": self.state_noise,
            "drift_tau_s": self.drift_tau_s,
            "optimum": self.ground_truth.optimum.tolist(),
            "defects": sorted(self.defects),
        }

    def _defect(self, name: str, t: float) -> bool:
        """``defect_until_s`` counts from the first tick."""
        return name in self.defects and (self.defect_until_s is None or t - (self._t_start or 0.0) < self.defect_until_s)

    @property
    def state(self) -> float:
        return self._s

    def tick(self, now: float) -> None:
        if not self.running:
            return
        if self._t_gen is None:
            self._t_gen = self._t_start = now
            self._next_horseshoe = self._next_fnirs = self._next_hr = self._ppg_t = self._next_band = now
        # never generate more than 5 s per tick (a stalled loop must not flood the scorer)
        n = int(math.floor(min(now - self._t_gen, 5.0) * self.fs))
        if n > 0:
            self._generate(n)
            if now - self._t_gen > 5.0:
                self._t_gen = now

    # -- generation -----------------------------------------------------------
    def _ar1(self, n: int, rho: float, sd: float, zi: float) -> tuple[np.ndarray, float]:
        drive = self.rng.standard_normal(n) * sd * math.sqrt(1 - rho**2)
        y, _ = lfilter([1.0], [1, -rho], drive, zi=[zi * rho])
        return y, float(y[-1])

    def _generate(self, n: int) -> None:
        fs = self.fs
        t = self._t_gen + np.arange(n) / fs
        rng = self.rng

        target = np.full(n, self._target_state)
        s, _ = lfilter([1 - self._state_lag], [1, -self._state_lag], target, zi=[self._s * self._state_lag])
        self._s = float(s[-1])
        ou, self._ou = self._ar1(n, self._ou_rho, self.state_noise, self._ou)
        s_eff = np.clip(s + ou, -0.5, 1.5)
        mw = 1.0 - s_eff  # mind-wandering-ness

        env = np.empty((4, n))
        for k in range(4):
            e, self._env_zi[k] = self._ar1(n, self._env_rho, self.burstiness, self._env_zi[k])
            env[k] = np.clip(1.0 + e, 0.2, 2.0)
        freqs = self._osc_f0 + rng.standard_normal(4) * self._osc_jit
        osc = np.empty((4, n))
        for k in range(4):
            ph = self._phase[k] + 2 * math.pi * freqs[k] * np.arange(1, n + 1) / fs
            osc[k] = np.sin(ph)
            self._phase[k] = ph[-1] % (2 * math.pi)

        eeg = np.empty((4, n))
        for ch in range(4):
            temporal = ch in (0, 3)
            white = rng.standard_normal(n)
            pink, self._pink_zi[ch] = lfilter(_PINK_B, _PINK_A, white, zi=self._pink_zi[ch])
            sig = 18.0 * pink
            # aperiodic slope: more white noise when on task -> flatter spectrum (temporal mostly)
            white_amp = (2.0 + 6.0 * self.aperiodic_gain * s_eff) if temporal else (2.0 + 1.5 * s_eff)
            sig += rng.standard_normal(n) * np.clip(white_amp, 0.5, None)
            # delta rises during mind-wandering on temporal channels
            delta_amp = (6.0 * (1.0 + self.delta_gain * mw)) if temporal else 4.0
            sig += delta_amp * env[0] * osc[0]
            # theta: nearly state-independent (frontal theta did not replicate)
            theta_amp = 5.0 * (1.0 + self.theta_gain * s_eff)
            sig += theta_amp * env[1] * osc[1]
            # alpha: sign depends on expertise profile; strongest on temporal channels
            alpha_base = 9.0 if temporal else 7.0
            sig += alpha_base * np.clip(1.0 + self.alpha_gain * s_eff, 0.2, None) * env[2] * osc[2]
            # beta rises during mind-wandering on frontal channels
            beta_amp = (4.0 * (1.0 + self.beta_gain * mw)) if not temporal else 3.0
            sig += beta_amp * env[3] * osc[3]
            eeg[ch] = sig

        markers: list[Packet] = []
        self._add_blinks(eeg, t, markers)
        self._add_jaw(eeg, t, markers)
        self._add_bad_fit(eeg, t)
        t0 = float(t[0])
        if self._defect("silent", t0):
            self._t_gen += n / fs
            return
        if self._defect("bad_fit", t0) or self._defect("all_bad_fit", t0):
            chans = range(4) if self._defect("all_bad_fit", t0) else (0, 3)
            for ch in chans:
                eeg[ch] += 400.0 * np.sin(2 * math.pi * 50.0 * t) + rng.standard_normal(n) * 120.0
        if self._defect("dry_ptp", t0):
            eeg += 110.0 * np.sin(2 * math.pi * 0.3 * t)[None, :] * np.array([[1.0], [0.9], [1.1], [1.0]])

        if self._defect("bands_only", t0):
            self._band_buf.extend(eeg.T.tolist())
            while self._next_band <= t[-1]:
                self._emit_bands(np.asarray(self._band_buf, dtype=float).T, self._next_band)
                self._next_band += 0.1
        else:
            if not self._defect("no_offset", t0) and not self._defect("volts", t0):
                eeg += 800.0  # Mind Monitor raw values are offset microvolts (~0..1682)
            if self._defect("volts", t0):
                eeg *= 1e-6
            step = 2 if self._defect("low_rate", t0) else 1
            for i in range(self._decim % step, n, step):
                self._emit(Packet("eeg", float(t[i]), eeg[:, i].tolist()))
            self._decim = (self._decim + n) % step
        self._emit_many(markers)
        self._emit_aux(t[-1])
        self._t_gen += n / fs

    def _emit_bands(self, eeg: np.ndarray, now: float) -> None:
        """Mind Monitor style absolute band powers (log10 uV^2/Hz, roughly -1..1), 10 Hz."""
        n = eeg.shape[1]
        if n < 128:
            return
        x = eeg - eeg.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(x, axis=1)) ** 2 / (n * self.fs)
        f = np.fft.rfftfreq(n, 1.0 / self.fs)
        for band, (lo, hi) in {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 44)}.items():
            m = (f >= lo) & (f < hi)
            vals = np.log10(np.maximum(spec[:, m].mean(axis=1), 1e-12))
            self._emit(Packet("band", float(now), {"band": band, "values": vals.tolist()}, f"/muse/elements/{band}_absolute"))

    def _add_blinks(self, eeg: np.ndarray, t: np.ndarray, markers: list[Packet]) -> None:
        dt = len(t) / self.fs
        if self._blink_until < t[0] and self.rng.random() < self.blink_rate_hz * dt:
            self._blink_start = float(t[0])
            self._blink_until = self._blink_start + 0.3
            markers.append(Packet("blink", float(t[0]), []))
        if self._blink_until >= t[0]:
            u = (t - self._blink_start) / 0.3
            shape = np.where((u >= 0) & (u <= 1), 0.5 - 0.5 * np.cos(2 * math.pi * u), 0.0)
            eeg[1] += 180.0 * shape
            eeg[2] += 170.0 * shape
            eeg[0] += 25.0 * shape
            eeg[3] += 25.0 * shape

    def _add_jaw(self, eeg: np.ndarray, t: np.ndarray, markers: list[Packet]) -> None:
        n = len(t)
        dt = n / self.fs
        if self._jaw_until < t[0] and self.rng.random() < self.jaw_rate_hz * dt:
            self._jaw_until = float(t[0]) + 1.5
            markers.append(Packet("jaw_clench", float(t[0]), []))
        if self._jaw_until >= t[0]:
            mask = t <= self._jaw_until
            emg = self.rng.standard_normal((4, n)) * 60.0
            emg = np.diff(emg, axis=1, prepend=emg[:, :1])  # crude high-pass: EMG is broadband/high-frequency
            eeg[:, mask] += emg[:, mask]

    def _add_bad_fit(self, eeg: np.ndarray, t: np.ndarray) -> None:
        dt = len(t) / self.fs
        for ch in (0, 3):  # only temporal channels lose contact in this model
            if self._bad_fit_until[ch] < t[0] and self.rng.random() < self.bad_fit_rate_hz * dt:
                self._bad_fit_until[ch] = float(t[0]) + self.rng.uniform(5, 15)
            if self._bad_fit_until[ch] >= t[0]:
                mask = t <= self._bad_fit_until[ch]
                eeg[ch, mask] += 400.0 * np.sin(2 * math.pi * 50.0 * t[mask]) + self.rng.standard_normal(int(mask.sum())) * 120.0

    def fit_values(self, now: float) -> list[float]:
        if self._defect("all_bad_fit", now):
            return [4.0] * 4
        if self._defect("bad_fit", now):
            return [4.0, 1.0, 1.0, 4.0]
        if self._defect("medium_fit", now):
            return [2.0] * 4
        return [4.0 if self._bad_fit_until[ch] >= now else 1.0 for ch in range(4)]

    def _emit_aux(self, now: float) -> None:
        if self._defect("silent", now):
            self._next_horseshoe = self._next_fnirs = self._next_hr = self._ppg_t = self._next_band = now
            return
        while self._next_horseshoe <= now:
            if not self._defect("no_horseshoe", self._next_horseshoe):
                self._emit(Packet("horseshoe", self._next_horseshoe, self.fit_values(self._next_horseshoe), "/muse/elements/horseshoe"))
            self._next_horseshoe += 0.1
        while self._next_fnirs <= now:
            drift = 0.02 * math.sin(2 * math.pi * self._next_fnirs / 90.0)
            self._emit(Packet("fnirs", self._next_fnirs, [1.0 + drift + self.rng.normal(0, 0.002) for _ in range(5)], "/muse/optics"))
            self._next_fnirs += 0.1
        while self._ppg_t <= now:
            v = 2000 + 300 * math.sin(2 * math.pi * self._hr_bpm / 60 * self._ppg_t) + self.rng.normal(0, 10)
            self._emit(Packet("ppg", self._ppg_t, [v, v * 0.8, v * 0.6], "/muse/ppg"))
            self._ppg_t += 1 / self._ppg_fs
        while self._next_hr <= now:
            self._hr_bpm += self.rng.normal(0, 0.5) - 0.05 * (self._hr_bpm - 66.0)
            self._emit(Packet("heart_rate", self._next_hr, [self._hr_bpm], "/muse/elements/heart_rate"))
            self._next_hr += 1.0
