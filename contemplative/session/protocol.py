"""Timing of the session protocol (report Section 2, "Learning from 15-25 samples")."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Protocol:
    settle_s: float = 10.0  # wait for the stream before anything is scored
    calib_block_s: float = 120.0  # eyes-open fixation 2 min, breath focus 2 min ...
    calib_repeats: int = 2  # ... repeated
    calib_discard_s: float = 15.0
    baseline_s: float = 60.0  # uniform field, 30-60 s (45 s scored)
    stimulus_s: float = 75.0  # 60-90 s (60 s scored)
    discard_s: float = 15.0  # first 15 s of every scored phase dropped (orienting + morph)
    morph_s: float = 10.0  # every parameter change is a >= 10 s morph
    n_trials: int = 16  # 15-20 per sit
    reward: str = "window"  # "window": one spec window over the scored part; "median": median of display scores
    min_valid_segments: int = 10  # fewer usable 2-s segments than this in a scored part -> trial invalid, repeated
    loop_dt: float = 0.1

    def scaled(self, factor: float) -> "Protocol":
        """Shorten every duration by ``factor`` (for quick simulations); the morph stays >= 10 s unless factor < 1 is forced."""
        return replace(
            self,
            settle_s=self.settle_s * factor,
            calib_block_s=self.calib_block_s * factor,
            calib_discard_s=self.calib_discard_s * factor,
            baseline_s=self.baseline_s * factor,
            stimulus_s=self.stimulus_s * factor,
            discard_s=self.discard_s * factor,
            morph_s=max(self.morph_s * factor, min(self.morph_s, 10.0)) if factor >= 1 else self.morph_s * factor,
        )

    @property
    def trial_s(self) -> float:
        return self.baseline_s + self.stimulus_s

    @property
    def calibration_s(self) -> float:
        return 2 * self.calib_block_s * self.calib_repeats

    def to_dict(self) -> dict:
        return asdict(self)
