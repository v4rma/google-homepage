"""The 6-axis stimulus design space (report Section 2). All axes live in [0, 1].

Axis        Physical meaning in the shader
----------  --------------------------------------------------------------
speed       drift speed of the noise field: 0 -> frozen, 1 -> 0.08 field-units/s
fractal     fractal dimension D of the texture, 1.1 -> 1.7 (Hurst H = 2 - D)
symmetry    rotational symmetry order, 1-fold -> 12-fold, with concentricity
color       colour temperature, warm (~2700 K) -> cool (~8000 K), fixed chroma
breath      amplitude of a 0.1 Hz (6 cpm) radial pulsation, 0 -> 5 % scale
texture     1-D latent path through noise "looks" (fbm -> ridged -> warped)

Mean luminance, contrast and field size are fixed in the shader and are not
part of the search space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

AXES = ("speed", "fractal", "symmetry", "color", "breath", "texture")
DIM = len(AXES)

# Warm start from the literature: D ~= 1.3, slow motion, mid symmetry,
# neutral colour, moderate breath pacing, mid texture.
LITERATURE_PRIOR = {"speed": 0.2, "fractal": (1.3 - 1.1) / 0.6, "symmetry": 0.5, "color": 0.5, "breath": 0.5, "texture": 0.5}


@dataclass(frozen=True)
class StimulusParams:
    speed: float = 0.2
    fractal: float = 0.33
    symmetry: float = 0.5
    color: float = 0.5
    breath: float = 0.5
    texture: float = 0.5

    @classmethod
    def from_vector(cls, x) -> "StimulusParams":
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        return cls(**{a: float(v) for a, v in zip(AXES, x)})

    @classmethod
    def from_dict(cls, d: dict) -> "StimulusParams":
        return cls(**{a: float(np.clip(d[a], 0.0, 1.0)) for a in AXES})

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, a) for a in AXES], dtype=float)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def to_physical(p: StimulusParams | dict) -> dict[str, float | int]:
    """Human-readable physical values, for logs and the dashboard."""
    if isinstance(p, StimulusParams):
        p = p.to_dict()
    return {
        "drift_units_per_s": round(0.08 * p["speed"], 4),
        "fractal_D": round(1.1 + 0.6 * p["fractal"], 3),
        "symmetry_fold": int(round(1 + 11 * p["symmetry"])),
        "color_temp_K": int(round(2700 + (8000 - 2700) * p["color"])),
        "breath_amp_pct": round(5.0 * p["breath"], 2),
        "texture_latent": round(p["texture"], 3),
    }
