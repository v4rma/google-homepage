# Temporal spectrum check: nothing periodic in 3–30 Hz

The scorer reads EEG power in 2–40 Hz. Any luminance modulation of the display in roughly 3–30 Hz drives a steady-state visual evoked response in the same band, so the reward would partly measure the stimulus rather than the wearer's state (report Section 2; the flicker/Lucia row). The design therefore forbids periodic content in that band, and this document records how that is checked.

There are two places where temporal structure can enter the display.

## 1. The backend's parameter schedule

The orchestrator sends step changes: a new parameter vector at the start of every stimulus phase and a baseline/stimulus mix change at every phase boundary. The page never applies them directly. `visuals.js` morphs every target with a smoothstep over `morph_s` seconds and clamps `morph_s` to at least 10 s (the only exception is the initial state sync when a page connects, which is instantaneous and happens before any scoring).

A smoothstep is continuous with continuous first derivative, so its spectrum falls as f⁻³ beyond ≈1/T. `python -m contemplative check-visuals` (`contemplative/visuals/temporal_check.py`) reconstructs the full schedule of a trial sequence the way the page morphs it, samples it at 120 Hz, and measures a Welch spectrum of the baseline/stimulus mix, each of the six parameter axes, and the 0.1 Hz breath pulsation. Pass criteria: the 3–30 Hz share of the fluctuation power is below 10⁻⁶ and at least 60 dB below the sub-1 Hz content. Measured with the default 10 s morph: shares of 10⁻¹¹–10⁻¹², 108–117 dB margin. `tests/test_visuals.py` runs the same check and also verifies the 10 s floor.

## 2. What the shader does by itself

Within a phase the only time-dependent terms in the fragment shader are:

- the drift of the 3-D value-noise field along its third axis at `0.08 · speed` noise units per second (≤ 0.08 u/s). Value noise is band-limited per octave; with six octaves at base scale 2.6 the highest spatial octave has ≈ 2.6 · 2⁵ ≈ 83 cycles per unit, so the fastest temporal frequency a pixel can see is ≈ 83 · 0.08 ≈ 6.7 Hz **in amplitude terms only if the texture were a pure tone at that octave**. The octave gain 2^−H with H ∈ [0.3, 0.9] and the smooth interpolation put most of the power in the low octaves (≤ 1 Hz at full speed), and the motion is a continuous translation of an aperiodic field, so there is no line in the spectrum — broad, steeply falling, 1/f-like content. This is the same statistic as slowly drifting clouds or water, which is the intent;
- the 0.1 Hz breath pulsation, a pure sinusoid with at most 5 % radial zoom;
- the smoothstep cross-fades between integer kaleidoscope orders, which only move when `symmetry` morphs (case 1).

Nothing is frame-rate locked; `requestAnimationFrame` timing jitter changes `dt`, not the parameters.

Because the shader's temporal content depends on the actual pixels, it is measured on the rendering itself rather than argued only analytically. Open the visuals page with `?diag=1` (or press `D`). The page reads back five 8×8 pixel patches every frame with `gl.readPixels`, converts to luminance, and every 20 s computes a DFT on a 0.1 Hz grid up to 40 Hz. It reports per patch: the 3–30 Hz share of the fluctuation power, the peak frequency in that band and its share, and the < 1 Hz share. The report is shown as an overlay, logged to the console, sent to the server (recorded as a `visuals_diag` marker in `markers.csv`) and relayed to the dashboard log.

Expected result at any parameter setting: the < 1 Hz share dominates, the 3–30 Hz share stays in the low percent at most (at high `speed`, high `fractal` and low `symmetry`, which is the worst case) with no single peak carrying more than a fraction of a percent of the power. A fixed-frequency peak in 3–30 Hz would indicate a bug (or a monitor/driver artefact) and must be fixed before scoring sessions.

## Things that would break the guarantee

- Changing `morph_s` below 10 s on the server does nothing (the page clamps), but editing the clamp in `visuals.js` does.
- Adding `uTime`-driven periodic terms to the shader other than the 0.1 Hz pulsation.
- Driving the visuals from TouchDesigner over OSC without re-implementing the ≥ 10 s morph on the TD side.
- Displays that flicker at low PWM backlight frequencies; prefer a monitor with DC dimming or full brightness.
