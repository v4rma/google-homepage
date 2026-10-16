# Contemplative visuals — stage-1 closed-loop prototype

Muse Athena EEG → focused-attention (FA) score → Bayesian optimiser → parameterised WebGL visual, with the loop closed every trial. A wearer sits in front of a screen; the system shows a slowly morphing abstract field, measures how "on task" the EEG looks relative to a preceding baseline, and uses a Gaussian-process optimiser to pick the next point in a 6-axis design space. Everything is recorded, and the optimiser's posterior persists across sessions so each sit continues where the last one stopped.

```
Muse Athena ──Bluetooth──▶ Mind Monitor ──OSC/UDP──▶ ingest/   (or LSL / CSV replay / synthetic)
                                                       │
                              raw + markers to disk ◀──┤
                                                       ▼
                                                    scorer/    2-s segments → artifact rejection → group PSD
                                                       │       → features from a JSON spec → per-wearer z-score
                                                       ▼
                                                    session/   baseline 60 s → stimulus 75 s (first 15 s discarded)
                                                       │       reward = window score(stimulus) − window score(baseline)
                                                       ▼
                                                    optimizer/ GP (ARD Matérn-5/2) + EI over [0,1]^6, persisted
                                                       │
                                                       ▼
                                                    visuals/   browser WebGL page (websocket) + OSC to TouchDesigner
```

Status: a working prototype that runs end to end against the synthetic wearer. It has **not** been run against a real Muse Athena yet — see "Assumptions that need a hardware check".

## Setup

Python 3.11+ (developed on 3.12).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e ".[dev]"
pip install pylsl==1.18.6                # optional, only for --source lsl (needs liblsl)
```

Run everything as a module from the repo root (`python -m contemplative ...`) or, after `pip install -e .`, as `contemplative ...`.

Quick checks:

```bash
python -m contemplative simulate --quiet       # headless closed loop, ~20 s; exits 0 if the optimum is recovered
python -m contemplative check-visuals          # spectrum of the morph schedule: no 3–30 Hz content
python -m contemplative check-stream --source synthetic --seconds 5   # the stream diagnostics on the synthetic wearer
pytest                                         # unit tests + the simulation / OSC integration tests
```

## Muse Athena via Mind Monitor (primary path), step by step

1. **Pair** the Athena with the phone running Mind Monitor and get a good fit (all four horseshoe indicators solid).
2. **Mind Monitor settings**: *OSC Stream Target IP* = the LAN IP of the computer running this program (Windows: `ipconfig`, the IPv4 address of the Wi-Fi adapter; macOS/Linux: `ifconfig`/`ip a`); *OSC Stream Port* = `5000` (the default here; `--osc-port` to change); *OSC Stream Brainwaves* = **All** (raw EEG must be streamed — band powers alone are not enough). Phone and PC must be on the same Wi-Fi, not a guest network with client isolation. Then tap the stream icon.
3. **Check the stream before anything else** (20 s, no session, nothing written):

```bash
python -m contemplative check-stream --source osc
```

It prints, live and as a final report: OSC packets per address with rates, raw EEG sample rate (overall and per channel), whether raw `/muse/eeg` arrives at all or only band-power messages, the horseshoe values as received per channel, the raw value range / median per channel with a units guess (Mind Monitor's ~800 µV offset, zero-centred, volts…), and per-segment rejection reasons with counts (`ptp`, `flat`, `fit`, `blink`, `jaw`, `emg`) — followed by a one-line **diagnosis** naming the single most likely problem and what to do. Exit code 0 means usable (possibly with a warning), 1 means it will not work yet. The same panel appears live on the dashboard ("Stream check", runs automatically when the server starts and on the **Check stream** button).

Typical diagnoses:

| diagnosis | meaning | fix |
|---|---|---|
| `no_packets` | nothing reaches Python | wrong Target IP / port, different Wi-Fi, or **Windows Defender Firewall** blocking inbound UDP for `python.exe` (allow it when Windows asks, or add an inbound rule for UDP 5000). Separate the two: run `python -m contemplative fake-muse` in a second terminal — if `check-stream` passes with the fake stream, the problem is on the phone/network side |
| `bands_only` | band powers arrive, no raw EEG | Mind Monitor → Settings → *OSC Stream Brainwaves* → **All** |
| `low_rate` | raw EEG well below 256 Hz | Wi-Fi loss / phone throttling: keep Mind Monitor in the foreground, screen on, battery optimisation off |
| `fit_unknown` (warn) | no horseshoe messages | fit is treated as *unknown*, not bad; scoring continues on peak-to-peak/flat checks only |
| `all_rejected_fit` / `temporal_rejected_fit` | horseshoe says the electrodes (TP9/TP10) are not seated | reseat behind the ears, hair out of the way, moisten the skin; or `--fit-tolerant` to accept medium fit |
| `all_rejected_ptp` | > 150 µV peak-to-peak everywhere | movement / jaw / loose band / mains; `--fit-tolerant` raises the threshold to 300 µV for dry electrodes |
| `units`, `units_note` | raw values not in the expected range | the offset is auto-detected and removed per segment; volts are rescaled; anything else is reported |

4. **Start the server** (the stream check runs again automatically):

```bash
python -m contemplative run --source osc --wearer alice            # add --fit-tolerant if check-stream said so
```

5. Open `http://127.0.0.1:8000/` on the display the wearer looks at (click or press `F` for fullscreen; the cursor and hints hide after a few seconds) and `http://127.0.0.1:8000/dashboard` on a second screen or laptop. Wait for the stream check panel to be green or amber, then press **Start session** (or pass `--autostart`). During calibration the dashboard shows the live count of valid windows per block (5 needed); a block that does not reach it is extended automatically (twice) before the session aborts with the same diagnosis as above.

The ingest listens on `0.0.0.0:5000` by default and understands the Mind Monitor OSC set: `/muse/eeg` (4–5 floats, µV with ~800 offset, 256 Hz), `/muse/elements/horseshoe` (1 good / 2 medium / 4 bad), `/muse/elements/blink`, `/muse/elements/jaw_clench`, `/muse/ppg`, `/muse/acc`, `/muse/gyro`, band `_absolute` messages, and anything under `/muse/optics*` / `/muse/fnirs*` (recorded as fNIRS). Unknown addresses are recorded under `other` so nothing streamed is lost. Channel order is always `TP9, AF7, AF8, TP10`.

### Robustness to Athena / Mind Monitor differences

- **No horseshoe messages** → fit is *unknown*, not bad: no fit-based rejection, the dashboard shows `fit ?` with a warning, and the peak-to-peak / flat / marker checks carry the artifact rejection.
- **Raw offset / scale** → the per-channel median of the first segment is reported as the offset and every segment is de-meaned, so a stream centred on 0 instead of 800 scores identically. Values that look like volts (MAD < 0.01) are rescaled ×10⁶; other implausible ranges are flagged.
- **`--fit-tolerant`** → accepts medium horseshoe fit (2) on TP9/TP10 as well as the frontal pair and raises the peak-to-peak threshold to 300 µV. Use it when `check-stream` reports fit/ptp rejection that reseating does not fix.
- **Band powers only** → the session aborts during settle naming the exact Mind Monitor setting; the 1/f exponent and alpha-frequency features cannot be computed from band powers, so they are not approximated.
- **No EEG at settle** → the settle phase waits up to 45 s for raw EEG, then aborts with the diagnosis instead of running 8 minutes of calibration on nothing.

### Rehearsing without hardware

`python -m contemplative fake-muse [--target host:port] [--synthetic-defect …]` sends the synthetic wearer as a Mind Monitor OSC stream. Defects: `silent`, `bands_only`, `no_horseshoe`, `no_offset`, `volts`, `medium_fit`, `bad_fit`, `all_bad_fit`, `dry_ptp`, `low_rate`. For example, in two terminals:

```bash
python -m contemplative fake-muse --synthetic-defect no_horseshoe
python -m contemplative check-stream --source osc              # -> WARN fit_unknown, both groups usable
```

Other sources:

```bash
python -m contemplative run --source lsl --lsl-type EEG          # muselsl / BlueMuse / Mind Monitor LSL
python -m contemplative run --source replay --replay data/alice/sessions/<id>/raw_eeg.csv
python -m contemplative run --source replay --replay mindMonitor_2026-10-15.csv --replay-loop
python -m contemplative run --source synthetic --synthetic-profile expert --autostart
```

Replay accepts this program's own `raw_eeg.csv` (+ `aux.jsonl` next to it) or a Mind Monitor CSV export (`RAW_*`, `HSI_*`, `Elements` columns).

## What a session looks like

1. **Settle** (10 s, up to 45 s if raw EEG has not arrived): uniform field while the stream stabilises; aborts with a diagnosis if no raw EEG arrives.
2. **Calibration** (only when `data/<wearer>/calibration.json` is missing, or with `--recalibrate`): 2 min eyes-open fixation on the uniform isoluminant field, then 2 min breath focus, repeated once (8 min total; `--calib-block`, `--calib-repeats`). Faint on-screen cues say which. The first 15 s of every block are discarded. Each block must yield 5 valid scoring windows (live count on the dashboard); if it does not, the block is extended up to twice before the session aborts with a diagnosis of the stream.
3. **Trials** (16 by default, `--trials`): baseline 60 s on the uniform field → stimulus 75 s with the first 15 s (the ≥10 s morph plus settling) discarded → reward = score over one spec window spanning the scored stimulus interval minus the same for the baseline. The optimiser proposes the next point; literature prior first, then 4 Latin-hypercube seeds, then expected improvement. ~36 min for 16 trials.
4. The wearer may press `B` ("better than the last one") or `W` ("worse") on the visuals page, or the dashboard buttons; the rating becomes a pseudo-observation on the GP.

Everything lands in `data/<wearer>/`:

```
calibration.json            per-feature mean/std/sign, chosen profile, separation
optimizer_state.json        all observations across sessions (the GP is refit from these)
sessions/<timestamp>/
    raw_eeg.csv             t,TP9,AF7,AF8,TP10 at 256 Hz
    aux.jsonl               horseshoe, blinks, jaw, PPG, fNIRS, bands, anything else
    markers.csv             phase starts/ends, params, ratings, log lines, visuals diagnostics
    scores.csv              every display-score window with features and quality
    trials.jsonl            per trial: params, baseline/stimulus scores, reward, rating, quality
    session.json / summary.json
```

### The dashboard

Stream check panel (packets per address, EEG rate, raw ranges and units, horseshoe, rejection reasons, diagnosis; runs while idle), live phase/trial/countdown, calibration progress (valid windows per block), current parameters (unit and physical: fractal D, fold order, Kelvin…), live score sparkline, **signal quality per channel** (horseshoe fit colour + fraction of segments kept, which channel groups are in use, and a visible warning when the scorer has fallen back to frontal-only because TP9/TP10 are badly seated), trial table with rewards and ratings, optimiser state (ARD length scales, GP best with its predicted value), and the calibration (signs chosen, separation d, notes).

### TouchDesigner

The browser page is the reference renderer, but every parameter update is also broadcast over OSC (default `127.0.0.1:10000`, `--td-osc host:port`, `--no-td-osc` to disable): `/visuals/params` (6 floats in `[0,1]`), `/visuals/<axis>` for `speed fractal symmetry color breath texture`, `/visuals/mode`, `/visuals/morph_s`, and `/score`. A TD patch should apply the same ≥10 s smoothstep morph.

## Scorer

The scorer is **data driven by a JSON feature spec** (`contemplative/scorer/spec/`). The default is `fa_on_task_score.json`, copied verbatim from the parallel FA feature analysis on ds001787 + OSF 3uszv (see `docs/` in the project store for the findings). The analysis found that on Muse-proximal channels the signal lives in the **temporal** group (TP9/TP10-like), not frontal:

| feature | group | on-task direction | sign |
|---|---|---|---|
| 1/f aperiodic exponent (log–log fit 2–40 Hz) | temporal | flatter when on task (steeper during mind-wandering) | fixed − |
| delta power | temporal | lower on task | fixed − |
| beta power | frontal | lower on task | fixed − |
| alpha power, theta/alpha ratio | temporal | **depends on expertise**: experts more alpha on task, novices less | **chosen by calibration** |
| individual alpha frequency | temporal | weak | profile weight |

Pipeline: 2-s segments, 50 % overlap → per-channel artifact rejection (peak-to-peak > 150 µV, flat < 0.5 µV std, blink marker → frontal channels dropped, jaw-clench marker → all dropped, horseshoe fit worse than the group limit → channel dropped; temporal requires fit 1, frontal tolerates 2) → mean PSD per channel group over the channels that survived ≥ 50 % of the window → features → per-wearer z-score with calibrated sign → weighted sum → standardised so fixation windows have mean 0, sd 1. Window 60 s / hop 5 s as in the spec (`--window 30` is acceptable per the analysis).

**TP9/TP10 are first-class**: when they are well seated the temporal features drive the score. When the temporal group is unusable for a window the scorer computes the frontal-only features, flags the window `fallback_frontal_only`, and the dashboard shows it. Windows with no usable group are invalid and do not enter rewards; a trial with too few valid windows is repeated with the same parameters.

**Profiles**: `--profile novice` (default, as the analysis recommends), `expert`, `default`, or `auto` (calibration picks whichever profile's weights separate fixation from breath focus better). Independently of the profile, **calibration sets every feature's scale** (mean/std with a per-type floor) and **chooses the sign of the expertise-dependent features** from the direction of the breath-minus-fixation contrast when |d| ≥ 0.2; otherwise it keeps the profile's sign and says so in the calibration notes.

The simpler frontal spec `fa_frontal_theta_alpha_v1.json` (theta/beta and alpha/beta ratios on AF7/AF8) stays as a fallback: it is used automatically if the default spec cannot be loaded, or explicitly with `--spec contemplative/scorer/spec/fa_frontal_theta_alpha_v1.json`. Any spec in the same format works with `--spec`.

## Optimiser

`GPOptimizer`: scikit-learn `GaussianProcessRegressor`, `Constant × Matérn(ν=2.5, ARD) + White`, `normalize_y`, hyperparameters by MAP with log-normal priors on the length scales (centre 0.5, bounded 0.2–3 in unit coordinates) so six dimensions and fifteen noisy points do not collapse the kernel. Acquisition: expected improvement with ξ = 0.1 against a "noisy" incumbent (max posterior mean at observed points), or UCB (`--acquisition ucb`); maximised over Sobol candidates with L-BFGS-B refinement. Warm start: literature prior (D ≈ 1.3, slow motion, mid symmetry, neutral colour) then 4 LHS seeds. State is a JSON file of observations, so the posterior carries across sessions and the kernel is refit from scratch each time. Pairwise ratings become pseudo-observations (previous reward ± half the reward spread, extra noise). "GP best" on the dashboard is argmax of μ − σ.

## Visuals

`contemplative/visuals/static/visuals.js` — one fragment shader, 3-D value-noise fBm over the 6 axes:

| axis | physical range | implementation |
|---|---|---|
| speed | 0–0.08 noise units/s | drift along the noise's third axis (integral of speed, so speed changes are smooth) |
| fractal | D 1.1–1.7 | fBm Hurst exponent H = 2 − D (per-octave gain 2^−H) |
| symmetry | 1–12-fold + concentricity | kaleidoscope fold, cross-faded between integer orders; polar blend for concentricity |
| color | 2700–8000 K | luminance-preserving tint |
| breath | 0–5 % radial zoom | 0.1 Hz sinusoid, the only periodic term |
| texture | fBm → ridged → domain-warped | continuous latent + noise seed offset |

Fixed: mean luminance (0.10 linear), contrast (0.16 peak modulation), field radius (0.46 of the short side; surround is the same mean grey). Baseline is the uniform isoluminant field at the same mean luminance, and every change — including baseline ↔ stimulus — is a smoothstep morph of at least 10 s (server-specified; the page clamps anything shorter). Fullscreen, dark chrome, cursor hidden.

**Nothing periodic in 3–30 Hz** — documented in `docs/temporal-spectrum-check.md`: `python -m contemplative check-visuals` measures the spectrum of the backend's parameter/morph schedule (3–30 Hz share < 10⁻¹¹, >100 dB below the sub-1 Hz content), and opening the page with `?diag=1` (or pressing `D`) measures the luminance spectrum of the actual rendering with `readPixels` and reports it on screen, in the console and to the session markers.

## Simulation mode

```bash
python -m contemplative simulate                      # 20 trials, seed 0, novice wearer, EI
python -m contemplative simulate --sessions 2         # two sits sharing calibration + posterior
python -m contemplative simulate --synthetic-profile expert --profile auto
python -m contemplative simulate --effect-scale 0.15 --state-noise 0.3   # realistic effect sizes
```

The synthetic wearer (`ingest/synthetic.py`) has a hidden ground truth `f(x)` (Gaussian bumps; mainly speed and fractal D, optimum away from the literature prior), a latent state that relaxes towards `f(x)` under the current stimulus and drifts, and EEG that follows the analysis: pink noise whose temporal 1/f exponent flattens with the state, delta down, frontal beta down, alpha down (novice) or up (expert), blinks, jaw clenches, periods of bad temporal fit, plus horseshoe / PPG / heart-rate / fNIRS streams. The loop runs on a virtual clock (a 55-minute session takes ~20 s). The run **asserts** that the GP's posterior optimum has regret ≤ 0.30, lies within 0.50 of the ground truth on its main axes, and beats the literature prior; `tests/test_synthetic_loop.py` runs the same thing under pytest.

An honest caveat: the synthetic defaults give a per-trial reward signal-to-noise of roughly 5, which is what makes 20 trials enough. At the effect sizes the analysis measured (single-window balanced accuracy ≈ 0.6) the per-trial SNR is closer to 0.2–0.3 (`--effect-scale 0.15`); with that, a 6-D optimiser needs many more trials — or sessions — than one sit, which is what the persisted posterior and the optional pairwise ratings are for.

## Assumptions that need a hardware check

- **Athena in Mind Monitor.** The Mind Monitor OSC message set above is documented for Muse 2/S; whether Athena streams through Mind Monitor with the same addresses, and whether its fNIRS and PPG arrive at all (and under which OSC paths — we capture `/muse/optics*` and `/muse/fnirs*` speculatively), is **not verified**. Nothing in the loop depends on fNIRS/PPG; they are recorded only.
- **LSL.** `--source lsl` expects a 4-channel `EEG` stream at 256 Hz in `TP9, AF7, AF8, TP10` order (muselsl convention); Athena support in muselsl/BlueMuse is unverified.
- **Sample timing.** OSC packets carry no device timestamps; the receive time on the host clock is used. Jitter at the few-ms level is irrelevant for 2-s spectral features but makes this recording unsuitable for ERP-style analysis.
- **Horseshoe semantics** (1/2/4) and the ~800 µV raw offset are taken from Mind Monitor's documentation; the artifact thresholds (150 µV peak-to-peak, 0.5 µV flat) are from the analysis spec and were tuned on wet-electrode data. Dry TP9/TP10 may be EMG-dominated; the fallback spec adds a 30–60 Hz EMG check that can be copied into the default spec's `artifact_rejection` if needed.
- **Display.** Luminance and gamma are not calibrated; "isoluminant" means equal sRGB-encoded values. Field size in degrees depends on the monitor and distance.

## Repository layout

```
contemplative/
  ingest/      base.py (Packet/EEGSource), osc_source.py, lsl_source.py, replay_source.py, synthetic.py (+ defects),
               recorder.py, stream_check.py (diagnostics), osc_fake.py (fake Mind Monitor sender)
  scorer/      spec/*.json, features.py, artifacts.py, windows.py, calibration.py, score.py
  optimizer/   gp_optimizer.py
  visuals/     params.py (axes, prior, physical mapping), osc_broadcast.py, temporal_check.py, static/ (pages)
  session/     protocol.py, clock.py, orchestrator.py, server.py, simulate.py
  cli.py       run | calibrate | simulate | check-visuals | check-stream | fake-muse
tests/         unit tests + integration test
docs/          temporal-spectrum-check.md
```
