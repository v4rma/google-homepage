"""Session orchestrator: calibration -> repeated (baseline -> stimulus) trials.

Protocol per trial::

    uniform field  (baseline_s)   ->   stimulus x_k  (stimulus_s)
    |<-discard->| scored          ->   |<-discard->| scored
    reward_k = score(stimulus scored part) - score(baseline scored part)

where ``score(interval)`` is the calibrated score of the feature spec computed
over one window spanning the whole scored part (45-60 s, within the spec's
30-60 s range).  The median of the running display scores is logged as well
(``reward_median``) and can be selected with ``Protocol.reward = "median"``.

The optimizer proposes x_{k+1} after each reward. Everything is logged to the
session folder; the raw stream is recorded by :class:`SessionRecorder`.
Listeners (the web server, OSC broadcaster) get events:

    {"type": "visuals", "mode": ..., "params": {...} | None, "morph_s": ...}
    {"type": "score", "t": ..., "score": ..., "valid": ..., "reasons": [...], "fit": [...]}
    {"type": "state", ...snapshot...}
    {"type": "trial", ...trial record...}
    {"type": "log", "msg": ...}
"""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..ingest.base import EEGSource
from ..ingest.recorder import SessionRecorder
from ..optimizer import GPOptimizer
from ..scorer import CalibrationFitter, FocusScorer, ScoreSample
from ..visuals.params import AXES, StimulusParams, to_physical
from .clock import RealClock
from .protocol import Protocol

Listener = Callable[[dict[str, Any]], None]


class Orchestrator:
    def __init__(
        self,
        source: EEGSource,
        scorer: FocusScorer,
        optimizer: GPOptimizer,
        protocol: Protocol,
        session_dir: str | Path,
        calibration_path: str | Path,
        clock=None,
        wearer: str = "default",
        force_calibration: bool = False,
        skip_calibration: bool = False,
        profile: str = "novice",
    ) -> None:
        self.source = source
        self.scorer = scorer
        self.optimizer = optimizer
        self.protocol = protocol
        self.clock = clock or RealClock()
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_path = Path(calibration_path)
        self.wearer = wearer
        self.force_calibration = force_calibration
        self.skip_calibration = skip_calibration
        self.profile = profile
        self.session_id = self.session_dir.name

        self.listeners: list[Listener] = []
        self.recorder = SessionRecorder(self.session_dir)
        self._scores_f = open(self.session_dir / "scores.csv", "w", newline="")
        self._scores = csv.writer(self._scores_f)
        self._scores.writerow(["t", "phase", "trial", "score", "raw", "valid", "reasons", "fit", "features"])
        self._trials_f = open(self.session_dir / "trials.jsonl", "a")

        self.phase = "idle"
        self.trial = 0
        self.phase_t0 = 0.0
        self.phase_t1 = 0.0
        self.params: StimulusParams | None = None
        self.proposal_tag = ""
        self.trials: list[dict[str, Any]] = []
        self.last_score: ScoreSample | None = None
        self.recent_scores: list[tuple[float, float | None]] = []
        self.calibration_info: dict[str, Any] | None = None
        self.running = False
        self.finished = False
        self.stop_requested = False
        self.message = ""
        self._rating: tuple[int, bool] | None = None
        self._rated_trials: set[int] = set()
        self._pending_params: tuple[StimulusParams, str] | None = None
        self._state_last_emit = -1.0
        self._cached_summary: dict[str, Any] = self.optimizer.summary()

    # -- events ------------------------------------------------------------------
    def add_listener(self, fn: Listener) -> None:
        self.listeners.append(fn)

    def _emit(self, ev: dict[str, Any]) -> None:
        for fn in self.listeners:
            try:
                fn(ev)
            except Exception as e:  # listeners must never break the loop
                print(f"[orchestrator] listener error: {e!r}")

    def log(self, msg: str) -> None:
        self.message = msg
        self._emit({"type": "log", "t": self.clock.now(), "msg": msg})
        self.recorder.marker(self.clock.now(), "log", {"msg": msg})

    def snapshot(self) -> dict[str, Any]:
        now = self.clock.now()
        return {
            "type": "state",
            "session_id": self.session_id,
            "wearer": self.wearer,
            "phase": self.phase,
            "trial": self.trial,
            "n_trials": self.protocol.n_trials,
            "phase_elapsed": round(now - self.phase_t0, 1) if self.phase != "idle" else 0.0,
            "phase_remaining": round(max(0.0, self.phase_t1 - now), 1) if self.phase != "idle" else 0.0,
            "params": self.params.to_dict() if self.params else None,
            "physical": to_physical(self.params) if self.params else None,
            "proposal_tag": self.proposal_tag,
            "last_score": (self.last_score.score if self.last_score and self.last_score.valid else None),
            "fit": self.scorer.fit,
            "quality": self.scorer.last_quality,
            "spec": self.scorer.spec.describe(),
            "calibrated": self.scorer.calibration is not None,
            "calibration": self.calibration_info,
            "trials": self.trials[-40:],
            "optimizer": self._cached_summary,
            "running": self.running,
            "finished": self.finished,
            "message": self.message,
            "scorer": {"valid": self.scorer.n_valid, "rejected": self.scorer.n_rejected},
            "source": self.source.describe(),
            "protocol": self.protocol.to_dict(),
        }

    # -- public control ------------------------------------------------------------
    def rate(self, better: bool) -> dict[str, Any]:
        """Pairwise rating from the wearer: current/last stimulus vs the one before."""
        target = self.trial if self.phase == "stimulus" else self.trial - 1
        if target < 1:
            return {"ok": False, "reason": "no previous trial to compare with"}
        if target in self._rated_trials:
            return {"ok": False, "reason": f"trial {target} already rated"}
        self._rating = (target, better)
        self.recorder.marker(self.clock.now(), "rating", {"trial": target, "better": better})
        applied = self._apply_rating()
        self.log(f"rating for trial {target}: {'better' if better else 'worse'} than previous" + (" (queued)" if not applied else ""))
        return {"ok": True, "trial": target, "applied": applied}

    def _apply_rating(self) -> bool:
        if self._rating is None:
            return False
        target, better = self._rating
        observed = [t["trial"] for t in self.trials if t.get("valid")]
        if target not in observed or target - 1 not in observed or observed[-1] != target:
            return False
        obs = self.optimizer.observe_pairwise(better, session=self.session_id)
        self._rated_trials.add(target)
        self._rating = None
        for t in self.trials:
            if t["trial"] == target:
                t["rating"] = "better" if better else "worse"
                t["pairwise_pseudo_y"] = obs.y if obs else None
        return obs is not None

    def request_stop(self) -> None:
        self.stop_requested = True

    # -- main loop ---------------------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        self.running = True
        self.source.start()
        self._write_session_json()
        try:
            await self._set_visuals("baseline", None)
            self.log("settling: waiting for the EEG stream")
            await self._phase("settle", self.protocol.settle_s, "baseline", None)
            if self.stop_requested:
                return self._finish()
            if self.scorer.calibration is not None and not self.force_calibration:
                self.calibration_info = self.scorer.calibration.summary()
                self.log(f"using existing calibration {self.calibration_path}")
            elif self.skip_calibration:
                self.log("calibration skipped; running with default signs and a running z-score")
            else:
                await self._calibrate()
            for k in range(1, self.protocol.n_trials + 1):
                if self.stop_requested:
                    break
                self.trial = k
                await self._run_trial(k)
            return self._finish()
        except Exception as e:
            self.log(f"session error: {e!r}")
            self._finish(error=repr(e))
            raise
        finally:
            self.running = False

    async def _run_trial(self, k: int) -> None:
        P = self.protocol
        base = await self._phase("baseline", P.baseline_s, "baseline", None)
        if self.stop_requested:
            return
        if self._pending_params is not None:
            params, tag = self._pending_params
            self._pending_params = None
        else:
            x, tag = self.optimizer.propose()
            params = StimulusParams.from_vector(x)
        self.params, self.proposal_tag = params, tag
        stim = await self._phase("stimulus", P.stimulus_s, "stimulus", params)
        base_v = [s for s in base["display"] if s is not None]
        stim_v = [s for s in stim["display"] if s is not None]
        bw, sw = base["window"], stim["window"]
        rec: dict[str, Any] = {
            "trial": k,
            "session": self.session_id,
            "t": self.clock.now(),
            "params": params.to_dict(),
            "physical": to_physical(params),
            "tag": tag,
            "n_baseline_display": len(base_v),
            "n_stimulus_display": len(stim_v),
            "baseline_median": statistics.median(base_v) if base_v else None,
            "stimulus_median": statistics.median(stim_v) if stim_v else None,
            "baseline_window": bw.score if bw.valid else None,
            "stimulus_window": sw.score if sw.valid else None,
            "baseline_quality": bw.quality,
            "stimulus_quality": sw.quality,
        }
        rec["reward_median"] = (rec["stimulus_median"] - rec["baseline_median"]) if base_v and stim_v else None
        rec["reward_window"] = (sw.score - bw.score) if (bw.valid and sw.valid) else None
        n_ok = min(bw.quality.get("n_segments", 0) * max(bw.quality.get("frontal_kept", 0), bw.quality.get("temporal_kept", 0)), sw.quality.get("n_segments", 0) * max(sw.quality.get("frontal_kept", 0), sw.quality.get("temporal_kept", 0)))
        reward = rec["reward_window"] if P.reward == "window" else rec["reward_median"]
        if reward is not None and n_ok >= P.min_valid_segments:
            rec["reward"] = float(reward)
            rec["valid"] = True
            self.optimizer.observe(params.vector(), rec["reward"], session=self.session_id, trial=k)
            fb = " [frontal-only]" if (sw.quality.get("fallback_frontal_only") or bw.quality.get("fallback_frontal_only")) else ""
            self.log(f"trial {k}: reward {rec['reward']:+.2f} ({tag}){fb}")
        else:
            rec["reward"] = None
            rec["valid"] = False
            self._pending_params = (params, tag)
            self.log(f"trial {k}: unusable (baseline {bw.reasons or 'ok'}, stimulus {sw.reasons or 'ok'}, usable segments {n_ok:.0f}); repeating parameters")
        self.trials.append(rec)
        self._apply_rating()
        self._cached_summary = self.optimizer.summary()
        rec["best_after"] = self._cached_summary["best"]
        self._trials_f.write(json.dumps(rec) + "\n")
        self._trials_f.flush()
        self._emit({"type": "trial", **rec})
        self._emit(self.snapshot())

    async def _calibrate(self) -> None:
        P = self.protocol
        spec = self.scorer.spec
        fitter = CalibrationFitter(spec, self.wearer, profile=self.profile)
        self.log("calibration: eyes open, soft gaze on the field; then breath focus")
        for rep in range(P.calib_repeats):
            for cond in ("calibration_fixation", "calibration_breath"):
                if self.stop_requested:
                    return
                res = await self._phase(cond, P.calib_block_s, cond, None, discard=P.calib_discard_s)
                # sliding spec windows over the scored part of the block
                t0, t1 = res["t0"] + P.calib_discard_s, res["t1"]
                # short (test) blocks: shrink the window and densify the hop so a
                # block still yields >= 5 windows; the default protocol is unaffected
                win = min(spec.window_s, 0.6 * (t1 - t0))
                hop = min(spec.hop_s, max(1.0, (t1 - t0 - win) / 4.0))
                n = 0
                for te in np.arange(t0 + win, t1 + 1e-6, hop):
                    wf = self.scorer.interval_features(te - win, te)
                    if wf is not None and wf.valid:
                        fitter.add(cond, wf.features)
                        n += 1
                self.log(f"calibration block {rep + 1}/{P.calib_repeats} {cond}: {n} valid windows")
        cal = fitter.fit()
        cal.save(self.calibration_path)
        shutil.copy(self.calibration_path, self.session_dir / "calibration.json")
        self.scorer.set_calibration(cal)
        self.calibration_info = cal.summary()
        sign_txt = ", ".join(f"{k}{'+' if v > 0 else '-'}" for k, v in cal.feature_sign.items() if cal.weights.get(k))
        self.log(f"calibration saved ({cal.profile}): separation d={cal.separation_d:.2f}; signs {sign_txt}")
        for n in cal.notes:
            self.log("  " + n)
        self._emit(self.snapshot())

    async def _phase(
        self,
        phase: str,
        duration: float,
        mode: str,
        params: StimulusParams | None,
        discard: float | None = None,
    ) -> dict[str, Any]:
        """Run one phase. Returns the display scores after the discard window and one window score over that part."""
        P = self.protocol
        discard = P.discard_s if discard is None else discard
        self.phase = phase
        self.phase_t0 = self.clock.now()
        self.phase_t1 = self.phase_t0 + duration
        await self._set_visuals(mode, params)
        self.recorder.marker(self.phase_t0, f"{phase}_start", {"trial": self.trial, "params": params.to_dict() if params else None, "mode": mode})
        display: list = []
        while self.clock.now() < self.phase_t1 and not self.stop_requested:
            now = self.clock.now()
            self.source.tick(now)
            packets = self.source.read()
            if packets:
                self.recorder.write_packets(packets)
                for s in self.scorer.push(packets):
                    self._on_score(s)
                    if self.phase_t0 + discard <= s.t <= self.phase_t1:
                        display.append(s.score if s.valid else None)
            if now - self._state_last_emit >= 1.0:
                self._state_last_emit = now
                self._emit(self.snapshot())
            await self.clock.sleep(P.loop_dt)
        t_end = self.clock.now()
        window = self.scorer.interval_score(self.phase_t0 + discard, t_end)
        self.recorder.marker(t_end, f"{phase}_end", {"trial": self.trial, "n_display": len(display), "window_score": window.score, "quality": window.quality})
        self.recorder.flush()
        return {"display": display, "window": window, "t0": self.phase_t0, "t1": t_end}

    def _on_score(self, s: ScoreSample) -> None:
        self.last_score = s
        self.recent_scores.append((s.t, s.score if s.valid else None))
        if len(self.recent_scores) > 600:
            self.recent_scores = self.recent_scores[-600:]
        self._scores.writerow(
            [f"{s.t:.3f}", self.phase, self.trial, "" if s.score is None else f"{s.score:.4f}", "" if s.raw is None else f"{s.raw:.4f}", int(s.valid), ";".join(s.reasons), ";".join(str(v) for v in s.fit), json.dumps(_clean(s.features))]
        )
        self._emit({"type": "score", "t": s.t, "score": s.score, "valid": s.valid, "reasons": s.reasons, "fit": s.fit, "quality": s.quality, "phase": self.phase, "trial": self.trial})

    async def _set_visuals(self, mode: str, params: StimulusParams | None) -> None:
        self.source.set_stimulus(params.to_dict() if params else None, mode)
        self._emit({"type": "visuals", "mode": mode, "params": params.to_dict() if params else None, "morph_s": self.protocol.morph_s, "t": self.clock.now()})

    def _write_session_json(self) -> None:
        info = {
            "session_id": self.session_id,
            "wearer": self.wearer,
            "started_wall": time.time(),
            "protocol": self.protocol.to_dict(),
            "source": self.source.describe(),
            "spec": self.scorer.spec.name,
            "axes": list(AXES),
            "optimizer": {"acquisition": self.optimizer.acquisition, "n_prior_observations": len(self.optimizer.observations)},
        }
        with open(self.session_dir / "session.json", "w") as f:
            json.dump(info, f, indent=2)

    def _finish(self, error: str | None = None) -> dict[str, Any]:
        self.phase = "error" if error else "idle"
        self.finished = True
        self.source.set_stimulus(None, "idle")
        self._emit({"type": "visuals", "mode": "idle", "params": None, "morph_s": self.protocol.morph_s, "t": self.clock.now()})
        self.optimizer.save()
        if self.optimizer.state_path and self.optimizer.state_path.exists():
            shutil.copy(self.optimizer.state_path, self.session_dir / "optimizer_state.json")
        summary = {
            "session_id": self.session_id,
            "n_trials": len(self.trials),
            "n_valid": sum(1 for t in self.trials if t.get("valid")),
            "rewards": [t.get("reward") for t in self.trials],
            "best": self.optimizer.best(),
            "length_scales": self.optimizer.length_scales(),
            "scorer": {"valid": self.scorer.n_valid, "rejected": self.scorer.n_rejected},
            "calibration": self.calibration_info,
            "error": error,
        }
        with open(self.session_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        self.log("session finished" if not error else f"session aborted: {error}")
        self.recorder.close()
        self._scores_f.close()
        self._trials_f.close()
        self.source.stop()
        self._emit(self.snapshot())
        return summary


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def _clean(d: dict[str, float]) -> dict[str, float | None]:
    """NaN -> None so the dict survives strict JSON consumers (browsers)."""
    return {k: (None if v is None or not np.isfinite(v) else round(float(v), 4)) for k, v in d.items()}
