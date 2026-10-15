"""Headless closed-loop simulation with the synthetic source (the integration test).

Runs calibration + ``n_trials`` trials per session on a virtual clock (optionally
several sessions sharing the persisted posterior), then compares the
optimizer's posterior optimum with the synthetic ground truth:

* **regret**: ``f(x*) - f(x_best) <= regret_tol`` (``f`` is normalised to a max of 1),
* **distance** on the ground truth's main axes (the two with the largest
  weight: speed and fractal dimension) ``<= dist_tol``,
* **improvement**: ``f(x_best) > f(literature prior)`` - the optimizer moved
  away from the warm start in the right direction.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..ingest.synthetic import GroundTruth, SyntheticSource
from ..optimizer import GPOptimizer
from ..scorer import Calibration, FocusScorer, load_spec
from ..visuals.params import AXES, LITERATURE_PRIOR
from .clock import SimClock
from .orchestrator import Orchestrator
from .protocol import Protocol


def run_simulation(
    n_trials: int = 20,
    seed: int = 0,
    profile: str = "novice",
    time_scale: float = 1.0,
    regret_tol: float = 0.30,
    dist_tol: float = 0.50,
    work_dir: str | Path | None = None,
    acquisition: str = "ei",
    verbose: bool = True,
    spec_path: str | None = None,
    scorer_profile: str = "novice",
    n_sessions: int = 1,
    state_noise: float | None = None,
    effect_scale: float = 1.0,
    protocol: Protocol | None = None,
) -> dict[str, Any]:
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="contemplative-sim-"))
    protocol = (protocol or Protocol(n_trials=n_trials)).scaled(time_scale)
    spec = load_spec(spec_path, profile="novice" if scorer_profile == "auto" else scorer_profile)
    gt = GroundTruth()
    opt_path = work / "optimizer_state.json"
    cal_path = work / "calibration.json"
    wall0 = time.time()
    summaries = []
    source = None
    src_kwargs = {"profile": profile, "ground_truth": gt}
    if state_noise is not None:
        src_kwargs["state_noise"] = state_noise
    src_kwargs["effect_scale"] = effect_scale
    for s in range(n_sessions):
        source = SyntheticSource(seed=seed + s, **src_kwargs)
        scorer = FocusScorer(spec)
        if cal_path.exists():
            scorer.set_calibration(Calibration.load(cal_path))
        optimizer = GPOptimizer(opt_path, acquisition=acquisition, seed=seed + s)
        orch = Orchestrator(
            source, scorer, optimizer, protocol, work / "sessions" / f"sim-{s:02d}", cal_path, clock=SimClock(), wearer="sim", profile=scorer_profile
        )
        if verbose:
            orch.add_listener(lambda ev: print("  " + ev["msg"]) if ev["type"] == "log" else None)
        summaries.append(asyncio.run(orch.run()))
    assert source is not None
    optimizer = GPOptimizer(opt_path, acquisition=acquisition, seed=seed)
    best = optimizer.best()
    x_best = np.array(best["x"])
    f_best, f_opt = gt(x_best), gt(gt.optimum)
    f_prior = gt(LITERATURE_PRIOR)
    main = gt.main_axes()
    dist = float(np.linalg.norm(x_best[main] - gt.optimum[main]))
    regret = float(f_opt - f_best)
    rewards = [o.y for o in optimizer.rewards]
    return {
        "ok": regret <= regret_tol and dist <= dist_tol and f_best > f_prior,
        "regret": regret,
        "regret_tol": regret_tol,
        "distance_main_axes": dist,
        "dist_tol": dist_tol,
        "main_axes": [AXES[i] for i in main],
        "x_best": {a: round(float(v), 3) for a, v in zip(AXES, x_best)},
        "x_opt": {a: round(float(v), 3) for a, v in zip(AXES, gt.optimum)},
        "f_best": f_best,
        "f_prior": f_prior,
        "n_rewards": len(rewards),
        "rewards": [round(r, 3) for r in rewards],
        "best_observed_reward": max(rewards) if rewards else None,
        "length_scales": optimizer.length_scales(),
        "calibration": summaries[0].get("calibration"),
        "scorer": summaries[-1].get("scorer"),
        "wall_s": round(time.time() - wall0, 1),
        "virtual_s": round((protocol.calibration_s + protocol.settle_s + n_trials * protocol.trial_s * n_sessions), 1),
        "work_dir": str(work),
    }
