"""Command line interface.

    python -m contemplative run       --source osc|lsl|replay|synthetic ...   server + session
    python -m contemplative calibrate --source ...                            calibration only
    python -m contemplative simulate  [--trials 20 --seed 0 --sessions 1]     headless loop + assertion
    python -m contemplative check-visuals                                     offline morph-schedule spectrum check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .ingest import make_source
from .optimizer import GPOptimizer
from .scorer import Calibration, FocusScorer, load_spec
from .session import Orchestrator, Protocol, RealClock


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=["osc", "lsl", "replay", "synthetic"], default="synthetic")
    p.add_argument("--osc-host", default="0.0.0.0", help="interface Mind Monitor streams to")
    p.add_argument("--osc-port", type=int, default=5000, help="Mind Monitor OSC target port")
    p.add_argument("--lsl-type", default="EEG")
    p.add_argument("--replay", help="raw_eeg.csv of a recorded session, or a Mind Monitor CSV export")
    p.add_argument("--replay-loop", action="store_true")
    p.add_argument("--synthetic-profile", choices=["novice", "expert"], default="novice")
    p.add_argument("--synthetic-seed", type=int, default=0)
    p.add_argument("--wearer", default="default")
    p.add_argument("--data-dir", default="data", help="per-wearer calibration, optimizer state and sessions live here")
    p.add_argument("--spec", help="feature spec JSON (default: shipped fa_on_task_score.json; fallback fa_frontal_theta_alpha_v1.json)")
    p.add_argument("--profile", choices=["novice", "expert", "default", "auto"], default="novice", help="weight profile; 'auto' lets calibration choose novice vs expert")
    p.add_argument("--window", type=float, help="override the spec's display window length (s), e.g. 30")
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--baseline", type=float, default=60.0)
    p.add_argument("--stimulus", type=float, default=75.0)
    p.add_argument("--discard", type=float, default=15.0)
    p.add_argument("--morph", type=float, default=10.0)
    p.add_argument("--calib-block", type=float, default=120.0)
    p.add_argument("--calib-repeats", type=int, default=2)
    p.add_argument("--reward", choices=["window", "median"], default="window")
    p.add_argument("--acquisition", choices=["ei", "ucb"], default="ei")
    p.add_argument("--recalibrate", action="store_true", help="run calibration even if one exists")
    p.add_argument("--skip-calibration", action="store_true")


def _server_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--td-osc", default="127.0.0.1:10000", help="TouchDesigner OSC target host:port for parameter broadcast")
    p.add_argument("--no-td-osc", action="store_true")
    p.add_argument("--autostart", action="store_true", help="start the session as soon as the server is up")


def build_protocol(a: argparse.Namespace) -> Protocol:
    return Protocol(
        n_trials=a.trials,
        baseline_s=a.baseline,
        stimulus_s=a.stimulus,
        discard_s=a.discard,
        morph_s=max(a.morph, 10.0),
        calib_block_s=a.calib_block,
        calib_repeats=a.calib_repeats,
        reward=a.reward,
    )


def build_source(a: argparse.Namespace, clock):
    if a.source == "osc":
        return make_source("osc", host=a.osc_host, port=a.osc_port, clock=clock.now)
    if a.source == "lsl":
        return make_source("lsl", stream_type=a.lsl_type, clock=clock.now)
    if a.source == "replay":
        if not a.replay:
            sys.exit("--replay PATH is required with --source replay")
        return make_source("replay", path=a.replay, loop=a.replay_loop)
    return make_source("synthetic", seed=a.synthetic_seed, profile=a.synthetic_profile)


def build_orchestrator(a: argparse.Namespace, protocol: Protocol) -> Orchestrator:
    clock = RealClock()
    wearer_dir = Path(a.data_dir) / a.wearer
    session_dir = wearer_dir / "sessions" / time.strftime("%Y%m%d-%H%M%S")
    cal_path = wearer_dir / "calibration.json"
    spec = load_spec(a.spec, profile="novice" if a.profile == "auto" else a.profile)
    if a.window:
        spec.window_s = float(a.window)
    scorer = FocusScorer(spec)
    if cal_path.exists() and not a.recalibrate:
        cal = Calibration.load(cal_path)
        if cal.spec_name != spec.name:
            print(f"[cli] calibration at {cal_path} is for spec {cal.spec_name!r}, current spec is {spec.name!r}; recalibrating")
        else:
            scorer.set_calibration(cal)
    optimizer = GPOptimizer(wearer_dir / "optimizer_state.json", acquisition=a.acquisition)
    source = build_source(a, clock)
    return Orchestrator(
        source,
        scorer,
        optimizer,
        protocol,
        session_dir,
        cal_path,
        clock=clock,
        wearer=a.wearer,
        force_calibration=a.recalibrate,
        skip_calibration=a.skip_calibration,
        profile=a.profile,
    )


def cmd_run(a: argparse.Namespace, calibrate_only: bool = False) -> None:
    import uvicorn

    from .session.server import SessionServer
    from .visuals.osc_broadcast import OSCBroadcaster

    protocol = build_protocol(a)
    if calibrate_only:
        protocol = Protocol(**{**protocol.to_dict(), "n_trials": 0})
        a.recalibrate = True
    osc = None
    if not a.no_td_osc:
        host, _, port = a.td_osc.partition(":")
        osc = OSCBroadcaster(host or "127.0.0.1", int(port or 10000))
    for w in protocol.warnings(a.window):
        print(f"[protocol] warning: {w}", flush=True)
    server = SessionServer(lambda: build_orchestrator(a, protocol), osc=osc, autostart=a.autostart or calibrate_only)
    print(f"visuals:   http://{a.host}:{a.port}/\ndashboard: http://{a.host}:{a.port}/dashboard", flush=True)
    if a.source == "osc":
        print(f"Mind Monitor: set OSC Stream Target to this machine's IP, port {a.osc_port}", flush=True)
    uvicorn.run(server.app, host=a.host, port=a.port, log_level="warning")


def cmd_simulate(a: argparse.Namespace) -> None:
    from .session.simulate import run_simulation

    protocol = Protocol(n_trials=a.trials, baseline_s=a.baseline, stimulus_s=a.stimulus, discard_s=a.discard, morph_s=a.morph, calib_block_s=a.calib_block, calib_repeats=a.calib_repeats, reward=a.reward)
    res = run_simulation(
        n_trials=a.trials,
        seed=a.seed,
        profile=a.synthetic_profile,
        time_scale=a.time_scale,
        regret_tol=a.regret_tol,
        dist_tol=a.dist_tol,
        work_dir=a.work_dir,
        acquisition=a.acquisition,
        verbose=not a.quiet,
        spec_path=a.spec,
        scorer_profile=a.profile,
        n_sessions=a.sessions,
        state_noise=a.state_noise,
        effect_scale=a.effect_scale,
        protocol=protocol,
    )
    print(json.dumps(res, indent=1, default=str))
    print("\nSIMULATION", "PASSED" if res["ok"] else "FAILED", f"(regret {res['regret']:.3f} <= {res['regret_tol']}, main-axis distance {res['distance_main_axes']:.3f} <= {res['dist_tol']}, f_best {res['f_best']:.3f} > f_prior {res['f_prior']:.3f})")
    sys.exit(0 if res["ok"] else 1)


def cmd_check_visuals(a: argparse.Namespace) -> None:
    from .visuals.temporal_check import check_morph_schedule

    res = check_morph_schedule(morph_s=a.morph)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["ok"] else 1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="contemplative", description="closed-loop contemplative visuals (Muse Athena -> FA score -> GP -> shader)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="start the web server and a session")
    _common(p_run)
    _server_args(p_run)
    p_cal = sub.add_parser("calibrate", help="run only the calibration routine (server + visuals page needed)")
    _common(p_cal)
    _server_args(p_cal)
    p_sim = sub.add_parser("simulate", help="headless closed loop with the synthetic source; asserts optimum recovery")
    _common(p_sim)
    p_sim.add_argument("--seed", type=int, default=0)
    p_sim.add_argument("--sessions", type=int, default=1, help="consecutive sits sharing the persisted posterior")
    p_sim.add_argument("--time-scale", type=float, default=1.0, help="scale every protocol duration (virtual time is free; this is for quick smoke runs)")
    p_sim.add_argument("--regret-tol", type=float, default=0.30)
    p_sim.add_argument("--dist-tol", type=float, default=0.50)
    p_sim.add_argument("--state-noise", type=float, help="synthetic wearer's slow state drift (sd); default 0.05, try 0.3 for realism")
    p_sim.add_argument("--effect-scale", type=float, default=1.0, help="scale the synthetic wearer's EEG response to the state; 0.1-0.2 approximates the analysis' effect sizes")
    p_sim.add_argument("--work-dir")
    p_sim.add_argument("--quiet", action="store_true")
    p_sim.set_defaults(trials=20)
    p_chk = sub.add_parser("check-visuals", help="spectrum check of the backend's parameter/morph schedule (no 3-30 Hz content)")
    p_chk.add_argument("--morph", type=float, default=10.0)

    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "calibrate":
        cmd_run(a, calibrate_only=True)
    elif a.cmd == "simulate":
        cmd_simulate(a)
    elif a.cmd == "check-visuals":
        cmd_check_visuals(a)


if __name__ == "__main__":
    main()
