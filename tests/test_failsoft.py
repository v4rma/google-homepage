"""Session fail-soft behaviour on a virtual clock: diagnostic aborts instead of a
bare ValueError, calibration block extension, fit-unknown scoring."""

import asyncio

from contemplative.ingest.synthetic import SyntheticSource
from contemplative.optimizer import GPOptimizer
from contemplative.scorer import FocusScorer, load_spec
from contemplative.session import Orchestrator, Protocol, SimClock

FAST = Protocol(settle_s=5.0, settle_max_s=20.0, calib_block_s=60.0, calib_repeats=1, calib_discard_s=5.0, n_trials=1, baseline_s=30.0, stimulus_s=40.0, discard_s=10.0)


def _run(tmp_path, defects, protocol=FAST, tolerant=False, **src_kw):
    spec = load_spec()
    spec.window_s = 20.0
    if tolerant:
        spec.apply_tolerant()
    src = SyntheticSource(seed=0, defects=defects, blink_rate_hz=0.0, jaw_rate_hz=0.0, bad_fit_rate_hz=0.0, **src_kw)
    orch = Orchestrator(src, FocusScorer(spec), GPOptimizer(tmp_path / "opt.json"), protocol, tmp_path / "s", tmp_path / "cal.json", clock=SimClock(), wearer="t")
    logs = []
    orch.add_listener(lambda ev: logs.append(ev["msg"]) if ev["type"] == "log" else None)
    summary = asyncio.run(orch.run())
    return orch, summary, logs


def test_silent_stream_aborts_with_network_diagnosis(tmp_path):
    orch, summary, logs = _run(tmp_path, {"silent"})
    assert orch.phase == "error" and summary["error"]
    assert "no raw EEG" in summary["error"]
    assert "Firewall" in summary["error"] and "Target IP" in summary["error"]
    assert "raw EEG samples 0" in summary["error"]
    # aborted during settle, long before any calibration block
    assert orch.clock.now() <= FAST.settle_max_s + 15
    assert not orch.running and (tmp_path / "s" / "summary.json").exists()


def test_bands_only_aborts_with_mind_monitor_setting(tmp_path):
    orch, summary, logs = _run(tmp_path, {"bands_only"})
    assert orch.phase == "error"
    assert "OSC Stream Brainwaves" in summary["error"]
    assert "band msgs" in summary["error"]


def test_all_bad_fit_extends_then_aborts_with_fit_advice(tmp_path):
    orch, summary, logs = _run(tmp_path, {"all_bad_fit"})
    assert orch.phase == "error"
    assert "valid windows" in summary["error"]
    assert "--fit-tolerant" in summary["error"] or "Reseat" in summary["error"]
    assert sum("Extending the block" in m for m in logs) == FAST.calib_max_extensions
    assert "ValueError" not in summary["error"]


def test_transient_bad_fit_recovers_through_block_extension(tmp_path):
    # electrodes off for the settle + first block, then fine: the block is extended once and calibration completes
    orch, summary, logs = _run(tmp_path, {"all_bad_fit"}, defect_until_s=FAST.settle_s + FAST.calib_block_s + 1.0)
    assert orch.phase == "idle", summary.get("error")
    assert summary["error"] is None
    assert summary["calibration"] is not None
    assert any("extension 1" in m for m in logs)
    assert any("calibration saved" in m for m in logs)


def test_no_horseshoe_session_runs_with_fit_unknown(tmp_path):
    orch, summary, logs = _run(tmp_path, {"no_horseshoe"})
    assert summary["error"] is None, summary["error"]
    assert orch.scorer.fit is None and not orch.scorer.buffer.fit_known
    assert summary["calibration"] is not None
    assert summary["n_valid"] == 1
    snap = orch.snapshot()
    assert snap["fit"] is None and snap["fit_known"] is False


def test_medium_fit_needs_tolerant_mode_for_temporal(tmp_path):
    orch, summary, _ = _run(tmp_path, {"medium_fit"})
    assert summary["error"] is None
    assert orch.scorer.last_quality["fallback_frontal_only"] is True
    orch, summary, _ = _run(tmp_path / "tol", {"medium_fit"}, tolerant=True)
    assert summary["error"] is None
    assert orch.scorer.last_quality["fallback_frontal_only"] is False
    assert orch.scorer.spec.describe()["tolerant"] is True


def test_calibration_progress_is_exposed_during_blocks(tmp_path):
    seen = []
    spec = load_spec()
    spec.window_s = 20.0
    src = SyntheticSource(seed=0, blink_rate_hz=0.0, jaw_rate_hz=0.0, bad_fit_rate_hz=0.0)
    orch = Orchestrator(src, FocusScorer(spec), GPOptimizer(tmp_path / "opt.json"), FAST, tmp_path / "s", tmp_path / "cal.json", clock=SimClock(), wearer="t")
    orch.add_listener(lambda ev: seen.append(ev["calibration_progress"]) if ev["type"] == "state" and ev.get("calibration_progress") else None)
    asyncio.run(orch.run())
    assert seen and seen[-1]["needed"] == FAST.calib_min_windows
    assert max(p["valid_windows"] for p in seen) >= FAST.calib_min_windows
    assert {p["condition"] for p in seen} == {"calibration_fixation", "calibration_breath"}
