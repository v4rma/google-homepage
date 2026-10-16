"""Integration test: the complete closed loop on a virtual clock.

Synthetic wearer with hidden ground truth -> recorder -> scorer (default spec,
calibration included) -> reward -> GP optimizer -> next stimulus, 20 trials.
Asserts that the posterior optimum is within tolerance of the ground truth and
better than the literature prior the optimizer started from.
"""

import json

import pytest

from contemplative.session.simulate import run_simulation


@pytest.mark.integration
def test_closed_loop_recovers_ground_truth(tmp_path):
    res = run_simulation(n_trials=20, seed=3, profile="novice", work_dir=tmp_path, verbose=False, regret_tol=0.30, dist_tol=0.50)
    assert res["n_rewards"] == 20
    assert res["calibration"]["n"][0] >= 5 and res["calibration"]["n"][1] >= 5
    # novice wearer: calibration must pick the negative alpha sign
    signs = res["calibration"]["signs"]
    assert any(k.startswith("temporal_alpha") or "alpha_abs" in k for k in signs), signs
    assert res["f_best"] > res["f_prior"]
    assert res["regret"] <= res["regret_tol"], res
    assert res["distance_main_axes"] <= res["dist_tol"], res
    assert res["ok"]
    # session artefacts
    sess = tmp_path / "sessions" / "sim-00"
    for name in ("raw_eeg.csv", "aux.jsonl", "markers.csv", "scores.csv", "trials.jsonl", "session.json", "summary.json", "calibration.json", "optimizer_state.json"):
        assert (sess / name).exists(), name
    trials = [json.loads(l) for l in (sess / "trials.jsonl").read_text().splitlines()]
    assert len(trials) == 20 and all("reward" in t for t in trials)
