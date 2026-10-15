import numpy as np

from contemplative.ingest.synthetic import GroundTruth
from contemplative.optimizer import GPOptimizer
from contemplative.visuals.params import AXES, DIM, LITERATURE_PRIOR


def test_first_proposals_are_prior_then_lhs_seeds(tmp_path):
    opt = GPOptimizer(tmp_path / "s.json", n_seeds=3, seed=1)
    x, tag = opt.propose()
    assert tag == "prior"
    assert np.allclose(x, [LITERATURE_PRIOR[a] for a in AXES])
    opt.observe(x, 0.0)
    seeds = []
    for k in range(3):
        x, tag = opt.propose()
        assert tag == f"lhs_seed_{k + 1}"
        assert x.shape == (DIM,) and (0 <= x).all() and (x <= 1).all()
        seeds.append(x)
        opt.observe(x, 0.0)
    # Latin hypercube: each axis has one point per third
    S = np.array(seeds)
    for j in range(DIM):
        assert sorted(np.floor(S[:, j] * 3).astype(int)) == [0, 1, 2]
    x, tag = opt.propose()
    assert tag == "ei"


def test_state_persists_across_instances(tmp_path):
    p = tmp_path / "s.json"
    opt = GPOptimizer(p, n_seeds=2, seed=3)
    for _ in range(3):
        x, _ = opt.propose()
        opt.observe(x, float(np.random.default_rng(0).normal()))
    opt2 = GPOptimizer(p, n_seeds=2, seed=3)
    assert len(opt2.observations) == 3
    assert opt2.seed_cursor == opt.seed_cursor
    assert opt2.seeds == opt.seeds
    x2, tag2 = opt2.propose()
    assert tag2 == "ei"
    assert opt2.best()["x"] is not None


def test_pairwise_rating_adds_pseudo_observation(tmp_path):
    opt = GPOptimizer(tmp_path / "s.json", n_seeds=2)
    assert opt.observe_pairwise(True) is None  # nothing to compare yet
    for y in (0.0, 1.0):
        x, _ = opt.propose()
        opt.observe(x, y)
    obs = opt.observe_pairwise(better=False)
    assert obs is not None and obs.kind == "pairwise"
    assert obs.y < 0.0  # worse than the previous trial's reward (0.0)
    assert obs.noise > 0
    assert len(opt.observations) == 3 and len(opt.rewards) == 2
    assert opt.summary()["n_pairwise"] == 1


def test_recovers_optimum_of_smooth_function(tmp_path):
    """Noise-free ground truth, 30 evaluations: the posterior best should be close to the optimum."""
    gt = GroundTruth()
    for acq in ("ei", "ucb"):
        opt = GPOptimizer(tmp_path / f"{acq}.json", acquisition=acq, n_seeds=4, seed=0)
        for _ in range(30):
            x, _ = opt.propose()
            opt.observe(x, gt(x))
        best = opt.best()
        xb = np.array(best["x"])
        main = gt.main_axes()
        assert gt(gt.optimum) - gt(xb) < 0.15, (acq, gt(xb))
        assert np.linalg.norm(xb[main] - gt.optimum[main]) < 0.3, (acq, xb)
        ls = opt.length_scales()
        assert ls is not None and all(0.2 <= v <= 3.0 for v in ls.values())


def test_handles_nan_free_state_with_duplicate_points(tmp_path):
    opt = GPOptimizer(tmp_path / "s.json", n_seeds=1)
    x0 = np.full(DIM, 0.5)
    for y in (0.1, 0.2, 0.15):
        opt.observe(x0, y)
    x, tag = opt.propose()
    assert tag in ("prior", "lhs_seed_1", "ei") and np.isfinite(x).all()
