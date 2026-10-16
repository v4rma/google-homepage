"""Bayesian optimisation over the 6 stimulus axes in [0,1]^6.

* GP with ARD Matérn-5/2 kernel + learned white noise (scikit-learn).
* Acquisition: Expected Improvement (default) or UCB, maximised over a Sobol
  candidate set followed by L-BFGS-B refinement of the best candidates.
* Warm start: the literature prior point first, then ``n_seeds`` Latin-
  hypercube seeds, then model-based proposals.
* Persistence: all observations (and the seed sequence) are stored in a JSON
  file so the posterior carries across sessions.
* Pairwise ratings ("this trial felt better/worse than the last one") are
  folded in as a *second, noisier observation* at the rated point: the rating
  is converted to a pseudo-reward ``y_prev ± margin`` with noise variance
  ``pairwise_noise``.  This keeps a single-output GP while letting comparisons
  pull the posterior (see report Section 2, "comparisons beat absolute scores").
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from ..visuals.params import AXES, DIM, LITERATURE_PRIOR


@dataclass
class Observation:
    x: list[float]
    y: float
    noise: float = 0.0  # extra per-observation noise variance (added to the learned white noise)
    kind: str = "reward"  # "reward" | "pairwise"
    session: str = ""
    trial: int = -1
    t: float = field(default_factory=time.time)


class GPOptimizer:
    def __init__(
        self,
        state_path: str | Path | None = None,
        acquisition: str = "ei",
        n_seeds: int = 4,
        ucb_kappa: float = 2.0,
        ei_xi: float = 0.1,
        pairwise_margin: float | None = None,
        pairwise_noise: float = 0.5,
        seed: int = 0,
        n_candidates: int = 2048,
        init_length_scale: float = 0.5,
        length_scale_bounds: tuple[float, float] = (0.2, 3.0),
        best_kappa: float = 1.0,
    ) -> None:
        self.init_length_scale = init_length_scale
        self.length_scale_bounds = length_scale_bounds
        self.best_kappa = best_kappa
        self.state_path = Path(state_path) if state_path else None
        self.acquisition = acquisition
        self.n_seeds = n_seeds
        self.ucb_kappa = ucb_kappa
        self.ei_xi = ei_xi
        self.pairwise_margin = pairwise_margin
        self.pairwise_noise = pairwise_noise
        self.rng = np.random.default_rng(seed)
        self.n_candidates = n_candidates
        self.observations: list[Observation] = []
        self.seeds: list[list[float]] = []
        self.seed_cursor = 0
        self._gp: GaussianProcessRegressor | None = None
        self._dirty = True
        if self.state_path and self.state_path.exists():
            self.load(self.state_path)
        if not self.seeds:
            self.seeds = self._make_seeds()

    # -- seeds -----------------------------------------------------------------
    def _make_seeds(self) -> list[list[float]]:
        prior = [LITERATURE_PRIOR[a] for a in AXES]
        lhs = qmc.LatinHypercube(d=DIM, seed=int(self.rng.integers(1 << 31))).random(self.n_seeds)
        return [prior] + [list(map(float, row)) for row in lhs]

    # -- persistence -----------------------------------------------------------
    def save(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else self.state_path
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "axes": list(AXES),
            "acquisition": self.acquisition,
            "seeds": self.seeds,
            "seed_cursor": self.seed_cursor,
            "observations": [asdict(o) for o in self.observations],
            "saved": time.time(),
        }
        with open(p, "w") as f:
            json.dump(state, f, indent=1)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            s = json.load(f)
        if s.get("axes") != list(AXES):
            raise ValueError("optimizer state axes do not match the current design space")
        self.seeds = s.get("seeds", [])
        self.seed_cursor = int(s.get("seed_cursor", 0))
        self.observations = [Observation(**o) for o in s.get("observations", [])]
        self._dirty = True

    # -- observations ------------------------------------------------------------
    @property
    def rewards(self) -> list[Observation]:
        return [o for o in self.observations if o.kind == "reward"]

    def observe(self, x, y: float, session: str = "", trial: int = -1) -> None:
        self.observations.append(Observation(list(map(float, np.clip(x, 0, 1))), float(y), 0.0, "reward", session, trial))
        self._dirty = True
        self.save()

    def observe_pairwise(self, better: bool, session: str = "") -> Observation | None:
        """Fold a 'better/worse than the previous trial' rating in as a pseudo-observation."""
        rew = self.rewards
        if len(rew) < 2:
            return None
        last, prev = rew[-1], rew[-2]
        ys = np.array([o.y for o in rew])
        margin = self.pairwise_margin if self.pairwise_margin is not None else max(0.5 * float(ys.std()), 0.1)
        y = prev.y + (margin if better else -margin)
        obs = Observation(list(last.x), float(y), self.pairwise_noise, "pairwise", session, last.trial)
        self.observations.append(obs)
        self._dirty = True
        self.save()
        return obs

    # -- model -------------------------------------------------------------------
    def _fit(self) -> GaussianProcessRegressor | None:
        if not self._dirty and self._gp is not None:
            return self._gp
        if len(self.observations) < 2:
            self._gp = None
            self._dirty = False
            return None
        X = np.array([o.x for o in self.observations])
        y = np.array([o.y for o in self.observations])
        alpha = np.array([o.noise for o in self.observations]) + 1e-6
        # With 15-25 noisy points in 6-D, maximum-likelihood hyperparameters run to
        # the bounds (all axes "irrelevant", or all length scales tiny = memorising
        # noise).  We therefore use MAP estimation: log-normal priors on the ARD
        # length scales (centred on 0.5 of the unit cube), amplitude and noise.
        # The report's premise is that the smoothness prior carries most of the
        # information at this sample size.
        kernel = ConstantKernel(1.0, (0.05, 20.0)) * Matern(
            length_scale=np.full(DIM, self.init_length_scale), length_scale_bounds=self.length_scale_bounds, nu=2.5
        ) + WhiteKernel(0.1, (1e-3, 2.0))
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True, optimizer=self._map_optimizer, n_restarts_optimizer=3, random_state=int(self.rng.integers(1 << 31))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            gp.fit(X, y)
        self._gp = gp
        self._dirty = False
        return gp

    # theta layout (log-space): [amplitude, ls_1..ls_DIM, noise]
    _PRIOR_MU = np.concatenate([[np.log(1.0)], np.full(DIM, np.log(0.5)), [np.log(0.1)]])
    _PRIOR_SD = np.concatenate([[1.0], np.full(DIM, 0.6), [1.0]])

    def _map_optimizer(self, obj_func, initial_theta, bounds):
        """sklearn ``optimizer`` hook: minimise -log marginal likelihood - log prior(theta)."""

        def penalised(theta):
            val, grad = obj_func(theta, eval_gradient=True)
            z = (theta - self._PRIOR_MU) / self._PRIOR_SD
            return val + 0.5 * float(z @ z), grad + z / self._PRIOR_SD

        res = minimize(penalised, initial_theta, jac=True, method="L-BFGS-B", bounds=bounds)
        return res.x, float(res.fun)

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        gp = self._fit()
        X = np.atleast_2d(X)
        if gp is None:
            return np.zeros(len(X)), np.ones(len(X))
        mu, sd = gp.predict(X, return_std=True)
        return mu, sd

    def length_scales(self) -> dict[str, float] | None:
        gp = self._fit()
        if gp is None:
            return None
        ls = gp.kernel_.k1.k2.length_scale
        return {a: float(v) for a, v in zip(AXES, np.atleast_1d(ls))}

    # -- acquisition -------------------------------------------------------------
    def _acq(self, X: np.ndarray, gp: GaussianProcessRegressor, y_best: float) -> np.ndarray:
        mu, sd = gp.predict(X, return_std=True)
        sd = np.maximum(sd, 1e-9)
        if self.acquisition == "ucb":
            return mu + self.ucb_kappa * sd
        z = (mu - y_best - self.ei_xi) / sd
        return (mu - y_best - self.ei_xi) * norm.cdf(z) + sd * norm.pdf(z)

    def _candidates(self) -> np.ndarray:
        sob = qmc.Sobol(d=DIM, scramble=True, seed=int(self.rng.integers(1 << 31)))
        C = sob.random(self.n_candidates)
        if self.observations:
            X = np.array([o.x for o in self.observations])
            jitter = X[self.rng.integers(len(X), size=64)] + self.rng.normal(0, 0.05, size=(64, DIM))
            C = np.vstack([C, np.clip(jitter, 0, 1)])
        return C

    def _maximise(self, f) -> tuple[np.ndarray, float]:
        C = self._candidates()
        vals = f(C)
        top = C[np.argsort(vals)[-5:]]
        best_x, best_v = top[-1], float(vals.max())
        for x0 in top:
            res = minimize(lambda x: -float(f(x[None, :])[0]), x0, method="L-BFGS-B", bounds=[(0, 1)] * DIM, options={"maxiter": 50})
            if res.success and -res.fun > best_v:
                best_x, best_v = np.clip(res.x, 0, 1), float(-res.fun)
        return best_x, best_v

    def propose(self) -> tuple[np.ndarray, str]:
        """Next parameters and a tag describing how they were chosen."""
        if self.seed_cursor < len(self.seeds):
            x = np.array(self.seeds[self.seed_cursor])
            tag = "prior" if self.seed_cursor == 0 else f"lhs_seed_{self.seed_cursor}"
            self.seed_cursor += 1
            self.save()
            return x, tag
        gp = self._fit()
        if gp is None:
            return self.rng.random(DIM), "random"
        # "noisy EI" incumbent: best posterior mean at an observed point rather
        # than the best (noise-inflated) observed value.
        X_obs = np.array([o.x for o in self.observations])
        y_best = float(gp.predict(X_obs).max())
        x, _ = self._maximise(lambda X: self._acq(X, gp, y_best))
        return x, self.acquisition

    def best(self, kappa: float | None = None) -> dict:
        """Current estimate of the optimum: argmax of the posterior lower bound ``mu - kappa * sd``.

        ``kappa`` (default ``self.best_kappa``) penalises uncertainty so the
        recommendation does not drift into unexplored corners; 0 gives the plain
        posterior-mean argmax.
        """
        kappa = self.best_kappa if kappa is None else kappa
        gp = self._fit()
        rew = self.rewards
        if gp is None:
            if rew:
                b = max(rew, key=lambda o: o.y)
                return {"x": b.x, "mean": b.y, "std": None, "source": "observed"}
            return {"x": [LITERATURE_PRIOR[a] for a in AXES], "mean": None, "std": None, "source": "prior"}

        def lcb(X):
            mu, sd = gp.predict(X, return_std=True)
            return mu - kappa * sd

        x, _ = self._maximise(lcb)
        mu, sd = gp.predict(x[None, :], return_std=True)
        return {"x": list(map(float, x)), "mean": float(mu[0]), "std": float(sd[0]), "source": f"posterior_mean-{kappa:g}sd"}

    def summary(self) -> dict:
        rew = self.rewards
        return {
            "n_observations": len(self.observations),
            "n_rewards": len(rew),
            "n_pairwise": len(self.observations) - len(rew),
            "seeds_remaining": max(0, len(self.seeds) - self.seed_cursor),
            "acquisition": self.acquisition,
            "length_scales": self.length_scales(),
            "best": self.best(),
        }
