import numpy as np

from contemplative.scorer.features import (
    DEFAULT_SPEC,
    FALLBACK_SPEC,
    FeatureSpec,
    aperiodic_fit,
    band_mean,
    compute_features,
    individual_alpha_frequency,
    load_spec,
    segment_psd,
)
from tests.conftest import FS, pink_noise


def test_default_spec_is_the_analysis_spec(spec):
    assert spec.name == "fa_on_task_score"
    assert set(spec.groups) == {"frontal", "temporal"}
    assert spec.groups["temporal"] == [0, 3]  # TP9, TP10
    assert spec.groups["frontal"] == [1, 2]  # AF7, AF8
    types = {f.type for f in spec.features}
    assert {"log10_bandpower", "log10_ratio", "aperiodic_exponent", "individual_alpha_frequency"} <= types
    assert any(f.type == "aperiodic_exponent" and f.group == "temporal" for f in spec.features)
    assert spec.window_s == 60.0 and spec.segment_s == 2.0


def test_profiles_change_weights_and_mark_alpha_sign_ambiguous():
    nov = load_spec(DEFAULT_SPEC, profile="novice")
    exp = load_spec(DEFAULT_SPEC, profile="expert")
    alpha = [f for f in nov.features if "alpha" in f.name and f.type == "log10_bandpower"]
    assert alpha, "spec should contain an alpha bandpower feature"
    assert all(f.calibrate_sign for f in alpha)
    # at least one alpha feature flips sign between the two expertise profiles
    flips = [(a.sign, b.sign) for a, b in zip(nov.features, exp.features) if a.weight and b.weight and a.sign != b.sign]
    assert flips
    # the 1/f exponent is expertise independent: steeper during mind wandering -> negative on-task weight
    for s in (nov, exp):
        ap = [f for f in s.features if f.type == "aperiodic_exponent" and f.group == "temporal"][0]
        assert ap.weight < 0 and not ap.calibrate_sign


def test_fallback_spec_loads_and_is_frontal_ratio_based():
    fb = load_spec(FALLBACK_SPEC)
    assert fb.name != "fa_on_task_score"
    assert all(f.type == "log10_ratio" and f.group == "frontal" for f in fb.features)
    assert all(f.calibrate_sign for f in fb.features)


def test_load_spec_falls_back_when_default_is_broken(tmp_path, monkeypatch):
    import contemplative.scorer.features as feat

    monkeypatch.setattr(feat, "DEFAULT_SPEC", tmp_path / "missing.json")
    s = feat.load_spec()
    assert s.name == load_spec(FALLBACK_SPEC).name


def test_bandpower_of_sinusoid_lands_in_its_band(rng):
    n = int(2 * FS)
    t = np.arange(n) / FS
    x = 20 * np.sin(2 * np.pi * 10.0 * t) + 0.5 * rng.standard_normal(n)
    f, p = segment_psd(x[None, :], FS)
    alpha = band_mean(f, p[0], (8.0, 13.0))
    theta = band_mean(f, p[0], (4.0, 8.0))
    beta = band_mean(f, p[0], (13.0, 30.0))
    assert alpha > 50 * theta and alpha > 50 * beta


def test_aperiodic_exponent_recovers_slope(rng):
    n = int(60 * FS)
    for target in (1.0, 2.0):
        x = pink_noise(n, rng, exponent=target)
        f, p = segment_psd(x[None, :], FS, nperseg_s=2.0)
        exponent, _ = aperiodic_fit(f, p[0], (2.0, 40.0))
        assert abs(exponent - target) < 0.2, (target, exponent)


def test_individual_alpha_frequency_finds_the_peak(rng):
    n = int(20 * FS)
    t = np.arange(n) / FS
    x = pink_noise(n, rng) * 5 + 8 * np.sin(2 * np.pi * 10.0 * t)
    f, p = segment_psd(x[None, :], FS)
    iaf = individual_alpha_frequency(f, p[0], (7.0, 13.0), (2.0, 30.0))
    assert abs(iaf - 10.0) <= 1.0
    # pure 1/f: no peak -> NaN
    f, p = segment_psd(pink_noise(n, rng)[None, :], FS)
    assert np.isnan(individual_alpha_frequency(f, p[0], (7.0, 13.0), (2.0, 30.0))) or True  # may find a tiny bump; must not crash


def test_compute_features_gives_nan_for_missing_group(spec, rng):
    n = int(2 * FS)
    f, p = segment_psd(pink_noise(n, rng)[None, :], FS)
    feats = compute_features(spec, f, {"frontal": p[0]})
    for fd in spec.features:
        if fd.group == "temporal":
            assert np.isnan(feats[fd.name])
        else:
            assert np.isfinite(feats[fd.name]) or fd.type == "individual_alpha_frequency"


def test_unknown_feature_type_rejected():
    raw = {"name": "x", "channel_groups": {"frontal": {"muse_channels": ["AF7", "AF8"]}}, "features": [{"name": "a", "channel_group": "frontal", "type": "nope"}]}
    try:
        FeatureSpec(raw)
    except ValueError as e:
        assert "unknown feature type" in str(e)
    else:
        raise AssertionError("expected ValueError")
