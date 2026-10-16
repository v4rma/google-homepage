import numpy as np

from contemplative.scorer.calibration import MIN_STD, Calibration, CalibrationFitter


def _features(spec, rng, alpha_shift: float, n: int = 12, scale: float = 0.1) -> list[dict]:
    """Synthetic window features around a baseline; alpha features shifted by alpha_shift."""
    base = {f.name: 0.0 for f in spec.features}
    out = []
    for _ in range(n):
        feats = {}
        for fd in spec.features:
            v = base[fd.name] + scale * rng.standard_normal()
            if fd.type == "log10_bandpower" and "alpha" in fd.name:
                v += alpha_shift
            if fd.type == "individual_alpha_frequency":
                v = 10.0  # quantised and constant -> would have zero std
            feats[fd.name] = v
        out.append(feats)
    return out


def test_calibration_picks_alpha_sign_from_data_in_both_directions(spec, rng):
    # novice-like wearer: less alpha during breath focus than during fixation
    fit = CalibrationFitter(spec, "w", profile="novice")
    for f in _features(spec, rng, 0.0):
        fit.add("calibration_fixation", f)
    for f in _features(spec, rng, -0.5):
        fit.add("calibration_breath", f)
    cal = fit.fit()
    alpha = [fd.name for fd in spec.features if fd.type == "log10_bandpower" and "alpha" in fd.name and fd.weight]
    assert all(cal.feature_sign[a] == -1.0 for a in alpha)
    # expert-like wearer: more alpha on task; the same novice profile must flip the sign
    fit = CalibrationFitter(spec, "w", profile="novice")
    for f in _features(spec, rng, 0.0):
        fit.add("calibration_fixation", f)
    for f in _features(spec, rng, +0.5):
        fit.add("calibration_breath", f)
    cal2 = fit.fit()
    assert all(cal2.feature_sign[a] == +1.0 for a in alpha)
    assert any("overrides profile sign" in n for n in cal2.notes)
    assert cal2.separation_d > 1.0


def test_calibration_sets_per_feature_scale_with_floor(spec, rng):
    fit = CalibrationFitter(spec, "w")
    for f in _features(spec, rng, 0.0, scale=0.3):
        fit.add("calibration_fixation", f)
    for f in _features(spec, rng, -0.4, scale=0.3):
        fit.add("calibration_breath", f)
    cal = fit.fit()
    for fd in spec.features:
        assert cal.feature_std[fd.name] >= MIN_STD.get(fd.type, MIN_STD["default"])
        if fd.type == "individual_alpha_frequency":
            assert cal.feature_std[fd.name] == MIN_STD["individual_alpha_frequency"]
        else:
            assert 0.15 < cal.feature_std[fd.name] < 0.6
    # fixation windows are standardised to mean 0 / sd 1
    s = np.array([cal.score(f) for f in fit.fixation])
    assert abs(s.mean()) < 1e-6 and abs(s.std(ddof=1) - 1.0) < 1e-6


def test_auto_profile_chooses_better_separating_weights(spec, rng):
    fit = CalibrationFitter(spec, "w", profile="auto")
    for f in _features(spec, rng, 0.0):
        fit.add("calibration_fixation", f)
    for f in _features(spec, rng, +0.6):
        fit.add("calibration_breath", f)
    cal = fit.fit()
    assert cal.profile in ("novice", "expert")
    assert any("profile chosen automatically" in n for n in cal.notes)


def test_calibration_roundtrip_and_missing_features(spec, rng, tmp_path):
    fit = CalibrationFitter(spec, "w")
    for f in _features(spec, rng, 0.0):
        fit.add("calibration_fixation", f)
    for f in _features(spec, rng, -0.5):
        fit.add("calibration_breath", f)
    cal = fit.fit()
    cal.save(tmp_path / "cal.json")
    cal2 = Calibration.load(tmp_path / "cal.json")
    feats = fit.breath[0]
    assert cal2.score(feats) == cal.score(feats)
    # frontal-only window (temporal NaN) still scores
    partial = {k: (np.nan if any(fd.name == k and fd.group == "temporal" for fd in spec.features) else v) for k, v in feats.items()}
    assert cal2.score(partial) is not None
    assert cal2.score({k: np.nan for k in feats}) is None


def test_too_few_windows_raises(spec, rng):
    fit = CalibrationFitter(spec, "w")
    for f in _features(spec, rng, 0.0, n=3):
        fit.add("calibration_fixation", f)
        fit.add("calibration_breath", f)
    try:
        fit.fit()
    except ValueError as e:
        assert "not enough" in str(e)
    else:
        raise AssertionError
