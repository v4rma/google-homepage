import numpy as np

from contemplative.visuals.params import AXES, LITERATURE_PRIOR, StimulusParams, to_physical
from contemplative.visuals.temporal_check import check_morph_schedule, schedule


def test_params_roundtrip_and_physical_mapping():
    p = StimulusParams.from_dict(LITERATURE_PRIOR)
    assert np.allclose(StimulusParams.from_vector(p.vector()).vector(), p.vector())
    ph = to_physical(p)
    assert abs(ph["fractal_D"] - 1.3) < 1e-9
    assert 1 <= ph["symmetry_fold"] <= 12
    assert to_physical(StimulusParams.from_vector(np.zeros(6)))["fractal_D"] == 1.1
    assert to_physical(StimulusParams.from_vector(np.ones(6)))["fractal_D"] == 1.7
    assert to_physical(StimulusParams.from_vector(np.ones(6)))["symmetry_fold"] == 12
    assert to_physical(StimulusParams.from_vector(np.zeros(6)))["symmetry_fold"] == 1
    assert list(p.to_dict()) == list(AXES)


def test_morph_schedule_has_no_3_30hz_content():
    res = check_morph_schedule(morph_s=10.0, n_trials=3)
    assert res["ok"], res
    for name, s in res["signals"].items():
        assert s["share_3_30hz"] < 1e-6, name
        assert s["share_below_1hz"] > 0.99, name


def test_morph_floor_is_ten_seconds():
    t, mix, _ = schedule(morph_s=1.0, n_trials=1, baseline_s=20, stimulus_s=20, fs=100.0)
    # the first morph starts at t=20 s; with a 10 s floor it must still be mid-way at 25 s
    i = int(np.searchsorted(t, 25.0))
    assert 0.3 < mix[i] < 0.7
    assert mix[int(np.searchsorted(t, 31.0))] < 1e-6
