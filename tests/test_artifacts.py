import numpy as np

from contemplative.scorer.artifacts import check_segment, usable_channels
from contemplative.scorer.features import segment_psd
from contemplative.scorer.score import FocusScorer
from contemplative.scorer.windows import SegmentBuffer
from tests.conftest import FS, eeg_packets, pink_noise


def clean_segment(rng, n=int(2 * FS)):
    return 800.0 + 10.0 * np.vstack([pink_noise(n, rng) for _ in range(4)])


def test_clean_segment_is_kept(spec, rng):
    chk = check_segment(clean_segment(rng), spec)
    assert chk.keep.all() and not chk.reasons


def test_peak_to_peak_rejects_only_the_bad_channel(spec, rng):
    x = clean_segment(rng)
    x[1, 100:160] += 300.0  # blink-sized deflection on AF7
    chk = check_segment(x, spec)
    assert not chk.keep[1] and chk.keep[[0, 2, 3]].all()
    assert "ptp" in chk.reasons[1]


def test_flat_channel_rejected(spec, rng):
    x = clean_segment(rng)
    x[3, :] = 812.0
    chk = check_segment(x, spec)
    assert not chk.keep[3] and "flat" in chk.reasons[3]


def test_markers_drop_frontal_for_blink_and_all_for_jaw(spec, rng):
    x = clean_segment(rng)
    chk = check_segment(x, spec, markers=["blink"])
    assert list(chk.keep) == [True, False, False, True]
    chk = check_segment(x, spec, markers=["jaw_clench"])
    assert not chk.keep.any()


def test_horseshoe_fit_threshold_is_stricter_for_temporal(spec, rng):
    x = clean_segment(rng)
    chk = check_segment(x, spec, fit=[2, 2, 2, 2])  # all medium
    assert list(chk.keep) == [False, True, True, False]
    chk = check_segment(x, spec, fit=[1, 4, 1, 1])
    assert list(chk.keep) == [True, False, True, True]


def test_emg_rejection_in_fallback_spec(fallback_spec, rng):
    n = int(2 * FS)
    x = clean_segment(rng)
    # 30-60 Hz band-limited muscle noise on TP9, small enough to pass the ptp check
    w = np.fft.rfft(rng.standard_normal(n))
    fr = np.fft.rfftfreq(n, 1.0 / FS)
    w[(fr < 30) | (fr > 60)] = 0
    emg = np.fft.irfft(w, n)
    x[0] += 15.0 * emg / emg.std()
    f, p = segment_psd(x - x.mean(axis=1, keepdims=True), FS)
    assert np.ptp(x[0]) < 150
    chk = check_segment(x, fallback_spec, f=f, psd=p)
    assert not chk.keep[0] and "emg" in chk.reasons[0]


def test_usable_channels_respects_min_fraction(spec):
    out = usable_channels(np.array([0.9, 0.9, 0.3, 0.1]), spec)
    assert out == {"frontal": [1], "temporal": [0]}
    out = usable_channels(np.array([0.1, 0.9, 0.9, 0.1]), spec)
    assert "temporal" not in out and out["frontal"] == [1, 2]


def test_window_falls_back_to_frontal_only_when_temporal_fit_is_bad(spec, rng):
    """TP9/TP10 are first class while seated well; when the horseshoe says bad, the
    window is scored from the frontal pair and flagged as a fallback."""
    buf = SegmentBuffer(spec)
    n = int(20 * FS)
    x = 800.0 + 10.0 * np.vstack([pink_noise(n, rng) for _ in range(4)])
    buf.push(eeg_packets(x))
    w = buf.window(2.0, 20.0)
    assert w is not None and w.valid
    assert set(w.groups_used) == {"frontal", "temporal"} and not w.fallback_frontal_only
    assert w.quality()["temporal_kept"] > 0.9

    buf2 = SegmentBuffer(spec)
    pk = eeg_packets(x)
    from contemplative.ingest.base import Packet

    pk.insert(0, Packet("horseshoe", 0.0, [4.0, 1.0, 1.0, 4.0]))
    buf2.push(pk)
    w2 = buf2.window(2.0, 20.0)
    assert w2 is not None and w2.valid
    assert w2.groups_used == ["frontal"] and w2.fallback_frontal_only
    assert w2.quality()["temporal_kept"] == 0.0
    for fd in spec.features:
        if fd.group == "temporal":
            assert np.isnan(w2.features[fd.name])


def test_scorer_counts_valid_and_rejected_windows(spec, rng):
    spec.window_s, spec.hop_s = 10.0, 2.0
    sc = FocusScorer(spec)
    n = int(30 * FS)
    x = 800.0 + 10.0 * np.vstack([pink_noise(n, rng) for _ in range(4)])
    out = sc.push(eeg_packets(x))
    assert out and sc.n_valid >= 5
    # a fully flat stretch gives invalid windows
    flat = np.full((4, int(30 * FS)), 800.0)
    sc.push(eeg_packets(flat, t0=30.0))
    assert sc.n_rejected > 0
    assert sc.last_quality is not None
