import numpy as np

from contemplative.ingest import make_source
from contemplative.ingest.base import CHANNELS, Packet
from contemplative.ingest.recorder import SessionRecorder
from contemplative.ingest.replay_source import ReplaySource
from contemplative.ingest.synthetic import GroundTruth, SyntheticSource


def test_recorder_replay_roundtrip(tmp_path):
    rec = SessionRecorder(tmp_path)
    pk = [Packet("eeg", i / 256.0, [800.0 + i, 801.0, 802.0, 803.0]) for i in range(512)]
    pk.append(Packet("horseshoe", 1.0, [1.0, 2.0, 1.0, 1.0], "/muse/elements/horseshoe"))
    pk.append(Packet("blink", 1.5, [1], "/muse/elements/blink"))
    rec.write_packets(pk)
    rec.marker(2.0, "test", {"a": 1})
    rec.close()
    assert (tmp_path / "raw_eeg.csv").exists() and (tmp_path / "markers.csv").exists()

    src = ReplaySource(tmp_path / "raw_eeg.csv")
    src.start()
    src.tick(100.0)
    assert len(src.read()) <= 1  # first tick only anchors the replay clock
    src.tick(101.0)
    out = src.read()
    eeg = [p for p in out if p.kind == "eeg"]
    assert 250 <= len(eeg) <= 260
    assert abs(eeg[0].t - 100.0) < 0.01  # re-timed onto the orchestrator clock
    assert any(p.kind == "horseshoe" for p in out)
    src.tick(103.0)
    out = src.read()
    assert any(p.kind == "blink" for p in out)
    assert src.finished


def test_replay_mind_monitor_export(tmp_path):
    p = tmp_path / "mm.csv"
    hdr = "TimeStamp,Delta_TP9,RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10,HSI_TP9,HSI_AF7,HSI_AF8,HSI_TP10,Elements\n"
    rows = [hdr]
    for i in range(256):
        ts = f"2026-10-15 12:00:{i / 256:06.3f}"
        el = "/muse/elements/blink" if i == 100 else ""
        rows.append(f"{ts},0.5,{800 + i % 5},801,802,803,1,2,1,1,{el}\n")
    p.write_text("".join(rows))
    src = ReplaySource(p)
    src.start()
    src.tick(0.0)
    src.tick(2.0)
    out = src.read()
    kinds = {q.kind for q in out}
    assert "eeg" in kinds and "horseshoe" in kinds and "blink" in kinds
    assert len([q for q in out if q.kind == "eeg"]) == 256


def test_synthetic_source_emits_expected_streams_and_responds_to_stimulus():
    gt = GroundTruth()
    src = SyntheticSource(seed=0, ground_truth=gt)
    src.start()
    src.tick(0.0)
    src.tick(2.0)
    out = src.read()
    kinds = {p.kind for p in out}
    assert {"eeg", "horseshoe", "ppg", "fnirs", "heart_rate"} <= kinds
    eeg = [p for p in out if p.kind == "eeg"]
    assert 500 <= len(eeg) <= 520 and len(eeg[0].data) == len(CHANNELS)
    assert 700 < np.mean([p.data[0] for p in eeg]) < 900  # Muse-like DC offset
    src.set_stimulus(dict(zip(("speed", "fractal", "symmetry", "color", "breath", "texture"), gt.optimum)), "stimulus")
    assert src.describe()["name"] == "synthetic"
    for k in range(1, 40):
        src.tick(2.0 + k)
    src.read()
    assert src.state > 0.7  # relaxed towards f(optimum) = 1


def test_ground_truth_shape():
    gt = GroundTruth()
    assert abs(gt(gt.optimum) - 1.0) < 1e-9
    assert gt(np.zeros(6)) < gt(gt.optimum)
    assert len(gt.main_axes()) == 2


def test_make_source_factory():
    assert make_source("synthetic", seed=1).name == "synthetic"
    import pytest

    with pytest.raises(ValueError):
        make_source("nope")
