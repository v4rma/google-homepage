"""Stream diagnostics: synthetic Mind Monitor streams with defects, both straight
into StreamStats and through a real OSC round trip (fake-muse -> MindMonitorOSCSource)."""

import socket
import time

import pytest

from contemplative.ingest.osc_fake import FakeMindMonitor
from contemplative.ingest.osc_source import MindMonitorOSCSource
from contemplative.ingest.stream_check import StreamStats, run_stream_check
from contemplative.ingest.synthetic import DEFECTS, SyntheticSource
from contemplative.scorer import load_spec


def _stats_for(defects, seconds=12.0, tolerant=False, **kw):
    """Feed a synthetic stream on a virtual clock straight into StreamStats."""
    spec = load_spec()
    if tolerant:
        spec.apply_tolerant()
    src = SyntheticSource(seed=1, defects=defects, blink_rate_hz=0.0, jaw_rate_hz=0.0, bad_fit_rate_hz=0.0, **kw)
    st = StreamStats(spec, listen="test")
    src.start()
    t = 0.0
    while t < seconds:
        t += 0.1
        src.tick(t)
        st.feed(src.read())
    src.stop()
    return st


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_every_defect_is_known_to_the_synthetic_source(defect):
    SyntheticSource(defects=[defect])


def test_unknown_defect_rejected():
    with pytest.raises(ValueError):
        SyntheticSource(defects=["nope"])


def test_clean_stream_is_ok():
    st = _stats_for(set())
    d = st.diagnose()
    assert d["code"] == "ok", d
    assert abs(st.eeg_rate() - 256) < 2
    assert st.horseshoe_last == [1.0, 1.0, 1.0, 1.0]
    assert st.raw_summary()["units_guess"].startswith("microvolts, Mind Monitor")
    assert st.usable_groups() == ["frontal", "temporal"]
    assert "DIAGNOSIS [ok]" in st.format_report()


def test_silent_stream():
    st = _stats_for({"silent"})
    d = st.diagnose()
    assert d["code"] == "no_packets" and d["severity"] == "fatal"
    assert "Firewall" in d["advice"] and "Target IP" in d["advice"]


def test_bands_only_names_the_mind_monitor_setting():
    st = _stats_for({"bands_only"})
    d = st.diagnose()
    assert d["code"] == "bands_only" and d["severity"] == "fatal"
    assert "OSC Stream Brainwaves" in d["advice"]
    assert set(st.bands) == {"delta", "theta", "alpha", "beta", "gamma"}


def test_no_horseshoe_is_unknown_not_bad():
    st = _stats_for({"no_horseshoe"})
    d = st.diagnose()
    assert d["code"] == "fit_unknown" and d["severity"] == "warn"
    assert st.buffer.fit is None and not st.buffer.fit_known
    # nothing was rejected for fit; both groups usable
    assert "fit" not in st.reason_total
    assert st.usable_groups() == ["frontal", "temporal"]


def test_offset_and_scale_detection():
    st = _stats_for({"no_offset"})
    r = st.raw_summary()
    assert abs(max(r["offset_estimate"], key=abs)) < 50
    assert r["units_guess"] == "microvolts, zero-centred"
    assert st.usable_groups() == ["frontal", "temporal"]
    st = _stats_for({"volts"})
    r = st.raw_summary()
    assert r["scale"] == 1e6 and r["units_guess"].startswith("volts")
    assert st.usable_groups() == ["frontal", "temporal"], st.reason_total
    assert st.diagnose()["severity"] != "fatal"


def test_bad_temporal_fit_falls_back_to_frontal():
    st = _stats_for({"bad_fit"})
    d = st.diagnose()
    assert d["code"].startswith("temporal_rejected_fit"), d
    assert d["severity"] == "warn"
    assert st.usable_groups() == ["frontal"]
    assert st.n_windows_fallback > 0


def test_all_bad_fit_is_fatal_and_suggests_fit_tolerant():
    st = _stats_for({"all_bad_fit"})
    d = st.diagnose()
    assert d["code"].startswith("all_rejected_") and d["severity"] == "fatal"
    assert "--fit-tolerant" in d["advice"] or "Reseat" in d["advice"]


def test_medium_fit_needs_tolerant_mode():
    st = _stats_for({"medium_fit"})
    assert "temporal" not in st.usable_groups()
    st = _stats_for({"medium_fit"}, tolerant=True)
    assert st.usable_groups() == ["frontal", "temporal"]
    assert st.diagnose()["code"] == "ok"


def test_dry_electrode_ptp_and_tolerant_threshold():
    st = _stats_for({"dry_ptp"})
    d = st.diagnose()
    assert d["code"] == "all_rejected_ptp" and d["severity"] == "fatal"
    assert "--fit-tolerant" in d["advice"]
    st = _stats_for({"dry_ptp"}, tolerant=True)
    assert st.usable_groups() == ["frontal", "temporal"]


def test_low_rate():
    st = _stats_for({"low_rate"})
    d = st.diagnose()
    assert d["code"] == "low_rate"
    assert 120 < st.eeg_rate() < 136


# -- OSC round trip --------------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _osc_roundtrip(defects, seconds=3.0, tolerant=False):
    port = _free_port()
    spec = load_spec()
    if tolerant:
        spec.apply_tolerant()
    src = SyntheticSource(seed=2, defects=defects, blink_rate_hz=0.0, jaw_rate_hz=0.0, bad_fit_rate_hz=0.0)
    fake = FakeMindMonitor(src, "127.0.0.1", port)
    fake.start(seconds=seconds + 1.0)
    osc = MindMonitorOSCSource("127.0.0.1", port)
    try:
        st = run_stream_check(osc, spec, seconds=seconds, listen=f"127.0.0.1:{port}")
    finally:
        fake.stop()
    return st


@pytest.mark.integration
def test_osc_roundtrip_clean_stream():
    st = _osc_roundtrip(set())
    assert st.n_eeg > 600, st.n_eeg
    assert 240 < st.eeg_rate() < 272
    assert "/muse/eeg" in st.addresses and "/muse/elements/horseshoe" in st.addresses
    assert st.horseshoe_last == [1.0, 1.0, 1.0, 1.0]
    assert st.raw_summary()["units_guess"].startswith("microvolts, Mind Monitor")
    assert st.diagnose()["severity"] != "fatal"


@pytest.mark.integration
def test_osc_roundtrip_defects():
    assert _osc_roundtrip({"silent"}).diagnose()["code"] == "no_packets"
    assert _osc_roundtrip({"bands_only"}).diagnose()["code"] == "bands_only"
    st = _osc_roundtrip({"no_horseshoe"})
    assert st.horseshoe_last is None and st.n_eeg > 600
    st = _osc_roundtrip({"bad_fit"}, seconds=4.0)
    assert st.horseshoe_last == [4.0, 1.0, 1.0, 4.0]
    assert st.usable_groups() == ["frontal"]


def test_osc_source_handles_five_value_eeg_and_unknown_addresses():
    port = _free_port()
    osc = MindMonitorOSCSource("127.0.0.1", port)
    osc.start()
    try:
        from pythonosc.udp_client import SimpleUDPClient

        c = SimpleUDPClient("127.0.0.1", port)
        c.send_message("/muse/eeg", [801.0, 802.0, 803.0, 804.0, 0.0])
        c.send_message("/muse/elements/horseshoe", [1.0, 2.0, 1.0, 4.0])
        c.send_message("/muse/elements/alpha_absolute", [0.5, 0.4, 0.3, 0.2])
        c.send_message("/muse/something_new", [1, 2, 3])
        time.sleep(0.3)
        pk = osc.read()
    finally:
        osc.stop()
    kinds = {p.kind for p in pk}
    assert {"eeg", "horseshoe", "band", "other"} <= kinds
    eeg = [p for p in pk if p.kind == "eeg"][0]
    assert eeg.data == [801.0, 802.0, 803.0, 804.0]
    band = [p for p in pk if p.kind == "band"][0]
    assert band.data["band"] == "alpha" and len(band.data["values"]) == 4
