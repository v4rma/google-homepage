"""Stream diagnostics: what is arriving, in what units, and why segments are rejected.

:class:`StreamStats` consumes the same packets the scorer does and keeps

* OSC/packet counts per address and per kind,
* EEG sample rate (overall and per channel, counting finite values),
* raw value ranges and a per-channel offset estimate (Mind Monitor centres on ~800 uV;
  Athena/other software may not),
* horseshoe fit values as received (and whether any arrived at all),
* band-power message counts (to detect "bands only, raw off"),
* per-segment rejection reasons through the scorer's own :class:`SegmentBuffer`.

:meth:`diagnose` turns that into the single most likely cause in plain language;
the CLI ``check-stream``, the dashboard's live panel and the orchestrator's abort
messages all use it.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Any

import numpy as np

from ..scorer.artifacts import usable_channels
from ..scorer.features import CHANNELS, FeatureSpec
from ..scorer.windows import SegmentBuffer
from .base import FS, Packet

NOMINAL_RATE = FS
FIRMWARE_RAW_MAX = 1682.815  # Mind Monitor's documented raw range 0..1682.815 uV


class StreamStats:
    def __init__(self, spec: FeatureSpec, listen: str = "", buffer: SegmentBuffer | None = None) -> None:
        self.spec = spec
        self.listen = listen  # "host:port" for messages
        # a shared buffer (the scorer's) avoids computing every PSD twice; the owner
        # then passes the new segments to feed()
        self.shared = buffer is not None
        self.buffer = buffer or SegmentBuffer(spec)
        self.addresses: Counter[str] = Counter()
        self.kinds: Counter[str] = Counter()
        self.t_first: float | None = None
        self.t_last: float | None = None
        self.eeg_t_first: float | None = None
        self.eeg_t_last: float | None = None
        self.n_eeg = 0
        self.n_finite = np.zeros(4, dtype=int)
        self.raw_min = np.full(4, np.inf)
        self.raw_max = np.full(4, -np.inf)
        self._recent: deque[list[float]] = deque(maxlen=int(10 * FS))
        self.horseshoe_last: list[float] | None = None
        self.horseshoe_hist: list[Counter[float]] = [Counter() for _ in range(4)]
        self.bands: Counter[str] = Counter()
        self.band_last: dict[str, list[float]] = {}
        self.n_segments = 0
        self.kept = np.zeros(4, dtype=int)
        self.reasons: list[Counter[str]] = [Counter() for _ in range(4)]
        self.reason_total: Counter[str] = Counter()
        self.n_windows = 0
        self.n_windows_valid = 0
        self.n_windows_fallback = 0
        self._last_window_t: float | None = None

    # -- intake ------------------------------------------------------------------------
    def feed(self, packets: list[Packet], segments=None) -> None:
        eeg_rows: list[list[float]] = []
        for p in packets:
            self.kinds[p.kind] += 1
            self.addresses[p.address or f"<{p.kind}>"] += 1
            if self.t_first is None:
                self.t_first = p.t
            self.t_last = p.t
            if p.kind == "eeg":
                self.n_eeg += 1
                if self.eeg_t_first is None:
                    self.eeg_t_first = p.t
                self.eeg_t_last = p.t
                if len(p.data) >= 4:
                    eeg_rows.append(p.data[:4])
            elif p.kind == "horseshoe" and len(p.data) >= 4:
                self.horseshoe_last = [float(x) for x in p.data[:4]]
                for i in range(4):
                    self.horseshoe_hist[i][float(p.data[i])] += 1
            elif p.kind == "band" and isinstance(p.data, dict):
                self.bands[p.data.get("band", "?")] += 1
                self.band_last[p.data.get("band", "?")] = list(p.data.get("values", []))
        if eeg_rows:
            a = np.asarray(eeg_rows, dtype=float)
            fin = np.isfinite(a)
            self.n_finite += fin.sum(axis=0)
            af = np.where(fin, a, np.nan)
            with np.errstate(all="ignore"):
                self.raw_min = np.fmin(self.raw_min, np.nanmin(af, axis=0))
                self.raw_max = np.fmax(self.raw_max, np.nanmax(af, axis=0))
            self._recent.extend(eeg_rows)
        if segments is None:
            segments = [] if self.shared else self.buffer.push(packets)
        for seg in segments:
            self.n_segments += 1
            self.kept += seg.keep
            for i, rs in seg.reasons.items():
                for r in rs:
                    self.reasons[i][r] += 1
                    self.reason_total[r] += 1
        # window-level validity every hop, over the spec window (or what exists so far)
        if self.eeg_t_last is not None and (self._last_window_t is None or self.eeg_t_last - self._last_window_t >= self.spec.hop_s):
            self._last_window_t = self.eeg_t_last
            win = min(self.spec.window_s, max(self.eeg_t_last - (self.eeg_t_first or self.eeg_t_last), 4.0))
            wf = self.buffer.window(self.eeg_t_last - win, self.eeg_t_last)
            if wf is not None:
                self.n_windows += 1
                self.n_windows_valid += int(wf.valid)
                self.n_windows_fallback += int(wf.fallback_frontal_only)

    # -- derived -------------------------------------------------------------------------
    @property
    def duration(self) -> float:
        return 0.0 if self.t_first is None or self.t_last is None else self.t_last - self.t_first

    def eeg_rate(self) -> float | None:
        if self.eeg_t_first is None or self.eeg_t_last is None or self.eeg_t_last <= self.eeg_t_first or self.n_eeg < 2:
            return None
        return (self.n_eeg - 1) / (self.eeg_t_last - self.eeg_t_first)

    def eeg_rate_per_channel(self) -> list[float | None]:
        r = self.eeg_rate()
        if r is None:
            return [None] * 4
        return [float(r * self.n_finite[i] / max(self.n_eeg, 1)) for i in range(4)]

    def raw_summary(self) -> dict[str, Any]:
        if not self._recent:
            return {"n": 0}
        a = np.asarray(self._recent, dtype=float)
        med = np.nanmedian(a, axis=0)
        mad = np.nanmedian(np.abs(a - med), axis=0)
        return {
            "n": int(a.shape[0]),
            "min": [None if not np.isfinite(v) else round(float(v), 3) for v in self.raw_min],
            "max": [None if not np.isfinite(v) else round(float(v), 3) for v in self.raw_max],
            "median": [round(float(v), 3) for v in med],
            "mad": [round(float(v), 4) for v in mad],
            "offset_estimate": [round(float(v), 1) for v in med],
            "scale": self.buffer.raw_scale,
            "units_guess": self.units_guess(med, mad),
            "note": self.buffer.scale_note,
        }

    @staticmethod
    def units_guess(med: np.ndarray, mad: np.ndarray) -> str:
        m, d = float(np.nanmedian(med)), float(np.nanmedian(mad))
        if not np.isfinite(d) or d == 0.0:
            return "constant"
        if d < 0.01:
            return "volts (rescaled x1e6)"
        if d > 5000:
            return "unknown (spread too large for microvolts)"
        if 400 < m < 1200:
            return "microvolts, Mind Monitor ~800 offset"
        if abs(m) < 200:
            return "microvolts, zero-centred"
        return f"microvolts, offset {m:.0f}"

    def kept_fraction(self) -> list[float]:
        return [float(self.kept[i] / self.n_segments) if self.n_segments else 0.0 for i in range(4)]

    def usable_groups(self) -> list[str]:
        if not self.n_segments:
            return []
        return sorted(usable_channels(np.asarray(self.kept_fraction()), self.spec))

    def summary(self) -> dict[str, Any]:
        hs = None
        if self.horseshoe_last is not None:
            hs = {
                "last": self.horseshoe_last,
                "n": int(self.kinds.get("horseshoe", 0)),
                "histogram": [{str(int(k)) if float(k).is_integer() else str(k): n for k, n in sorted(h.items())} for h in self.horseshoe_hist],
            }
        return {
            "listen": self.listen,
            "duration_s": round(self.duration, 1),
            "n_packets": int(sum(self.kinds.values())),
            "addresses": dict(self.addresses.most_common()),
            "kinds": dict(self.kinds),
            "eeg": {
                "n": self.n_eeg,
                "rate_hz": None if self.eeg_rate() is None else round(self.eeg_rate(), 1),
                "rate_per_channel_hz": [None if r is None else round(r, 1) for r in self.eeg_rate_per_channel()],
                "nominal_hz": NOMINAL_RATE,
                "raw": self.raw_summary(),
            },
            "horseshoe": hs,
            "bands": {"n": dict(self.bands), "last": self.band_last},
            "markers": {"blink": int(self.kinds.get("blink", 0)), "jaw_clench": int(self.kinds.get("jaw_clench", 0))},
            "segments": {
                "n": self.n_segments,
                "kept_fraction": {CHANNELS[i]: round(f, 2) for i, f in enumerate(self.kept_fraction())},
                "reasons": dict(self.reason_total.most_common()),
                "reasons_per_channel": {CHANNELS[i]: dict(self.reasons[i].most_common()) for i in range(4)},
                "usable_groups": self.usable_groups(),
            },
            "windows": {"n": self.n_windows, "valid": self.n_windows_valid, "frontal_only": self.n_windows_fallback},
            "spec": {"name": self.spec.name, "tolerant": self.spec.tolerant, "max_fit": self.spec.group_max_fit, "ptp_max_uv": self.spec.artifact.get("per_channel_peak_to_peak_max_uv", 150.0)},
            "diagnosis": self.diagnose(),
        }

    # -- the single most likely cause ---------------------------------------------------
    def diagnose(self) -> dict[str, str]:
        """Return {"code", "severity" ('ok' | 'warn' | 'fatal'), "cause", "advice"}."""
        where = f" on {self.listen}" if self.listen else ""
        n_pkt = sum(self.kinds.values())
        if n_pkt == 0:
            return _d(
                "no_packets",
                "fatal",
                f"no OSC packets received at all{where}",
                "Check Mind Monitor -> Settings -> OSC Stream Target IP is THIS computer's LAN IP (ipconfig / ifconfig) and the port matches; phone and PC must be on the same Wi-Fi (no guest/client-isolation network); on Windows allow python.exe through Windows Defender Firewall for inbound UDP (or add a rule for the port), then tap the stream icon in Mind Monitor. Run 'python -m contemplative fake-muse' on this PC: if check-stream then passes, the problem is network/phone side.",
            )
        if self.n_eeg == 0:
            if self.bands:
                return _d(
                    "bands_only",
                    "fatal",
                    "band-power messages arrive but no raw EEG (/muse/eeg)",
                    "Raw streaming is off in Mind Monitor. Settings -> 'OSC Stream Brainwaves' -> set to 'All' (raw EEG is required: the 1/f exponent and alpha-frequency features cannot be computed from band powers). Then restart the stream.",
                )
            return _d(
                "no_eeg",
                "fatal",
                f"OSC packets arrive ({', '.join(a for a, _ in self.addresses.most_common(4))}) but no /muse/eeg",
                "Enable raw EEG in Mind Monitor (Settings -> OSC Stream Brainwaves -> All) and make sure the headband is connected (horseshoe visible in the app). If the addresses above do not start with /muse/, a different sender is targeting this port.",
            )
        rate = self.eeg_rate() or 0.0
        if self.duration >= 3.0 and rate < 0.7 * NOMINAL_RATE:
            return _d(
                "low_rate",
                "fatal" if rate < 0.5 * NOMINAL_RATE else "warn",
                f"raw EEG arrives at {rate:.0f} Hz instead of {NOMINAL_RATE:.0f} Hz",
                "Wi-Fi packet loss or the phone is throttling the app: keep Mind Monitor in the foreground with the screen on, move the phone closer to the router / PC, disable battery optimisation for Mind Monitor. Spectral features are unreliable below ~200 Hz.",
            )
        raw = self.raw_summary()
        if raw.get("n", 0) and raw["units_guess"] == "constant":
            return _d("flat_stream", "fatal", "raw EEG values are constant", "The headband is probably off the head or disconnected; check the horseshoe in Mind Monitor.")
        if raw.get("n", 0) and raw["units_guess"].startswith("unknown"):
            return _d("units", "fatal", f"raw EEG values are not in a plausible microvolt range (median {raw['median']}, spread {raw['mad']})", "Check which app/version is streaming; the scorer expects microvolts (Mind Monitor: 0..1682 with ~800 offset).")
        if self.n_segments >= 3:
            frac = self.kept_fraction()
            min_frac = float(self.spec.artifact.get("min_fraction_segments_kept_per_channel", 0.5))
            if max(frac) < min_frac:
                top, n = _top_reason(self.reason_total)
                return _d(f"all_rejected_{top}", "fatal", f"every channel is rejected in most segments (top reason '{top}', {n} channel-segments; kept {', '.join(f'{CHANNELS[i]} {100 * frac[i]:.0f}%' for i in range(4))})", _REJECT_ADVICE.get(top, "Inspect the per-channel reasons above."))
            temporal = [frac[i] for i in self.spec.groups.get("temporal", [])]
            if temporal and max(temporal) < min_frac:
                rs = Counter()
                for i in self.spec.groups["temporal"]:
                    rs.update(self.reasons[i])
                top, _ = _top_reason(rs)
                return _d(f"temporal_rejected_{top}", "warn", f"TP9/TP10 are unusable (top reason '{top}'); scoring will fall back to the frontal pair", _REJECT_ADVICE.get(top, ""))
        if self.horseshoe_last is None and self.duration >= 3.0:
            return _d("fit_unknown", "warn", "no horseshoe (fit) messages received; fit is treated as unknown, not bad", "Scoring proceeds without fit-based rejection. Mind Monitor normally sends /muse/elements/horseshoe at ~10 Hz; if this is Athena, the app may not emit it. Rely on the peak-to-peak/flat checks and the kept-fraction display.")
        if raw.get("note"):
            return _d("units_note", "warn", raw["note"], "Nothing to do unless features look wrong; the offset is removed per segment.")
        if self.n_segments == 0:
            return _d("warming_up", "warn", f"EEG arriving at {rate:.0f} Hz, waiting for the first {self.spec.segment_s:.0f}-s segment", "")
        return _d("ok", "ok", f"raw EEG at {rate:.0f} Hz, {self.n_segments} segments, usable groups: {', '.join(self.usable_groups()) or 'none yet'}", "")

    def format_report(self) -> str:
        s = self.summary()
        lines = [f"stream check{(' on ' + self.listen) if self.listen else ''}: {s['duration_s']} s, {s['n_packets']} packets"]
        lines.append("  per address:")
        for a, n in s["addresses"].items():
            lines.append(f"    {a:36s} {n:7d}  ({n / max(s['duration_s'], 1e-9):7.1f}/s)")
        e = s["eeg"]
        lines.append(f"  raw EEG (/muse/eeg): {e['n']} samples, {e['rate_hz'] or 0} Hz (nominal {e['nominal_hz']:.0f}); per channel {e['rate_per_channel_hz']}")
        r = e["raw"]
        if r.get("n"):
            lines.append(f"  raw values: min {r['min']} max {r['max']} median {r['median']} -> {r['units_guess']}" + (f"; {r['note']}" if r.get("note") else ""))
        if s["horseshoe"]:
            lines.append(f"  horseshoe: {s['horseshoe']['n']} messages, last {s['horseshoe']['last']}, per channel {s['horseshoe']['histogram']}")
        else:
            lines.append("  horseshoe: none received (fit unknown)")
        lines.append(f"  band-power messages: {s['bands']['n'] or 'none'}")
        lines.append(f"  markers: blink {s['markers']['blink']}, jaw_clench {s['markers']['jaw_clench']}")
        g = s["segments"]
        lines.append(f"  segments: {g['n']}, kept {g['kept_fraction']}, usable groups {g['usable_groups'] or 'none'}")
        lines.append(f"  rejection reasons: {g['reasons'] or 'none'}; per channel {g['reasons_per_channel']}")
        lines.append(f"  windows: {s['windows']['valid']} valid of {s['windows']['n']} ({s['windows']['frontal_only']} frontal-only); spec {s['spec']['name']}{' (fit-tolerant)' if s['spec']['tolerant'] else ''}")
        d = s["diagnosis"]
        lines.append(f"  DIAGNOSIS [{d['severity']}] {d['cause']}")
        if d["advice"]:
            lines.append(f"    -> {d['advice']}")
        return "\n".join(lines)


_REJECT_ADVICE = {
    "fit": "Horseshoe says the electrodes are not seated (temporal needs fit 1, frontal 1-2). Reseat the band: pull the hair out from under TP9/TP10 behind the ears, moisten the skin slightly, tighten a notch. Or run with --fit-tolerant to accept medium fit.",
    "ptp": "Peak-to-peak above the threshold on every channel: movement, muscle tension, mains hum or a loose band. Sit still, relax the jaw and forehead, keep the phone/charger cables away; --fit-tolerant raises the threshold to 300 uV for dry electrodes.",
    "flat": "Channels are flat: the headband is off the head, powered off, or the stream sends constants.",
    "emg": "Broadband 30-60 Hz power: jaw/temporalis tension or a very loose temporal electrode. Relax the jaw, reseat behind the ears.",
    "jaw": "Mind Monitor reports continuous jaw clenching: relax the jaw; if it persists with a relaxed jaw the detector is being triggered by electrode noise - reseat.",
    "blink": "Continuous blink markers: frontal channels dropped. Rest the eyes or check that AF7/AF8 make contact.",
}


_REASON_PRIORITY = ("fit", "flat", "ptp", "emg", "jaw", "blink")


def _top_reason(c: Counter) -> tuple[str, int]:
    """Most frequent reason, preferring the more actionable one (fit > flat > ptp ...)
    when counts are close: a segment with horseshoe 4 usually also fails ptp."""
    if not c:
        return "?", 0
    best, n_best = c.most_common(1)[0]
    for r in _REASON_PRIORITY:
        if r in c and c[r] >= 0.5 * n_best:
            return r, c[r]
    return best, n_best


def _d(code: str, severity: str, cause: str, advice: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "cause": cause, "advice": advice}


def run_stream_check(source, spec: FeatureSpec, seconds: float = 20.0, listen: str = "", progress=None, clock=None, dt: float = 0.1) -> StreamStats:
    """Blocking check: start ``source``, feed packets for ``seconds`` real seconds, stop it."""
    import time as _time

    stats = StreamStats(spec, listen)
    now_fn = clock or _time.monotonic
    source.start()
    t0 = now_fn()
    last_prog = t0
    try:
        while True:
            now = now_fn()
            source.tick(now)
            pk = source.read()
            if pk:
                stats.feed(pk)
            if progress and now - last_prog >= 1.0:
                last_prog = now
                progress(stats, now - t0)
            if now - t0 >= seconds:
                break
            _time.sleep(dt)
    finally:
        source.stop()
    return stats
