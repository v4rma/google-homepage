"""Mind Monitor OSC stream (primary hardware path).

In Mind Monitor: Settings -> OSC Stream Target IP = this machine's LAN IP,
OSC Stream Port = the ``port`` given here (default 5000), then tap the
"stream" icon.  Mind Monitor's documented OSC message set
(https://mind-monitor.com/FAQ.php#oscspec) that we consume:

    /muse/eeg                       TP9 AF7 AF8 TP10 [AUX]   raw EEG, microvolts (0..1682, centre ~800)
    /muse/elements/horseshoe        TP9 AF7 AF8 TP10          1 good, 2 medium, 4 bad
    /muse/elements/touching_forehead  0/1
    /muse/elements/blink            1
    /muse/elements/jaw_clench       1
    /muse/elements/{delta,theta,alpha,beta,gamma}_absolute   1 float (avg) or 4 floats (per channel)
    /muse/elements/{...}_relative, {...}_session_score
    /muse/acc  /muse/gyro           3 floats
    /muse/ppg                       3 floats (Muse 2 / S / Athena)
    /muse/batt                      battery info

Assumptions that need a hardware check (see README):
* Heart rate: Mind Monitor shows HR in-app; whether/where it is sent over OSC
  is not confirmed, so ``/muse/elements/heart_rate`` and ``/muse/hr`` are both accepted.
* Athena fNIRS: no documented address.  Anything under ``/muse/optics*`` or
  ``/muse/fnirs*`` is captured as ``fnirs`` packets and recorded.
* Timestamps: Mind Monitor OSC carries no sample timestamps; we stamp packets
  on arrival and rely on the 256 Hz nominal rate.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .base import EEGSource, Packet

_BANDS = ("delta", "theta", "alpha", "beta", "gamma")


class MindMonitorOSCSource(EEGSource):
    name = "osc"

    def __init__(self, host: str = "0.0.0.0", port: int = 5000, clock=None) -> None:
        super().__init__()
        self.host, self.port = host, port
        self._clock = clock or time.monotonic
        self._server = None
        self._thread: threading.Thread | None = None
        self.n_eeg = 0
        self.last_eeg_t: float | None = None

    def start(self) -> None:
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer

        super().start()
        d = Dispatcher()
        d.map("/muse/eeg", self._on_eeg)
        d.map("/muse/elements/horseshoe", self._on_simple, "horseshoe")
        d.map("/muse/elements/blink", self._on_marker, "blink")
        d.map("/muse/elements/jaw_clench", self._on_marker, "jaw_clench")
        d.map("/muse/ppg", self._on_simple, "ppg")
        d.map("/muse/elements/heart_rate", self._on_simple, "heart_rate")
        d.map("/muse/hr", self._on_simple, "heart_rate")
        d.map("/muse/optics*", self._on_simple, "fnirs")
        d.map("/muse/fnirs*", self._on_simple, "fnirs")
        d.map("/muse/acc", self._on_simple, "acc")
        d.map("/muse/gyro", self._on_simple, "gyro")
        for b in _BANDS:
            d.map(f"/muse/elements/{b}_absolute", self._on_band, b)
        d.set_default_handler(self._on_other, needs_reply_address=False)
        self._server = ThreadingOSCUDPServer((self.host, self.port), d)
        self._thread = threading.Thread(target=self._server.serve_forever, name="osc-server", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        super().stop()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # -- handlers (python-osc passes (address, *args) or (address, fixed, *args)) ----
    def _on_eeg(self, address: str, *args: Any) -> None:
        vals = [float(v) for v in args[:4]]
        if len(vals) < 4:
            return
        t = self._clock()
        self.n_eeg += 1
        self.last_eeg_t = t
        self._emit(Packet("eeg", t, vals, address))

    def _on_simple(self, address: str, fixed: Any, *args: Any) -> None:
        kind = fixed[0] if isinstance(fixed, (list, tuple)) else fixed
        self._emit(Packet(kind, self._clock(), [float(v) for v in args], address))

    def _on_marker(self, address: str, fixed: Any, *args: Any) -> None:
        kind = fixed[0] if isinstance(fixed, (list, tuple)) else fixed
        self._emit(Packet(kind, self._clock(), [], address))

    def _on_band(self, address: str, fixed: Any, *args: Any) -> None:
        band = fixed[0] if isinstance(fixed, (list, tuple)) else fixed
        self._emit(Packet("band", self._clock(), {"band": band, "values": [float(v) for v in args]}, address))

    def _on_other(self, address: str, *args: Any) -> None:
        self._emit(Packet("other", self._clock(), [a if isinstance(a, (int, float, str)) else str(a) for a in args], address))

    def describe(self):
        return {"name": self.name, "listen": f"{self.host}:{self.port}", "n_eeg": self.n_eeg}
