"""Fake Mind Monitor: sends a synthetic Muse stream as Mind Monitor OSC messages.

    python -m contemplative fake-muse --target 127.0.0.1:5000 [--synthetic-defect no_horseshoe ...]

Used to exercise the OSC ingest, ``check-stream`` and the fail-soft paths without
hardware, and on the user's PC to separate "nothing reaches Python" (firewall,
wrong target IP) from "the headband stream is unusable": if the fake stream
passes ``check-stream`` locally but Mind Monitor's does not, the problem is on
the network/phone side.
"""

from __future__ import annotations

import threading
import time

from .base import EEGSource, Packet


class FakeMindMonitor:
    def __init__(self, source: EEGSource, host: str = "127.0.0.1", port: int = 5000, speed: float = 1.0) -> None:
        from pythonosc.udp_client import SimpleUDPClient

        self.source = source
        self.client = SimpleUDPClient(host, port)
        self.speed = speed
        self.n_sent = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- packet -> OSC -----------------------------------------------------------
    def send_packet(self, p: Packet) -> None:
        c = self.client
        if p.kind == "eeg":
            c.send_message("/muse/eeg", [float(v) for v in p.data[:4]] + [0.0])  # 5th value: AUX, as Mind Monitor sends
        elif p.kind == "horseshoe":
            c.send_message("/muse/elements/horseshoe", [float(v) for v in p.data[:4]])
        elif p.kind == "blink":
            c.send_message("/muse/elements/blink", [1])
        elif p.kind == "jaw_clench":
            c.send_message("/muse/elements/jaw_clench", [1])
        elif p.kind == "ppg":
            c.send_message("/muse/ppg", [float(v) for v in p.data])
        elif p.kind == "heart_rate":
            c.send_message("/muse/elements/heart_rate", [float(v) for v in p.data])
        elif p.kind == "fnirs":
            c.send_message("/muse/optics", [float(v) for v in p.data])
        elif p.kind == "band":
            c.send_message(f"/muse/elements/{p.data['band']}_absolute", [float(v) for v in p.data["values"]])
        else:
            return
        self.n_sent += 1

    # -- real-time pump -----------------------------------------------------------
    def run(self, seconds: float | None = None) -> None:
        self.source.start()
        t0 = time.monotonic()
        try:
            while not self._stop.is_set():
                now = (time.monotonic() - t0) * self.speed
                if seconds is not None and now >= seconds * self.speed:
                    break
                self.source.tick(now)
                for p in self.source.read():
                    self.send_packet(p)
                time.sleep(0.02)
        finally:
            self.source.stop()

    def start(self, seconds: float | None = None) -> None:
        self._thread = threading.Thread(target=self.run, args=(seconds,), name="fake-muse", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
