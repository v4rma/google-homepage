"""Lab Streaming Layer source (``pip install contemplative[lsl]``; needs liblsl).

Expects a muselsl-style EEG stream: type ``EEG``, 256 Hz, channels
TP9, AF7, AF8, TP10, Right AUX.  If a ``PPG`` stream is present it is read
too.  Whether muselsl / BlueMuse can connect to a Muse S Athena is unverified
(see README); the Muse 2 / Muse S Gen 2 layout is assumed.
"""

from __future__ import annotations

import threading
import time

from .base import EEGSource, Packet


class LSLSource(EEGSource):
    name = "lsl"

    def __init__(self, stream_type: str = "EEG", timeout: float = 10.0, clock=None, with_ppg: bool = True) -> None:
        super().__init__()
        self.stream_type = stream_type
        self.timeout = timeout
        self.with_ppg = with_ppg
        self._clock = clock or time.monotonic
        self._threads: list[threading.Thread] = []
        self.n_eeg = 0

    def start(self) -> None:
        try:
            import pylsl
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("pylsl is not installed; pip install 'contemplative[lsl]' and install liblsl") from e
        super().start()
        streams = pylsl.resolve_byprop("type", self.stream_type, timeout=self.timeout)
        if not streams:
            raise RuntimeError(f"no LSL stream of type {self.stream_type!r} found within {self.timeout}s")
        self._spawn(pylsl.StreamInlet(streams[0], max_chunklen=12), "eeg")
        if self.with_ppg:
            ppg = pylsl.resolve_byprop("type", "PPG", timeout=1.0)
            if ppg:
                self._spawn(pylsl.StreamInlet(ppg[0]), "ppg")

    def _spawn(self, inlet, kind: str) -> None:
        th = threading.Thread(target=self._pull_loop, args=(inlet, kind), daemon=True, name=f"lsl-{kind}")
        th.start()
        self._threads.append(th)

    def _pull_loop(self, inlet, kind: str) -> None:
        import pylsl

        offset: float | None = None
        name = inlet.info().name()
        while self.running:
            chunk, stamps = inlet.pull_chunk(timeout=0.2)
            if not chunk:
                continue
            if offset is None:
                # map LSL clock onto the orchestrator clock once
                offset = self._clock() - (stamps[-1] + inlet.time_correction())
            for row, ts in zip(chunk, stamps):
                t = ts + offset
                if kind == "eeg":
                    self.n_eeg += 1
                    self._emit(Packet("eeg", t, [float(v) for v in row[:4]], name))
                else:
                    self._emit(Packet(kind, t, [float(v) for v in row], name))

    def describe(self):
        return {"name": self.name, "type": self.stream_type, "n_eeg": self.n_eeg}
