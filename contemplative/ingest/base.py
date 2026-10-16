"""Common packet model and source interface.

Every source (OSC, LSL, replay, synthetic) emits :class:`Packet` objects into a
thread-safe queue which the orchestrator drains with :meth:`EEGSource.read`.
Channel order follows Muse / muselsl: TP9, AF7, AF8, TP10.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

CHANNELS = ("TP9", "AF7", "AF8", "TP10")
FS = 256.0

# Packet kinds. ``data`` layout per kind:
#   eeg        : list[float] of 4 raw microvolt samples (TP9, AF7, AF8, TP10)
#   horseshoe  : list[float] of 4 fit values (1 = good, 2 = medium, 4 = bad), Mind Monitor convention
#   blink      : [] (marker)
#   jaw_clench : [] (marker)
#   ppg        : list[float] raw PPG channels
#   heart_rate : [bpm]
#   fnirs      : list[float] (Athena optics; layout unverified)
#   band       : {"band": str, "values": list[float]} device-side band powers (Mind Monitor only)
#   acc / gyro : list[float] of 3
#   other      : anything else, kept for the recording
KINDS = ("eeg", "horseshoe", "blink", "jaw_clench", "ppg", "heart_rate", "fnirs", "band", "acc", "gyro", "other")


@dataclass(slots=True)
class Packet:
    kind: str
    t: float  # seconds, on the orchestrator clock
    data: Any = field(default_factory=list)
    address: str = ""  # original OSC address / LSL stream name when applicable


class EEGSource:
    """Base class. Subclasses push packets with :meth:`_emit` and implement start/stop."""

    name = "base"

    def __init__(self) -> None:
        self._q: queue.Queue[Packet] = queue.Queue()
        self._running = threading.Event()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._running.set()

    def stop(self) -> None:
        self._running.clear()

    @property
    def running(self) -> bool:
        return self._running.is_set()

    # -- data ----------------------------------------------------------------
    def _emit(self, pkt: Packet) -> None:
        self._q.put(pkt)

    def _emit_many(self, pkts: Iterable[Packet]) -> None:
        for p in pkts:
            self._q.put(p)

    def read(self, max_items: int = 100_000) -> list[Packet]:
        """Drain everything queued since the last call (non-blocking)."""
        out: list[Packet] = []
        try:
            while len(out) < max_items:
                out.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return out

    # -- hooks the orchestrator may call; no-ops for hardware sources ---------
    def set_stimulus(self, params: dict[str, float] | None, mode: str) -> None:
        """Synthetic source uses this to know what the wearer is 'seeing'."""

    def tick(self, now: float) -> None:
        """Virtual-clock sources generate data up to ``now`` here."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}
