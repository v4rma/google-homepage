"""Replay a recorded session (our ``raw_eeg.csv`` + ``aux.jsonl``) or a Mind
Monitor CSV export, paced by the orchestrator clock.

Our own format: ``raw_eeg.csv`` with header ``t,TP9,AF7,AF8,TP10`` and an
optional ``aux.jsonl`` next to it.  Mind Monitor export: header containing
``RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10`` and ``HSI_*`` fit columns plus an
``Elements`` column with ``/muse/elements/blink`` style markers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .base import CHANNELS, EEGSource, Packet


class ReplaySource(EEGSource):
    name = "replay"

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        super().__init__()
        self.path = Path(path)
        self.loop = loop
        self._packets: list[Packet] = []
        self._idx = 0
        self._t0: float | None = None  # replay-start on the orchestrator clock
        self._duration = 0.0
        self._load()

    # -- loading ------------------------------------------------------------
    def _load(self) -> None:
        with open(self.path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header[:5] == ["t", *CHANNELS]:
                self._load_native(reader)
            elif any(h.startswith("RAW_") for h in header):
                self._load_mind_monitor(header, reader)
            else:
                raise ValueError(f"unrecognised CSV header in {self.path}: {header[:6]}")
        self._packets.sort(key=lambda p: p.t)
        if self._packets:
            t_first = self._packets[0].t
            for p in self._packets:
                p.t -= t_first
            self._duration = self._packets[-1].t

    def _load_native(self, reader) -> None:
        for row in reader:
            self._packets.append(Packet("eeg", float(row[0]), [float(v) for v in row[1:5]]))
        aux = self.path.with_name("aux.jsonl")
        if aux.exists():
            with open(aux) as f:
                for line in f:
                    d = json.loads(line)
                    self._packets.append(Packet(d["kind"], float(d["t"]), d["data"], d.get("address", "")))

    def _load_mind_monitor(self, header: list[str], reader) -> None:
        from datetime import datetime

        col = {h: i for i, h in enumerate(header)}
        raw = [col[f"RAW_{c}"] for c in CHANNELS]
        hsi = [col.get(f"HSI_{c}") for c in CHANNELS]
        el = col.get("Elements")
        ts = col.get("TimeStamp")
        t_prev = None
        k = 0
        for row in reader:
            if not row or len(row) <= max(raw):
                continue
            try:
                vals = [float(row[i]) for i in raw]
            except ValueError:
                continue
            if ts is not None and row[ts]:
                try:
                    t = datetime.fromisoformat(row[ts].replace(" ", "T")).timestamp()
                except ValueError:
                    t = k / 256.0
            else:
                t = k / 256.0
            if t_prev is not None and t <= t_prev:
                t = t_prev + 1 / 256.0  # Mind Monitor timestamps repeat within a second
            t_prev = t
            k += 1
            self._packets.append(Packet("eeg", t, vals))
            if all(h is not None for h in hsi) and row[hsi[0]]:
                try:
                    self._packets.append(Packet("horseshoe", t, [float(row[i]) for i in hsi], "/muse/elements/horseshoe"))
                except ValueError:
                    pass
            if el is not None and row[el]:
                if "blink" in row[el]:
                    self._packets.append(Packet("blink", t, [], row[el]))
                if "jaw" in row[el]:
                    self._packets.append(Packet("jaw_clench", t, [], row[el]))

    # -- playback -----------------------------------------------------------
    def tick(self, now: float) -> None:
        if not self.running or not self._packets:
            return
        if self._t0 is None:
            self._t0 = now
        rel = now - self._t0
        while self._idx < len(self._packets) and self._packets[self._idx].t <= rel:
            p = self._packets[self._idx]
            self._emit(Packet(p.kind, p.t + self._t0, p.data, p.address))
            self._idx += 1
        if self._idx >= len(self._packets) and self.loop:
            self._idx = 0
            self._t0 = now

    @property
    def finished(self) -> bool:
        return not self.loop and self._idx >= len(self._packets)

    def describe(self):
        return {"name": self.name, "path": str(self.path), "duration_s": round(self._duration, 1), "n_packets": len(self._packets)}
