"""Writes every packet and marker of a session to disk.

Layout of a session folder::

    raw_eeg.csv      t,TP9,AF7,AF8,TP10
    aux.jsonl        one JSON object per non-EEG packet (horseshoe, blink, ppg, fnirs, ...)
    markers.csv      t,label,payload_json
    scores.csv       t,score,raw,valid,reasons      (written by the orchestrator)
    trials.csv       trial,t_start,params...,reward (written by the orchestrator)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .base import CHANNELS, Packet


class SessionRecorder:
    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._eeg_f = open(self.folder / "raw_eeg.csv", "w", newline="")
        self._eeg = csv.writer(self._eeg_f)
        self._eeg.writerow(["t", *CHANNELS])
        self._aux_f = open(self.folder / "aux.jsonl", "w")
        self._mark_f = open(self.folder / "markers.csv", "w", newline="")
        self._mark = csv.writer(self._mark_f)
        self._mark.writerow(["t", "label", "payload"])
        self.n_eeg = 0

    def write_packets(self, pkts: Iterable[Packet]) -> None:
        for p in pkts:
            if p.kind == "eeg":
                self._eeg.writerow([f"{p.t:.4f}", *(f"{v:.3f}" for v in p.data)])
                self.n_eeg += 1
            else:
                self._aux_f.write(json.dumps({"t": round(p.t, 4), "kind": p.kind, "address": p.address, "data": p.data}) + "\n")

    def marker(self, t: float, label: str, payload: dict[str, Any] | None = None) -> None:
        self._mark.writerow([f"{t:.4f}", label, json.dumps(payload or {})])
        self._mark_f.flush()

    def flush(self) -> None:
        for f in (self._eeg_f, self._aux_f, self._mark_f):
            f.flush()

    def close(self) -> None:
        for f in (self._eeg_f, self._aux_f, self._mark_f):
            try:
                f.close()
            except Exception:
                pass
