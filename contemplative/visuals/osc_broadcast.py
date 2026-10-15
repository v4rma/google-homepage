"""Broadcast stimulus parameters over OSC so TouchDesigner (or anything else)
can render the same design space later.

Messages (all floats in [0,1] unless stated):

    /visuals/params   speed fractal symmetry color breath texture  morph_s
    /visuals/<axis>   value                       (one message per axis)
    /visuals/mode     "stimulus" | "baseline" | "calibration_fixation" | "calibration_breath" | "idle"
    /visuals/morph_s  seconds the receiver should take to morph to the new values
    /score            current focused-attention z-score (float), ~1 Hz

TouchDesigner: an OSC In CHOP/DAT listening on the target port (default 10000)
receives these directly.
"""

from __future__ import annotations

from .params import AXES, StimulusParams


class OSCBroadcaster:
    def __init__(self, host: str = "127.0.0.1", port: int = 10000, enabled: bool = True) -> None:
        self.enabled = enabled
        self.host, self.port = host, port
        self._client = None
        if enabled:
            from pythonosc.udp_client import SimpleUDPClient

            self._client = SimpleUDPClient(host, port)

    def send_params(self, params: StimulusParams, morph_s: float, mode: str) -> None:
        if not self._client:
            return
        d = params.to_dict()
        self._client.send_message("/visuals/params", [float(d[a]) for a in AXES] + [float(morph_s)])
        for a in AXES:
            self._client.send_message(f"/visuals/{a}", float(d[a]))
        self._client.send_message("/visuals/morph_s", float(morph_s))
        self._client.send_message("/visuals/mode", mode)

    def send_mode(self, mode: str, morph_s: float) -> None:
        if not self._client:
            return
        self._client.send_message("/visuals/mode", mode)
        self._client.send_message("/visuals/morph_s", float(morph_s))

    def send_score(self, score: float) -> None:
        if self._client:
            self._client.send_message("/score", float(score))
