from .base import CHANNELS, FS, EEGSource, Packet
from .recorder import SessionRecorder
from .replay_source import ReplaySource
from .synthetic import GroundTruth, SyntheticSource


def make_source(kind: str, **kwargs) -> EEGSource:
    """Factory used by the CLI. ``kind`` is one of osc | lsl | replay | synthetic."""
    if kind == "synthetic":
        return SyntheticSource(**kwargs)
    if kind == "replay":
        return ReplaySource(**kwargs)
    if kind == "osc":
        from .osc_source import MindMonitorOSCSource

        return MindMonitorOSCSource(**kwargs)
    if kind == "lsl":
        from .lsl_source import LSLSource

        return LSLSource(**kwargs)
    raise ValueError(f"unknown source kind {kind!r}")


__all__ = [
    "CHANNELS",
    "FS",
    "EEGSource",
    "Packet",
    "SessionRecorder",
    "ReplaySource",
    "SyntheticSource",
    "GroundTruth",
    "make_source",
]
