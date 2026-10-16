"""Real and virtual clocks so the orchestrator runs identically live and headless."""

from __future__ import annotations

import asyncio
import time


class RealClock:
    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, dt: float) -> None:
        await asyncio.sleep(dt)

    wall_speed = 1.0


class SimClock:
    """Virtual time: ``sleep`` advances instantly. Yields to the event loop occasionally."""

    def __init__(self, start: float = 0.0, yield_every: int = 50) -> None:
        self._t = start
        self._n = 0
        self._yield_every = yield_every

    def now(self) -> float:
        return self._t

    async def sleep(self, dt: float) -> None:
        self._t += dt
        self._n += 1
        if self._n % self._yield_every == 0:
            await asyncio.sleep(0)

    wall_speed = float("inf")
