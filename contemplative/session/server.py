"""FastAPI server: serves the visuals page and dashboard, relays orchestrator
events over websockets, accepts ratings and start/stop, and broadcasts
parameters over OSC for TouchDesigner.

Routes
    GET  /                 full-screen visuals page
    GET  /dashboard        local dashboard
    GET  /static/...       assets
    WS   /ws/visuals       -> {"type":"visuals", mode, params, morph_s[, immediate]}
                           <- {"type":"rate", better} | {"type":"diag", ...}
    WS   /ws/dashboard     -> state / score / scores / trial / log / stream_check events
                           <- {"type":"start"} | {"type":"stop"} | {"type":"rate", better} | {"type":"check_stream"}
    GET  /api/state
    POST /api/start  /api/stop  /api/rate {"better": bool}  /api/check-stream

The stream check (what arrives, units, rejection reasons - see
ingest/stream_check.py) runs while no session is running: automatically when
the server starts (unless --autostart) and on demand from the dashboard.  It
owns the source while it runs and releases it before a session starts.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..ingest.stream_check import StreamStats
from ..visuals.osc_broadcast import OSCBroadcaster
from ..visuals.params import StimulusParams
from .orchestrator import Orchestrator

STATIC = Path(__file__).resolve().parent.parent / "visuals" / "static"


def _jsonable(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


class SessionServer:
    def __init__(
        self,
        make_orchestrator: Callable[[], Orchestrator],
        osc: OSCBroadcaster | None = None,
        autostart: bool = False,
        make_source: Callable[[], Any] | None = None,
        make_spec: Callable[[], Any] | None = None,
        source_listen: str = "",
        check_seconds: float = 20.0,
    ) -> None:
        self.make_orchestrator = make_orchestrator
        self.osc = osc
        self.autostart = autostart
        self.make_source = make_source
        self.make_spec = make_spec
        self.source_listen = source_listen
        self.check_seconds = check_seconds
        self.check_task: asyncio.Task | None = None
        self.check_source = None
        self.last_check: dict[str, Any] | None = None
        self.orch: Orchestrator | None = None
        self.task: asyncio.Task | None = None
        self.visuals_clients: set[WebSocket] = set()
        self.dash_clients: set[WebSocket] = set()
        self.last_visuals: dict[str, Any] = {"type": "visuals", "mode": "idle", "params": None, "morph_s": 10}
        self.recent_scores: list[tuple[float, float | None]] = []
        self.log_lines: list[dict[str, Any]] = []
        self.app = FastAPI(title="contemplative visuals")
        self._routes()

    # -- orchestration -------------------------------------------------------------
    def _on_event(self, ev: dict[str, Any]) -> None:
        ev = _jsonable(ev)
        if ev["type"] == "visuals":
            self.last_visuals = ev
            if self.osc:
                if ev.get("params"):
                    self.osc.send_params(StimulusParams.from_dict(ev["params"]), ev["morph_s"], ev["mode"])
                else:
                    self.osc.send_mode(ev["mode"], ev["morph_s"])
            self._broadcast(self.visuals_clients, ev)
            return
        if ev["type"] == "score":
            self.recent_scores.append((ev["t"], ev["score"] if ev["valid"] else None))
            if len(self.recent_scores) > 1200:
                self.recent_scores = self.recent_scores[-1200:]
            if self.osc and ev["valid"]:
                self.osc.send_score(ev["score"])
        elif ev["type"] == "log":
            self.log_lines.append(ev)
            self.log_lines = self.log_lines[-300:]
        self._broadcast(self.dash_clients, ev)

    def _broadcast(self, clients: set[WebSocket], ev: dict[str, Any]) -> None:
        data = json.dumps(ev)
        for ws in list(clients):
            asyncio.ensure_future(self._send(ws, data, clients))

    async def _send(self, ws: WebSocket, data: str, clients: set[WebSocket]) -> None:
        try:
            await ws.send_text(data)
        except Exception:
            clients.discard(ws)

    # -- stream check (idle only) ---------------------------------------------------------
    def start_check(self) -> dict[str, Any]:
        if self.orch is not None and self.orch.running:
            return {"ok": False, "reason": "a session is running; the stream check uses the session's own statistics"}
        if self.check_task is not None and not self.check_task.done():
            return {"ok": False, "reason": "a stream check is already running"}
        if self.make_source is None or self.make_spec is None:
            return {"ok": False, "reason": "stream check not configured"}
        self.check_task = asyncio.ensure_future(self._run_check())
        return {"ok": True}

    def stop_check(self) -> None:
        if self.check_task is not None and not self.check_task.done():
            self.check_task.cancel()
        if self.check_source is not None:
            try:
                self.check_source.stop()
            finally:
                self.check_source = None

    async def _run_check(self) -> None:
        import time as _time

        spec = self.make_spec()
        try:
            source = self.make_source()
            self.check_source = source
            stats = StreamStats(spec, listen=self.source_listen)
            source.start()
        except Exception as e:
            self.last_check = {"type": "stream_check", "running": False, "error": f"{e!r}", "summary": None}
            self._broadcast(self.dash_clients, self.last_check)
            self._on_event({"type": "log", "t": 0, "msg": f"stream check could not start: {e!r} (is another program or a previous run holding the port?)"})
            return
        t0 = _time.monotonic()
        last = t0
        try:
            while True:
                now = _time.monotonic()
                source.tick(now)
                pk = source.read()
                if pk:
                    stats.feed(pk)
                if now - last >= 1.0 or now - t0 >= self.check_seconds:
                    last = now
                    self.last_check = {"type": "stream_check", "running": now - t0 < self.check_seconds, "elapsed_s": round(now - t0, 1), "seconds": self.check_seconds, "summary": _jsonable(stats.summary())}
                    self._broadcast(self.dash_clients, self.last_check)
                if now - t0 >= self.check_seconds:
                    break
                await asyncio.sleep(0.1)
            d = stats.diagnose()
            self._on_event({"type": "log", "t": 0, "msg": f"stream check [{d['severity']}]: {d['cause']}" + (f" -> {d['advice']}" if d["advice"] else "")})
        except asyncio.CancelledError:
            pass
        finally:
            try:
                source.stop()
            finally:
                if self.check_source is source:
                    self.check_source = None

    def start_session(self) -> dict[str, Any]:
        if self.orch is not None and self.orch.running:
            return {"ok": False, "reason": "a session is already running"}
        self.stop_check()  # release the source (OSC port) before the session binds it
        self.orch = self.make_orchestrator()
        self.orch.add_listener(self._on_event)
        self.task = asyncio.ensure_future(self._run())
        return {"ok": True, "session_id": self.orch.session_id}

    async def _run(self) -> None:
        assert self.orch is not None
        try:
            await self.orch.run()
        except Exception:
            # the orchestrator has already logged, closed its files and broadcast
            # phase "error"; swallow here so the server keeps serving the dashboard
            import traceback

            traceback.print_exc()

    def stop_session(self) -> dict[str, Any]:
        if self.orch is None or not self.orch.running:
            return {"ok": False, "reason": "no running session"}
        self.orch.request_stop()
        return {"ok": True}

    def rate(self, better: bool) -> dict[str, Any]:
        if self.orch is None:
            return {"ok": False, "reason": "no session"}
        return self.orch.rate(bool(better))

    def state(self) -> dict[str, Any]:
        if self.orch is None:
            return {"type": "state", "phase": "idle", "running": False, "finished": False, "trial": 0, "n_trials": 0, "trials": [], "scorer": {"valid": 0, "rejected": 0}, "fit": None, "fit_known": False, "source": {"listen": self.source_listen}, "session_id": "-", "wearer": "-", "message": "run the stream check, then press Start", "optimizer": {}, "calibrated": False, "spec": {}, "quality": {}, "stream": None, "calibration_progress": None}
        return _jsonable(self.orch.snapshot())

    # -- routes ------------------------------------------------------------------------
    def _routes(self) -> None:
        app = self.app
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

        @app.get("/")
        async def index():
            return FileResponse(STATIC / "index.html")

        @app.get("/dashboard")
        async def dashboard():
            return FileResponse(STATIC / "dashboard.html")

        @app.get("/api/state")
        async def api_state():
            return JSONResponse(self.state())

        @app.post("/api/start")
        async def api_start():
            return JSONResponse(self.start_session())

        @app.post("/api/stop")
        async def api_stop():
            return JSONResponse(self.stop_session())

        @app.post("/api/rate")
        async def api_rate(body: dict):
            return JSONResponse(self.rate(bool(body.get("better", True))))

        @app.post("/api/check-stream")
        async def api_check_stream():
            return JSONResponse(self.start_check())

        @app.get("/api/check-stream")
        async def api_check_stream_last():
            return JSONResponse(self.last_check or {"type": "stream_check", "running": False, "summary": None})

        @app.on_event("startup")
        async def _startup():
            if self.autostart:
                self.start_session()
            elif self.make_source is not None:
                self.start_check()

        @app.websocket("/ws/visuals")
        async def ws_visuals(ws: WebSocket):
            await ws.accept()
            self.visuals_clients.add(ws)
            try:
                await ws.send_text(json.dumps({**self.last_visuals, "immediate": True}))
                while True:
                    msg = json.loads(await ws.receive_text())
                    if msg.get("type") == "rate":
                        res = self.rate(bool(msg.get("better", True)))
                        self._broadcast(self.dash_clients, {"type": "log", "t": self.orch.clock.now() if self.orch else 0, "msg": f"rating from visuals page: {res}"})
                    elif msg.get("type") == "diag":
                        self._broadcast(self.dash_clients, msg)
                        if self.orch is not None:
                            self.orch.recorder.marker(self.orch.clock.now(), "visuals_diag", msg)
            except WebSocketDisconnect:
                pass
            finally:
                self.visuals_clients.discard(ws)

        @app.websocket("/ws/dashboard")
        async def ws_dashboard(ws: WebSocket):
            await ws.accept()
            self.dash_clients.add(ws)
            try:
                await ws.send_text(json.dumps(self.state()))
                await ws.send_text(json.dumps({"type": "scores", "points": self.recent_scores}))
                if self.last_check and not (self.orch is not None and self.orch.running):
                    await ws.send_text(json.dumps(self.last_check))
                for line in self.log_lines[-100:]:
                    await ws.send_text(json.dumps(line))
                while True:
                    msg = json.loads(await ws.receive_text())
                    t = msg.get("type")
                    if t == "start":
                        res = self.start_session()
                    elif t == "stop":
                        res = self.stop_session()
                    elif t == "check_stream":
                        res = self.start_check()
                    elif t == "rate":
                        res = self.rate(bool(msg.get("better", True)))
                    else:
                        res = {"ok": False, "reason": "unknown command"}
                    await ws.send_text(json.dumps({"type": "log", "t": self.orch.clock.now() if self.orch else 0, "msg": f"{t}: {res}"}))
                    await ws.send_text(json.dumps(self.state()))
            except WebSocketDisconnect:
                pass
            finally:
                self.dash_clients.discard(ws)
