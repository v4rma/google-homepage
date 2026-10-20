"""HTTP API and static UI. Run with: uvicorn avatar.api:app --reload"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, load_settings
from .llm import LLMError, LLMProvider, make_provider
from .logging_utils import configure_logging, log_error
from .pipeline import Pipeline
from .storage import ALL_STATUSES, Store

STATIC_DIR = Path(__file__).parent / "static"


class EpisodeIn(BaseModel):
    title: str = ""
    situation: str = Field(min_length=1)
    immediate_thought: str = Field(min_length=1)
    emotion_body: str = Field(min_length=1)
    action_outcome: str = Field(min_length=1)
    comparable_occasion: str = ""
    reserved: bool = False


class ReservedIn(BaseModel):
    reserved: bool


class AnswerIn(BaseModel):
    answer: str = ""


class CorrectionIn(BaseModel):
    text: str = ""


class StatusIn(BaseModel):
    status: str


class NextQuestionIn(BaseModel):
    episode_id: str | None = None


class RatingIn(BaseModel):
    rating: str


def create_app(store: Store | None = None, provider: LLMProvider | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    store = store or Store(settings.db_path)
    provider = provider or make_provider(settings)
    pipeline = Pipeline(store, provider, settings)

    app = FastAPI(title="Avatar", version=__version__, docs_url="/api/docs", redoc_url=None)
    app.state.store, app.state.pipeline = store, pipeline

    def _404(obj, what: str = "record"):
        if obj is None:
            raise HTTPException(404, f"{what} not found")
        return obj

    def _run(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'"))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except LLMError as exc:
            log_error("api.llm_error", exc)
            raise HTTPException(502, f"LLM provider error ({type(exc).__name__}); see server log for the error type")

    # ---------------------------------------------------------------- meta
    @app.get("/api/status")
    def status():
        return {"version": __version__, "provider": getattr(provider, "name", "?"), "model": settings.model if settings.provider != "mock" else None,
                "counts": store.counts(), "pending_question": store.pending_turn()}

    # ---------------------------------------------------------------- episodes
    @app.get("/api/episodes")
    def list_episodes():
        eps = store.list_episodes()
        turns = store.list_turns()
        by_ep: dict[str, int] = {}
        for t in turns:
            by_ep[t["episode_id"]] = by_ep.get(t["episode_id"], 0) + 1
        return [{**e, "turn_count": by_ep.get(e["id"], 0)} for e in eps]

    @app.post("/api/episodes", status_code=201)
    def create_episode(body: EpisodeIn):
        return store.add_episode(**body.model_dump())

    @app.get("/api/episodes/{episode_id}")
    def get_episode(episode_id: str):
        ep = _404(store.get_episode(episode_id), "episode")
        evidence = []
        for a in store.list_assumptions():
            for e in a["supporting"] + a["contradicting"]:
                if e["episode_id"] == episode_id:
                    evidence.append({**e, "proposition": a["proposition"], "status": a["status"]})
        return {**ep, "observations": store.list_observations(episode_id), "turns": store.list_turns(episode_id),
                "interpretations": store.list_interpretations(episode_id=episode_id), "evidence": evidence, "predictions": store.list_predictions(episode_id)}

    @app.patch("/api/episodes/{episode_id}/reserved")
    def set_reserved(episode_id: str, body: ReservedIn):
        _404(store.get_episode(episode_id), "episode")
        return store.set_reserved(episode_id, body.reserved)

    @app.delete("/api/episodes/{episode_id}", status_code=204)
    def delete_episode(episode_id: str):
        _404(store.get_episode(episode_id), "episode")
        store.delete_episode(episode_id)

    @app.post("/api/episodes/{episode_id}/extract")
    def extract(episode_id: str):
        return _run(pipeline.extract, episode_id)

    @app.post("/api/episodes/{episode_id}/hypothesize")
    def hypothesize(episode_id: str):
        return _run(pipeline.hypothesize, episode_id)

    @app.post("/api/episodes/{episode_id}/process")
    def process(episode_id: str):
        return _run(pipeline.process_episode, episode_id)

    # ---------------------------------------------------------------- assumptions
    @app.get("/api/assumptions")
    def list_assumptions():
        return store.list_assumptions()

    @app.get("/api/assumptions/{assumption_id}")
    def get_assumption(assumption_id: str):
        a = _404(store.get_assumption(assumption_id), "assumption")
        return {**a, "interpretations": store.list_interpretations(assumption_id=assumption_id)}

    @app.post("/api/assumptions/{assumption_id}/critique")
    def critique(assumption_id: str):
        return _run(pipeline.critique, assumption_id)

    @app.post("/api/assumptions/{assumption_id}/correct")
    def correct(assumption_id: str, body: CorrectionIn):
        _404(store.get_assumption(assumption_id), "assumption")
        return _run(pipeline.correct, assumption_id, body.text)

    @app.post("/api/assumptions/{assumption_id}/reject")
    def reject(assumption_id: str, body: CorrectionIn):
        _404(store.get_assumption(assumption_id), "assumption")
        return _run(pipeline.reject, assumption_id, body.text)

    @app.patch("/api/assumptions/{assumption_id}/status")
    def set_status(assumption_id: str, body: StatusIn):
        _404(store.get_assumption(assumption_id), "assumption")
        if body.status not in ALL_STATUSES:
            raise HTTPException(400, f"status must be one of {ALL_STATUSES}")
        return store.set_status(assumption_id, body.status, trigger="user_status_change")

    @app.delete("/api/assumptions/{assumption_id}", status_code=204)
    def delete_assumption(assumption_id: str):
        _404(store.get_assumption(assumption_id), "assumption")
        store.delete_assumption(assumption_id)

    @app.delete("/api/evidence/{evidence_id}", status_code=204)
    def delete_evidence(evidence_id: str):
        store.delete_evidence(evidence_id)

    # ---------------------------------------------------------------- interview
    @app.get("/api/interview")
    def list_turns():
        return store.list_turns()

    @app.post("/api/interview/next")
    def next_question(body: NextQuestionIn):
        return _run(pipeline.next_question, body.episode_id)

    @app.post("/api/interview/{turn_id}/answer")
    def answer(turn_id: str, body: AnswerIn):
        return _run(pipeline.answer, turn_id, body.answer)

    @app.delete("/api/interview/{turn_id}", status_code=204)
    def skip_question(turn_id: str):
        t = _404(store.get_turn(turn_id), "question")
        if t.get("answer"):
            raise HTTPException(400, "answered questions are part of the record and cannot be removed")
        store.delete_turn(turn_id)

    # ---------------------------------------------------------------- holdout
    @app.get("/api/holdout")
    def holdout_overview():
        reserved = [e for e in store.list_episodes() if e["reserved"]]
        preds = store.list_predictions()
        return {"reserved": reserved, "predictions": preds}

    @app.post("/api/holdout/{episode_id}/run")
    def holdout_run(episode_id: str):
        return _run(pipeline.holdout, episode_id)

    @app.patch("/api/predictions/{prediction_id}/rating")
    def rate(prediction_id: str, body: RatingIn):
        if body.rating not in ("accurate", "partly", "wrong", ""):
            raise HTTPException(400, "rating must be accurate | partly | wrong")
        return _404(store.rate_prediction(prediction_id, body.rating), "prediction")

    # ---------------------------------------------------------------- demo
    @app.post("/api/demo/seed")
    def demo_seed():
        from .demo_data import seed

        return {"created": [e["id"] for e in seed(store)]}

    # ---------------------------------------------------------------- UI
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(STATIC_DIR / "index.html")

    return app


_app: FastAPI | None = None


def __getattr__(name: str):
    """Lazily build the default app so that importing this module (e.g. in tests) has no side effects."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
