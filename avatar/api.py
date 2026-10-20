"""HTTP API and static UI. Run with: uvicorn avatar.api:app --reload"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, load_settings
from .jobs import JobQueue
from .llm import LLMProvider, make_provider
from .logging_utils import configure_logging
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
    jobs = JobQueue(provider)

    app = FastAPI(title="Avatar", version=__version__, docs_url="/api/docs", redoc_url=None)
    app.state.store, app.state.pipeline, app.state.jobs = store, pipeline, jobs

    def _404(obj, what: str = "record"):
        if obj is None:
            raise HTTPException(404, f"{what} not found")
        return obj

    def _job(kind: str, label: str, fn, *args, episode_id: str | None = None, assumption_id: str | None = None, **kwargs):
        """Queue a model-backed call and return the job record (HTTP 202). The UI polls /api/jobs/{id}."""
        job = jobs.submit(kind, label, lambda: fn(*args, **kwargs), episode_id=episode_id, assumption_id=assumption_id)
        return JSONResponse(job, status_code=202)

    def _ep_label(episode_id: str) -> str:
        ep = store.get_episode(episode_id)
        return (ep.get("title") or episode_id) if ep else episode_id

    # ---------------------------------------------------------------- jobs
    @app.get("/api/jobs")
    def list_jobs(active: bool = False):
        return jobs.list(active_only=active)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, wait: float = 0.0):
        return _404(jobs.get(job_id, wait=min(max(wait, 0.0), 25.0)), "job")

    # ---------------------------------------------------------------- meta
    @app.get("/api/status")
    def status():
        describe = getattr(provider, "describe", None)
        return {"version": __version__, "provider": getattr(provider, "name", "?"), "model": settings.model if settings.provider == "openai" else None,
                "provider_detail": describe() if callable(describe) else {}, "jobs_active": jobs.active_count(),
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

    @app.post("/api/episodes/{episode_id}/extract", status_code=202)
    def extract(episode_id: str):
        _404(store.get_episode(episode_id), "episode")
        return _job("extract", f"Extract: {_ep_label(episode_id)}", pipeline.extract, episode_id, episode_id=episode_id)

    @app.post("/api/episodes/{episode_id}/hypothesize", status_code=202)
    def hypothesize(episode_id: str):
        ep = _404(store.get_episode(episode_id), "episode")
        if ep["reserved"]:
            raise HTTPException(400, "reserved (holdout) episodes are not analysed; un-reserve it first")
        return _job("hypothesize", f"Hypothesize: {_ep_label(episode_id)}", pipeline.hypothesize, episode_id, episode_id=episode_id)

    @app.post("/api/episodes/{episode_id}/process", status_code=202)
    def process(episode_id: str):
        ep = _404(store.get_episode(episode_id), "episode")
        if ep["reserved"]:
            raise HTTPException(400, "reserved (holdout) episodes are not analysed; un-reserve it first")
        return _job("process", f"Analyse: {_ep_label(episode_id)}", pipeline.process_episode, episode_id, episode_id=episode_id)

    # ---------------------------------------------------------------- assumptions
    @app.get("/api/assumptions")
    def list_assumptions():
        return store.list_assumptions()

    @app.get("/api/assumptions/{assumption_id}")
    def get_assumption(assumption_id: str):
        a = _404(store.get_assumption(assumption_id), "assumption")
        return {**a, "interpretations": store.list_interpretations(assumption_id=assumption_id)}

    @app.post("/api/assumptions/{assumption_id}/critique", status_code=202)
    def critique(assumption_id: str):
        _404(store.get_assumption(assumption_id), "assumption")
        return _job("critique", "Critique a candidate", pipeline.critique, assumption_id, assumption_id=assumption_id)

    @app.post("/api/assumptions/{assumption_id}/correct", status_code=202)
    def correct(assumption_id: str, body: CorrectionIn):
        _404(store.get_assumption(assumption_id), "assumption")
        if not body.text.strip():
            raise HTTPException(400, "correction text is required")
        # The correction is recorded immediately; only the re-wording waits for the model.
        store.add_correction(assumption_id, body.text.strip())
        return _job("correct", "Apply your correction", pipeline.correct, assumption_id, body.text, record_correction=False, assumption_id=assumption_id)

    @app.post("/api/assumptions/{assumption_id}/reject", status_code=202)
    def reject(assumption_id: str, body: CorrectionIn):
        _404(store.get_assumption(assumption_id), "assumption")
        # Rejection takes effect immediately; the model is only asked for an alternative afterwards.
        if body.text.strip():
            store.add_correction(assumption_id, body.text.strip())
        store.set_status(assumption_id, "rejected", trigger="user_rejection", reason="The user rejected this interpretation.")
        return _job("reject", "Offer an alternative after your rejection", pipeline.reject, assumption_id, body.text, mark_rejected=False, assumption_id=assumption_id)

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

    @app.post("/api/interview/next", status_code=202)
    def next_question(body: NextQuestionIn):
        if body.episode_id:
            _404(store.get_episode(body.episode_id), "episode")
        return _job("interview_question", "Choose the next question", pipeline.next_question, body.episode_id, episode_id=body.episode_id)

    @app.post("/api/interview/{turn_id}/answer", status_code=202)
    def answer(turn_id: str, body: AnswerIn):
        t = _404(store.get_turn(turn_id), "question")
        if t.get("answer"):
            raise HTTPException(400, "this question has already been answered")
        return _job("interview_update", "Revise candidates from your answer", pipeline.answer, turn_id, body.answer, episode_id=t["episode_id"])

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

    @app.post("/api/holdout/{episode_id}/run", status_code=202)
    def holdout_run(episode_id: str):
        ep = _404(store.get_episode(episode_id), "episode")
        if not ep["reserved"]:
            raise HTTPException(400, "only reserved (holdout) episodes can be used for prediction tests")
        if not pipeline.active_assumptions():
            raise HTTPException(400, "no active candidate assumptions to test")
        return _job("holdout", f"Prediction test: {_ep_label(episode_id)}", pipeline.holdout, episode_id, episode_id=episode_id)

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
