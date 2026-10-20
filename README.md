# Avatar — prototype

Avatar is a small application around an existing LLM that develops, and tests, a *provisional* model of a
person's underlying assumptions from concrete episodes they write. The design brief it follows is summarised in
[docs/design-brief.md](docs/design-brief.md); the central problem is telling an assumption that explains the
person's experience apart from a compelling story the model has invented about them.

This is the first prototype: prompting plus a small memory store, no training.

## What it does

1. **Episode entry.** The person writes a short account in five parts: situation (with the words actually used),
   immediate thought, emotion and bodily feeling, action and outcome, and a comparable occasion that went
   differently. The original account is stored immutably (a database trigger refuses edits) and separately from
   anything added later during the interview.
2. **Four pipeline stages on one LLM, each with its own prompt** (`avatar/prompts.py`):
   - *Extract* — events, exact quotations, expectations, emotions, actions, and what is missing. Every quotation
     is checked against the person's text; paraphrases are flagged.
   - *Hypothesize* — up to three candidate explanations using the current episode plus a few retrieved past
     episodes, including situational explanations. Candidates that duplicate an existing assumption are attached
     to it instead.
   - *Critique* — contradictions, unsupported leaps, alternative explanations, and questions that would resolve
     the uncertainty.
   - *Interview and update* — one question at a time, grounded in a specific episode and chosen to discriminate
     between competing candidates; "I don't know" is accepted; the answer drives revision. The same stage handles
     the person's own corrections and rejections.
3. **Assumption records** as specific propositions ("If X then Y", "To stay safe/accepted/free I must Z", or an
   explicitly situational statement) with every field of the brief's template: situations, supporting episode IDs
   with exact quotations, contradicting episodes, alternative explanations, prediction for a new situation, what
   would weaken it, the person's corrections, and a status (`proposed` / `under examination` / `revised` /
   `rejected`). Every change is recorded as a revision with the previous version kept.
4. **Observations vs. interpretations.** Observations (extracted from the person's words) and interpretations
   (anything the model produced) live in separate tables. Evidence links carry a `verified` flag that is only
   true when the quotation appears verbatim in the person's account or answers; unverified quotations are shown
   but never counted. Model text is never used as evidence, and a rejection by the person sets the status to
   `rejected` and asks for an alternative; a guard strips any model text that describes the person's answer as
   "resistance", "defensiveness", "denial", and so on.
5. **Retrieval** of a few relevant past episodes per stage with BM25 keyword retrieval (no external service;
   reserved episodes are excluded).
6. **Assumptions panel** in the UI to inspect each candidate with full provenance, run a critique, correct it in
   your own words, reject it, change its status, remove an evidence link, or delete it.
7. **Prediction test (holdout).** Episodes marked *reserved* are never shown to the hypothesis stages. In the
   test, the model gets the situation only; each candidate assumption predicts the thought, feeling and action;
   a generic no-assumption baseline is predicted too; and the comparison with what was recorded is shown side by
   side with a match score and an optional rating by the person.
8. **Privacy.** No episode content in diagnostic logs; provider key only from the environment; see
   *Data retention* below.
9. **Nothing blocks on the model.** Every model-backed action is a background job: the API answers `202` with a
   job record, the UI shows a "waiting for the Avatar agent" strip and keeps working, and the view refreshes when
   the job finishes. Episodes, corrections and rejections are stored immediately; only the model's follow-up waits.

## Stack and why

- **Python 3.11+ / FastAPI** — small surface, typed request models, automatic API docs at `/api/docs`.
- **SQLite via the standard library** — a local single-file database is exactly what the brief asks for; the
  immutability guarantee is a trigger in the schema rather than application convention.
- **No ORM, no frontend build.** A single-page UI in plain HTML/JS keeps the prototype inspectable and easy to
  change; there is nothing to compile.
- **One provider interface, four implementations.** `AgentBridgeProvider` (the default) hands each request to the
  Cursor project agent through a file inbox/outbox, so no API key is needed; `OpenAICompatProvider` talks to any
  OpenAI-compatible chat-completions endpoint over `httpx` and is used only when `OPENAI_API_KEY` is set;
  `MockProvider` is a deterministic keyword rule set for tests and offline plumbing checks (it must be chosen
  explicitly and is not a model); `ScriptedProvider` returns canned responses in tests. All pipeline logic
  (validation of quotations, revision bookkeeping, guards) sits outside the providers, so swapping the model
  changes nothing else.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate     # or: uv venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                             # or: uv pip install -e ".[dev]"
cp .env.example .env                                # optional; variables can also be exported directly
```

Configuration is by environment variables only (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `AVATAR_LLM_PROVIDER` | `openai` if `OPENAI_API_KEY` is set, else `agent` | `agent` (Cursor project agent via file bridge), `openai` (any OpenAI-compatible endpoint) or `mock` (keyword rules, tests only) |
| `AVATAR_AGENT_BRIDGE_DIR` | `data/agent-bridge` | directory holding `analysis-inbox/` and `analysis-outbox/` |
| `AVATAR_AGENT_TIMEOUT` | `120` | seconds a stage waits for the agent before the job is marked "not answered" |
| `OPENAI_API_KEY` | — | provider key (optional); never written to disk or logs |
| `AVATAR_LLM_MODEL` | `gpt-4o-mini` | model name for the `openai` provider |
| `AVATAR_LLM_BASE_URL` | `https://api.openai.com/v1` | endpoint; point at a local server to keep data on your machine |
| `AVATAR_DB_PATH` | `data/avatar.db` | SQLite file |
| `AVATAR_LOG_LEVEL` | `INFO` | diagnostic log level |

## Run

```bash
# optional: load the clearly synthetic demo dataset (13 episodes, 3 reserved for the prediction test)
python scripts/seed_demo.py

uvicorn avatar.api:app --reload
# open http://127.0.0.1:8000
```

Suggested walk-through in the UI: *Episodes* → "Load demo data" → open an episode → "Analyse" (Extract →
Hypothesize → Critique) for a handful of episodes → *Assumptions* to inspect provenance → *Interview* → "Ask the
next question" and answer → correct or reject a candidate → *Prediction test* → "Run prediction test" on a
reserved episode.

## The agent bridge (default provider)

With no key set, Avatar does not call any model API. Each stage request is written to
`<AVATAR_AGENT_BRIDGE_DIR>/analysis-inbox/<id>.json` and the pipeline waits (in a background job) for
`.../analysis-outbox/<id>.json`. The Cursor project agent that owns this prototype reads pending requests, acts
as the model for that stage, and writes the answer. `scripts/agent_inbox.py` does the bookkeeping:

```bash
export AVATAR_AGENT_BRIDGE_DIR=/path/to/bridge
python scripts/agent_inbox.py list                    # pending requests
python scripts/agent_inbox.py wait --timeout 500      # block until one is pending, then print it
python scripts/agent_inbox.py show <id>               # print a request (system prompt + user prompt)
python scripts/agent_inbox.py answer <id> --file a.json
```

Request file: `{"id", "stage", "created_at", "system", "user", "payload"}` — `system` is the common rules prompt,
`user` the stage instructions plus the input records as JSON, `payload` the same records as structured data. The
answer is the bare JSON object the stage schema asks for (or `{"id", "response": {...}}`). The id is a hash of the
stage and the user prompt, so an identical request reuses an existing answer and an answer that arrives after a
timeout is picked up by "Retry" in the UI. Pending work for a processor is simply: inbox files without an outbox
file of the same name.

If nobody is answering, the UI stays usable: episodes are saved immediately, the job strip shows
"Waiting for the Avatar agent" with the request id, and after `AVATAR_AGENT_TIMEOUT` seconds the job is marked as
not answered with a Retry button. The request remains in the inbox.

## Milestone check

`scripts/run_milestone.py` runs the whole pipeline on the demo dataset in a throwaway database (with the mock
provider unless `AVATAR_LLM_PROVIDER` is set, so it never waits on the inbox unattended) and prints a
Markdown report: the candidates after each episode, the interview questions and the synthetic persona's answers,
the resulting assumption record in the brief's template, the revision history after a user correction, and the
prediction-test tables with a generic baseline for comparison. The committed output is in
[docs/milestone-check.md](docs/milestone-check.md).

```bash
python scripts/run_milestone.py                          # print
python scripts/run_milestone.py --write docs/milestone-check.md
```

## Tests

```bash
pytest
```

The tests never call a real model. `tests/test_storage.py` covers the storage layer (immutability, quotation
verification, separation of observations and interpretations, revisions, cascades). `tests/test_pipeline.py`
drives every stage with a scripted provider (checking what the model receives and how its output is validated),
plus an end-to-end run with the rule-based mock. `tests/test_api.py` exercises the HTTP API and UI serving.

## About the mock provider

`MockProvider` is **not** a language model. It is a few hundred lines of keyword rules that produce output in the
same JSON shape as the real prompts request, covering three themes (autonomy, evaluation, exhaustion) that the
synthetic dataset was written around. It exists so the pipeline, storage, UI and milestone script can be run and
tested without a key, and it will produce shallow results on anything outside the demo data. Anything beyond
plumbing — the quality of hypotheses, questions and critiques — can only be judged with the real provider.

## Privacy and data retention

- **Local storage.** Everything is in one SQLite file (`AVATAR_DB_PATH`). Deleting it deletes the record. Episode
  deletion in the UI cascades to observations, interview turns, evidence links, predictions and interpretations.
- **Logs.** Diagnostic logging goes through `avatar/logging_utils.py`, which only accepts identifiers, numbers and
  short tokens; anything that looks like free text is redacted. Exceptions are logged by type only. Pipeline run
  bookkeeping in the database stores stage, ids, timing and error type, not content. Uvicorn's access log shows
  URL paths (which contain record ids) but not request bodies.
- **What leaves the machine.** With the `openai` provider, each stage sends the relevant episode text, extracted
  observations, interview answers and current assumption records to the endpoint in `AVATAR_LLM_BASE_URL`. With
  the `mock` provider nothing leaves the machine.
- **Provider retention settings.** Before sending real material, check the provider's data-retention and
  training policies for *API* traffic (they differ from consumer products), enable zero-data-retention or an
  equivalent where offered, and prefer an endpoint with a data-processing agreement. Pointing
  `AVATAR_LLM_BASE_URL` at a local OpenAI-compatible server keeps all data local.
- **Not covered yet.** No encryption at rest, no authentication (bind the server to localhost), no export or
  retention schedule, no consent flow. These are prerequisites before any use beyond the author's own machine,
  and before any use with material from another person.

## Layout

```
avatar/
  api.py             FastAPI app and routes
  config.py          settings from environment variables
  db.py              SQLite schema (tables, immutability trigger)
  storage.py         repository layer
  jobs.py            background job queue (model calls never block the UI)
  retrieval.py       BM25 retrieval of related episodes
  prompts.py         the stage prompts
  pipeline/          extract, hypothesize, critique, interview (+ corrections), holdout
  llm/               provider interface; agent bridge (default), OpenAI-compatible, mock, scripted (tests)
  demo_data.py       synthetic dataset and the persona's scripted answers
  static/            single-page UI
scripts/             seed_demo.py, run_milestone.py, agent_inbox.py
tests/               storage, pipeline, API
docs/                design brief, milestone report
```

## Known gaps

- Analysis quality depends on who answers the inbox. The OpenAI-compatible provider is implemented but has not
  been exercised against a live endpoint in this environment; the mock is plumbing-only.
- Jobs live in memory: a server restart forgets queued jobs (inbox requests and answers persist on disk).
- Retrieval is keyword-based (BM25); embeddings would help with paraphrased situations.
- Pipeline calls are synchronous; a slow provider blocks the request.
- Single user, no authentication; intended to run on localhost.
- The "does it feel revealing" measure from the brief is not collected (only the prediction-match rating is).
