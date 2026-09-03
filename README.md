# Docent — Backend

Docent is an AI triage service for open-source GitHub repositories. It watches a
repo's issues, keeps a searchable corpus of **already-resolved** issues, and — on
demand — runs an open issue through a retrieval-augmented agent that researches
the problem and drafts a grounded reply citing how similar problems were solved
before.

The reference target is [`pydantic/pydantic`](https://github.com/pydantic/pydantic),
configurable via `REPO_OWNER` / `REPO_NAME`.

**Contents** — [How it works](#how-it-works) · [The agent](#the-agent) ·
[Retrieval](#retrieval) · [Stack](#stack) · [Project layout](#project-layout) ·
[Data model](#data-model) · [Background jobs](#background-jobs) ·
[API](#api) · [Getting started](#getting-started) ·
[Configuration](#configuration) · [Migrations](#database--migrations) ·
[Design notes](#design-notes) · [Status](#status)

---

## How it works

Docent keeps **two stores** that play opposite roles:

- **Corpus** (`closed-issues`) — closed issues that were *actually solved*, each
  with the original question, the fix, an embedding and a full-text vector. This
  is the knowledge the agent retrieves *from*.
- **Queue** (`open-issues`) — the currently-open issues shown on the dashboard,
  the ones you can run the agent *on*.

```
                          GitHub
             closed issues │ open issues
        ┌──────────────────┼──────────────────┐
        ▼                                     ▼
  closed-issues poller (5m)            open-issues poller (30m)
  backfill (one-shot)                  diff GitHub vs queue
        │                              ├─ new  → insert into queue
        ▼                              └─ gone → promote to corpus
   ┌─────────────┐   retrieve      ┌─────────────┐
   │   CORPUS    │◄────────────────│    QUEUE    │
   │ closed-     │   promote on    │  open-      │
   │ issues      │   resolution    │  issues     │
   └─────┬───────┘                 └─────┬───────┘
         │ hybrid search + rerank        │ POST /agent/run
         ▼                               ▼
   ┌──────────────────────────────────────────────┐
   │  Agent:  intake → research ⇄ tools           │
   │          → finalize → draft                  │
   └──────────────────────────────────────────────┘
```

### Lifecycle of an issue

1. Opened on GitHub → the open-issues poller inserts it into the **queue**.
2. It shows on the dashboard; a user clicks **Run Agent** (`POST /agent/run`).
3. **Intake** summarizes the body and embeds `title + question`; **research**
   runs a tool loop over the corpus and live GitHub via MCP; **finalize** turns
   the transcript into a structured brief; **draft** writes a grounded reply.
4. Later the issue is closed on GitHub → the poller notices it left the open set,
   fetches the real fix, and if it's a genuine resolution **promotes it into the
   corpus** (and drops it from the queue). The corpus is now a little smarter for
   the next question.

---

## The agent

The agent is a compiled **LangGraph** state machine (`agent/graph.py`, exported as
`graph` and registered in `langgraph.json`). It is async end to end and invoked
via `agent.run.run_agent(github_number)`.

```
        START
          │
          ▼
      ┌────────┐
      │ intake │  summarize + embed the issue (cached on the row)
      └───┬────┘
          ▼
      ┌──────────┐   tool calls?   ┌───────┐
      │ research │────── yes ─────►│ tools │
      │  (LLM)   │◄────────────────└───────┘
      └────┬─────┘   tool results
           │ no tool calls / iteration cap reached
           ▼
   ┌──────────────────┐
   │ finalize_research│  structured ResearchBrief (Pydantic)
   └────────┬─────────┘
            ▼
        ┌───────┐
        │ draft │  markdown reply grounded in the brief
        └───┬───┘
            ▼
           END
```

### Nodes

| Node                | File                       | What it does |
| ------------------- | -------------------------- | ------------ |
| `intake`            | `agent/intake/node.py`     | Loads the open issue, summarizes the body with Claude Haiku 4.5, embeds `title + question` with Voyage, and **writes both back to the row** so a re-run is free. Skips all of it if the row is already populated. |
| `research`          | `agent/research/node.py`   | Binds the available tools and lets the model drive a ReAct-style loop. Seeds the conversation on the first pass; every later pass sees the full message history. |
| `tools`             | `agent/research/tools.py`  | Executes the requested calls, converts each result to a `ToolMessage`, and appends `{name, args}` to the run's tool log. |
| `finalize_research` | `agent/research/node.py`   | Re-invokes the model with `with_structured_output(ResearchBrief)` to force the transcript into a typed brief. |
| `draft`             | `agent/draft/node.py`      | Renders the brief into a compact block and asks the model for a maintainer-voiced reply in GitHub-flavored markdown. |

### Conditional edge

`agent/research/edges.py` inspects the last message: if it carries `tool_calls`,
the graph routes to `tools` and loops back into `research`; otherwise it routes to
`finalize_research`. The loop is bounded — once `research_iterations` reaches
`RESEARCH_MAX_ITERATIONS` (default **6**), `research` binds **no tools at all**, so
the model physically cannot request another one and the next hop is always
`finalize_research`.

### Tools available to research

| Tool | Source | Limit |
| ---- | ------ | ----- |
| `search_corpus` | Local hybrid retrieval over `closed-issues` | **Once per run.** Gated by the `rag_used` flag; a second attempt returns "already searched, reuse the earlier results" instead of burning an embedding + rerank call. |
| GitHub read tools | GitHub MCP server (`langchain-mcp-adapters`) | Read-only allow-list, loaded once per process and cached. |

The MCP allow-list (`agent/research/github_mcp.py`) is exactly:
`issue_read`, `list_issues`, `search_issues`, `pull_request_read`,
`list_pull_requests`, `search_pull_requests`, `get_file_contents`, `get_commit`,
`list_commits`, `search_code`. The client also sends `X-MCP-Readonly: true`, so
the allow-list is belt *and* braces — nothing in the toolset can write to GitHub.

### Agent state

`agent/state.py` — a `TypedDict` threaded through every node:

| Field | Type | Notes |
| ----- | ---- | ----- |
| `github_number`, `title`, `original_question` | `int`, `str`, `str` | Seeded from the queue row |
| `body_summary` | `str \| None` | Filled by intake |
| `embedding` | `list[float] \| None` | Filled by intake |
| `retrieved` | `list[dict]` | Reranked corpus hits |
| `tool_calls` | `Annotated[list, add]` | Append-only audit log of every call |
| `messages` | `Annotated[list, add_messages]` | The research transcript |
| `research_brief` | `dict \| None` | `ResearchBrief.model_dump()` |
| `research_iterations` | `int` | Loop counter against the cap |
| `rag_used` | `bool` | One-shot latch for `search_corpus` |
| `draft` | `str \| None` | Final reply |
| `needs_escalation` | `bool` | Mirrored from the brief |

### The research brief

`schemas/research.py` — the structured contract between research and draft:

```python
class ResearchBrief(BaseModel):
    problem_restatement: str
    root_cause_hypothesis: str
    corpus_references: list[CorpusReference]   # github_number, url, why_relevant
    external_finding: list[str]
    suggested_response_direction: str
    open_questions: list[str]
    confidence: float                          # 0.0 – 1.0
    needs_escalation: bool
```

Forcing the model through this schema is what makes the draft step cheap and
grounded: it never sees the raw tool transcript, only fields it is told to cite.

### Prompts

Every prompt lives in `ai/prompts/` and is surfaced through
`Settings.prompts`, so prompt changes are configuration, not code edits:

| Prompt | Used by | Shape |
| ------ | ------- | ----- |
| `SUMMARY_SYSTEM_PROMPT` / `SUMMARY_USER_TEMPLATE` | intake | Faithful 3–6 sentence problem statement; explicitly forbidden from proposing fixes |
| `RESEARCH_SYSTEM_PROMPT` / `RESEARCH_USER_TEMPLATE` | research | Explains both tool families and the one-call corpus budget |
| `RESEARCH_FINALIZE_INSTRUCTION` | finalize | "Cite only what you established; lower confidence and escalate if unsure" |
| `DRAFT_SYSTEM_PROMPT` / `DRAFT_USER_TEMPLATE` | draft | Maintainer voice — lead with the technical substance, no thanks/sign-offs, link related issues |

All three stages wrap untrusted input (issue bodies, tool output) in tags and
instruct the model to treat it strictly as data, never as instructions.

---

## Retrieval

`search_corpus` is a four-stage pipeline over the closed-issue corpus:

```
query ──┬─► vector search   (pgvector cosine distance, k=20)  ──┐
        │                                                       ├─► RRF merge ─► rerank ─► top 6
        └─► keyword search  (Postgres FTS, ts_rank_cd, k=20)  ──┘   (top 30)   (Voyage)
```

| Stage | Implementation | Detail |
| ----- | -------------- | ------ |
| Embed | `rag/embeddings/embed_issue.py` | Voyage `voyage-4`, 1024 dims, `input_type="query"` |
| Vector | `rag/retrieval/vector_retrieval.py` | `ClosedIssue.embeddings.cosine_distance(...)`, ordered ascending |
| Keyword | `rag/retrieval/keyword_retriever.py` | `websearch_to_tsquery('english', …)` against the generated `search_vector` column, ranked with `ts_rank_cd(..., 32)` |
| Fusion | `rag/retrieval/merge.py` | Reciprocal Rank Fusion, `score += 1 / (60 + rank)` per list — no score normalization needed between two incomparable ranking systems |
| Rerank | `rag/retrieval/reranker.py` | Voyage `rerank-2.5` over the top 30 candidates, returns 6 |

Hits are formatted for the model as `#number title (url)` plus 300-char excerpts
of the problem and the fix, and the full reranked payload is kept in state under
`retrieved`.

---

## Stack

| Concern             | Tool                                        |
| ------------------- | ------------------------------------------- |
| HTTP API            | **FastAPI** + Uvicorn                       |
| Background jobs     | **Celery** + **Redis** (broker & scheduler) |
| Data + vectors      | **PostgreSQL 16** + **pgvector**            |
| Keyword search      | Postgres full-text search (generated `tsvector`) |
| ORM & migrations    | **SQLAlchemy 2.0** + **Alembic**            |
| Agent orchestration | **LangGraph** + **LangChain**               |
| Generation          | **Anthropic** — Sonnet for research/draft, Haiku for summaries |
| Embeddings & rerank | **Voyage AI** (`voyage-4`, `rerank-2.5`)    |
| External tools      | **GitHub MCP server** via `langchain-mcp-adapters` |
| Packaging           | **uv** + Docker Compose                     |

---

## Project layout

| Path                        | Purpose                                                         |
| --------------------------- | --------------------------------------------------------------- |
| `api/server.py`             | FastAPI app entrypoint (`api.server:app`), mounts all routers    |
| `api/routes/`               | `fetch_issue`, `corpus_backfill`, `run_agent`                    |
| `api/crud/issues.py`        | Corpus insert helper                                             |
| `agent/graph.py`            | Node/edge wiring, compiled `graph`                               |
| `agent/state.py`            | `AgentState` TypedDict                                           |
| `agent/run.py`              | Loads the queue row and invokes the graph                        |
| `agent/intake/`             | Summarize + embed + cache                                        |
| `agent/research/`           | Research node, tool node, conditional edge, MCP client, LLM factory |
| `agent/draft/`              | Brief rendering + reply generation                               |
| `agent/gate/`, `agent/escalate/` | Reserved for the review/escalation stages *(empty)*         |
| `ai/summary_llm.py`         | Cached Anthropic client + intake summarizer                      |
| `ai/prompts/`               | All system prompts and user templates                            |
| `rag/embeddings/`           | Cached Voyage client, single + batched embedding helpers         |
| `rag/retrieval/`            | Vector, keyword, RRF merge, rerank                               |
| `rag/ingestion/issues.py`   | Promotion of resolved issues into the corpus                     |
| `ingestion/issues.py`       | GitHub GraphQL fetchers (open, closed, by-number) + fix parsing   |
| `workers/celery_app.py`     | Celery app + beat schedules                                      |
| `workers/tasks/`            | Pollers (open, closed) and the paginated corpus backfill          |
| `database/models/`          | `ClosedIssue`, `OpenIssue`                                       |
| `database/db.py`            | Engine, `SessionLocal`, `Base`, `get_db`                         |
| `schemas/`                  | Pydantic DTOs — issue transfer objects and `ResearchBrief`        |
| `core/config.py`            | Typed settings + prompt registry                                 |
| `migrations/`               | Alembic environment and versions                                 |
| `db/init/`                  | Compose-time SQL (`CREATE EXTENSION vector`)                     |
| `mcp/`, `evals/`, `tests/`, `observability/`, `security/` | Placeholders for planned work      |

---

## Data model

**`closed-issues`** — the retrieval corpus (one row per resolved issue):

| Column              | Notes                                              |
| ------------------- | -------------------------------------------------- |
| `id`                | UUID primary key                                   |
| `github_number`     | unique, indexed                                    |
| `title`             |                                                    |
| `original_question` | the issue body                                     |
| `fix_summary`       | how it was resolved (closing PR body / last comment) |
| `url`, `closed_at`, `created_at` |                                       |
| `tags`              | GitHub labels (`text[]`)                           |
| `embeddings`        | `vector(1024)` over `title + original_question`    |
| `search_vector`     | generated, stored `tsvector` over title + body — keeps FTS in sync automatically |

**`open-issues`** — the work queue (one row per open issue):

| Column              | Notes                                              |
| ------------------- | -------------------------------------------------- |
| `id`                | UUID primary key                                   |
| `github_number`     | unique, indexed                                    |
| `title`, `original_question`, `url`, `tags` | filled at sync time        |
| `body_summary`      | nullable — filled lazily at intake                 |
| `embeddings`        | `vector(1024)`, nullable — filled lazily at intake  |

Embeddings and summaries are left empty by the poller and computed only when the
agent actually runs on an issue, so the corpus is embedded exactly once and open
issues are embedded only if someone works on them. Intake writes the results back,
so re-running the agent on the same issue costs nothing extra.

### How a "fix" is detected

`ingestion/issues.py` pulls, for each closed issue, the closing event's linked PR
body and the last comment. The PR body wins; the comment is only accepted if it
passes `_looks_like_fix()` — at least 40 characters and free of noise markers like
*"duplicate of"*, *"please provide a reproducible example"*, *"closing as stale"*
or *"automatically closed"*. Issues with no usable fix are skipped entirely, which
is what keeps the corpus a store of **solutions** rather than of closed tickets.

---

## Background jobs

Configured in `workers/celery_app.py`:

| Task                   | Schedule     | Does                                                        |
| ---------------------- | ------------ | ----------------------------------------------------------- |
| `poll_github_issues`   | every 5 min  | Pull newly *closed* issues with a fix into the corpus. Stops at the first already-known number, so a steady-state poll is one GraphQL page. |
| `poll_open_issues`     | every 30 min | Diff GitHub's open set against the queue: insert what's new, and hand what disappeared to `ingest_closed_issues` — which refetches it by number, promotes genuine resolutions into the corpus, and deletes it from the queue. |
| `backfill_corpus_page` | on demand    | One-shot historical backfill. Self-reschedules page by page with a 2 s delay, backing off to 60 s when the GraphQL rate-limit budget drops below 100. |

Run the worker and scheduler together (`worker --beat`) or as separate processes
(`worker`, `beat`) — the compose stack runs them separately.

---

## API

| Method & path                            | Purpose                                     |
| ---------------------------------------- | ------------------------------------------- |
| `POST /agent/run`                        | Run the full agent on one queued open issue |
| `POST /issues/fetch`                     | Fetch the latest open issue from GitHub     |
| `POST /admin/corpus/backfill?limit=1500` | Kick off the corpus backfill (202, async)   |

Interactive docs are at `/docs` once the API is up.

### `POST /agent/run`

```bash
curl -X POST localhost:8000/agent/run \
  -H 'content-type: application/json' \
  -d '{"github_number": 12345}'
```

```jsonc
{
  "draft": "…markdown reply…",
  "research_brief": {
    "problem_restatement": "…",
    "root_cause_hypothesis": "…",
    "corpus_references": [
      { "github_number": 9876, "url": "https://github.com/…", "why_relevant": "…" }
    ],
    "external_finding": ["…"],
    "suggested_response_direction": "…",
    "open_questions": ["…"],
    "confidence": 0.72,
    "needs_escalation": false
  },
  "tool_calls": [{ "name": "search_corpus", "args": { "query": "…" } }]
}
```

The issue must already be in the `open-issues` queue — an unknown number returns
`404`. The run is synchronous and takes as long as the tool loop needs.

---

## Getting started

### Docker (self-contained)

This repo ships its own `docker-compose.yml` — Postgres (pgvector), Redis, the
API, and the Celery worker + beat as separate services. From a fresh clone:

```bash
cp .env.example .env          # then fill in your API keys
docker compose up --build     # api :8000 · postgres :5433 · redis :6379
docker compose run --rm backend uv run alembic upgrade head   # first-time schema
```

The `postgres` service auto-creates the pgvector extension from `db/init/`, and
`DATABASE_URL` / `REDIS_URL` are wired to the compose services (overriding
whatever is in `.env`, which is used only when running on the host).

Then seed the corpus and the queue:

```bash
curl -X POST 'localhost:8000/admin/corpus/backfill?limit=1500'   # historical corpus
docker compose exec celery_worker uv run celery -A workers.celery_app.app \
  call poll_open_issues                                          # fill the queue now
```

### Local (uv)

Dependencies are managed with [uv](https://github.com/astral-sh/uv). You still
need a reachable Postgres (with pgvector) and Redis — the compose above is the
easiest way to get them.

```bash
uv sync
cp .env.example .env          # fill in secrets + point DATABASE_URL/REDIS_URL at your services
uv run alembic upgrade head
uv run uvicorn api.server:app --reload                                  # API on :8000
uv run celery -A workers.celery_app.app worker --beat --loglevel=info   # workers
```

### LangGraph Studio

`langgraph.json` exposes the compiled graph as `docent`, so the agent can be
stepped through node by node without the API or Celery:

```bash
uv sync --group dev
uv run langgraph dev
```

---

## Configuration

All settings are read from the environment (see `.env.example`) and validated by
`core/config.py` (`pydantic-settings`), which is cached with `lru_cache` — the
process reads the environment once.

| Variable                   | Required | Default | Purpose                              |
| -------------------------- | -------- | ------- | ------------------------------------ |
| `GITHUB_PAT_KEY`           | yes      | —       | GitHub token, used for both GraphQL and MCP auth |
| `REPO_OWNER` / `REPO_NAME` | yes      | —       | Repository to triage                 |
| `ANTHROPIC_API_KEY`        | yes      | —       | Research, drafting, summaries        |
| `VOYAGEAI_API_KEY`         | yes      | —       | Embeddings and reranking             |
| `OPENAI_API_KEY`           | yes      | —       | Reserved for alternate models; must be *present*, may be left blank |
| `DATABASE_URL`             | yes      | —       | Postgres connection (host-facing)    |
| `REDIS_URL`                | yes      | —       | Celery broker; read directly by `workers/celery_app.py` |
| `RESEARCH_MODEL`           | no       | `claude-sonnet-4-6` | Model for research + draft |
| `RESEARCH_MAX_ITERATIONS`  | no       | `6`     | Hard cap on research tool-loop passes |
| `GITHUB_MCP_URL`           | no       | `https://api.githubcopilot.com/mcp/` | GitHub MCP endpoint |
| `RERANK_MODEL`             | no       | `rerank-2.5` | Voyage reranker                 |
| `REQUEST_TIMEOUT`          | no       | `30`    | Outbound request timeout (seconds)   |
| `DEBUG`                    | no       | `false` | Debug flag                           |

`POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` are consumed by
`docker-compose.yml` to provision the database container.

---

## Database & migrations

Schema changes go through Alembic; never rely on autogenerate blindly against a
populated database (it will try to drop renamed tables). Migrations live in
`migrations/versions/`:

| Revision       | Change                                                |
| -------------- | ----------------------------------------------------- |
| `32d36e234bec` | Baseline of the existing schema                       |
| `b7e2c1a9f4d3` | Split the single `issues` table into `open-issues` / `closed-issues` |
| `9a1c7e5b2f84` | Add the generated `search_vector` column for full-text search |

```bash
uv run alembic upgrade head          # apply
uv run alembic revision -m "..."     # new revision (hand-edit for renames)
uv run alembic downgrade -1          # roll back one
```

The `embeddings` (pgvector) column is excluded from autogenerate diffs in
`migrations/env.py`.

---

## Design notes

A few decisions that are easy to miss from the code alone:

- **Two tables, not one flag.** Corpus and queue have genuinely different shapes
  (`fix_summary` is mandatory in one, meaningless in the other) and opposite
  access patterns — one is read by retrieval, one is written by the poller.
- **Lazy, cached enrichment.** Summaries and embeddings are computed at intake,
  not at sync, and persisted on the row. Most open issues are never triaged, and
  the ones that are only pay once.
- **One corpus search per run.** The `rag_used` latch caps the most expensive tool
  (embed + two queries + rerank) at one call, and the research prompt tells the
  model to spend it deliberately rather than probing.
- **Hard iteration ceiling.** The cap isn't a prompt request — past the limit the
  tools simply aren't bound, so a run cannot loop indefinitely.
- **Read-only by construction.** MCP is restricted twice over (allow-list plus the
  `X-MCP-Readonly` header), so no code path can mutate the upstream repository.
- **Structured hand-off.** Draft consumes a validated `ResearchBrief`, never the
  raw transcript — the schema is what makes "cite your sources" enforceable.
- **Prompt-injection posture.** Issue bodies and tool results are wrapped in tags
  and every prompt states that this content is data, never instructions.
- **Cached clients.** The Anthropic, Voyage, MCP and LLM factories are all
  `lru_cache`d so importing a module never opens a connection and per-request
  cost stays flat.

---

## Status

**Working**

- Issue ingestion for both open and closed issues, with fix detection and noise
  filtering.
- The corpus: embeddings, generated full-text vectors, hybrid retrieval, RRF and
  reranking.
- The polling / backfill jobs and the open↔closed promotion lifecycle.
- The full agent path — `intake → research ⇄ tools → finalize_research → draft` —
  with GitHub MCP tooling, a structured research brief, and `POST /agent/run`.

**Next**

- `gate` — review the draft before anything is posted, using the brief's
  `confidence` and `needs_escalation` signals that are already produced today.
- `escalate` — hand low-confidence issues to a human with the brief attached.
- A runs table persisting drafts, briefs and tool logs for auditability.
- Evals: draft-vs-real-fix comparison over the corpus, plus retrieval metrics.
- Observability, auth on the admin routes, and the dashboard frontend.
