# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A Streamlit research agent backed by a local **vLLM** server exposing an
OpenAI-compatible API (serving **`openai/gpt-oss-20b`**). It ports the reliability
*substrate* from `agentic_patterns/.../claude_code_from_scratch_v2.ipynb` (minus
the coding tools) and gives it two research tools — **Tavily** (web) and
**arXiv** — plus a **LangSmith-style run tree** that shows every
step the model takes. Sibling of `personal_assistant`; same auth / container /
deploy pattern. Runs on port **8503** (quest_portfolio 8501, personal_assistant 8502).

## The langchain / raw split (important)

The project deliberately runs **two model plumbings against one vLLM server**:

- **LangChain owns the agent loop and the tools** — `ChatOpenAI` (OpenAI-compatible,
  pointed at vLLM) + `create_tool_calling_agent` + `AgentExecutor`, with
  Tavily/arXiv as LangChain tools (`tools.py`).
- **Raw `requests` → vLLM `/v1/chat/completions` owns the reliability substrate** —
  the bespoke machinery with no clean LangChain equivalent: difficulty/type routing,
  self-consistency (parallel majority vote), verifier asymmetry, architect→editor,
  self-refine, adversarial probe (`reasoning_engine.py`). These need behaviour the
  loop doesn't: per-call token counts, json-constrained output, cheap parallel
  sampling. gpt-oss returns its reasoning out-of-band; `chat_complete` folds it back
  inline as `<think>...</think>` so the substrate's `split_think`/`parse_json` are
  uniform.
- **One trace over both** — a single `RunRecorder` (`trace_view.py`) is *both* a
  LangChain `BaseCallbackHandler` *and* the engine tracer (`span`/`event`/
  `model_request`/`model_response`). `trace_view.install()` rebinds
  `reasoning_engine.tracer` to it, so the LC loop and the raw substrate feed one
  nested run tree, rendered as HTML.

## Running the App

**Locally (from `src/`):**
```bash
cd src
streamlit run streamlit_app.py
```

**Build and run with Podman:**
```bash
podman build -f deploy/Containerfile -t localhost/reasoning-machine:latest .
bash src/start_container.sh
```

The container reaches the host's vLLM via `host.docker.internal`
(`--add-host=host.docker.internal:host-gateway` in `start_container.sh`).

## Environment Variables (`.env`)

See `.env.example`. Loaded from project-root `.env`, passed via `--env-file`:
- `ADMIN_USERNAME` / `ADMIN_PASSWORD_HASH` (bcrypt) / `COOKIE_KEY` — auth
- `TAVILY_API_KEY` — web-search tool
- `VLLM_BASE_URL` — vLLM endpoint (`http://localhost:8000/v1` local;
  `http://host.docker.internal:8000/v1` in the container)
- `VLLM_API_KEY` — unused by vLLM, but the protocol needs a non-empty token
- `AGENT_REASONING_MODEL` / `AGENT_FAST_MODEL` / `AGENT_LEAD_MODEL` — model
  overrides; all default to `openai/gpt-oss-20b` (gpt-oss is one model, so the
  reasoning/fast/lead roles collapse onto it)

## Architecture (`src/`)

- **`reasoning_engine.py`** — RAW substrate only. `chat_complete` (single vLLM
  client; folds gpt-oss's reasoning channel into an inline `<think>` block),
  parsing helpers, `think_then_answer`, `estimate_difficulty`, `classify_problem`,
  `self_consistency`, `verifier_score`, `asymmetric_solve`, `adaptive_think`,
  `architect_editor_solve`, `self_refine`, `adversarial_probe`. Module-global
  `tracer` (no-op by default; rebound by `trace_view`).
- **`tools.py`** — LangChain tools (Tavily/arXiv) + a reasoning-aware
  `ChatOpenAI` (`disable_streaming=True` so gpt-oss's out-of-band reasoning is
  recovered) + `AgentExecutor`; `research_agent()` runs a turn with the recorder as
  callback; `extract_sources()` pulls `{title, link}` from tool observations.
- **`trace_view.py`** — `RunRecorder` (LC callback handler + engine tracer) and
  `render_trace()` → HTML; `install()` rebinds the engine tracer; `reset()`/`finish()`
  bracket a turn.
- **`streamlit_app.py`** — UI: auth gate, sidebar (mode = Research agent / Direct
  reasoning; post-processing = self-refine / verify / adversarial), per-turn
  `recorder.reset()`, answer + sources + thinking + embedded run-trace expander.
- **`auth_config.py`** — streamlit-authenticator wrapper, cookie
  `reasoning_machine_cookie`.

## vLLM dependency

The model is **not** bundled in the image — it runs on the host's vLLM server at
`:8000` (OpenAI-compatible), serving `openai/gpt-oss-20b`. Launch vLLM with a
gpt-oss reasoning parser and tool-call parser, and `--served-model-name
openai/gpt-oss-20b` so it matches `VLLM_MODEL`. See the `personal_assistant`
vLLM setup doc for the host/systemd setup. Note gpt-oss is MXFP4 (~14 GB) and will
contend with other GPU workloads (e.g. the local training job) on the 24 GB card.
