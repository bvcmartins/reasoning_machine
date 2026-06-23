"""Reasoning engine — the v2 notebook's reliability *substrate*, in raw code.

The langchain/raw split for this project: LangChain owns the agent loop and the
tools (see `tools.py`); the run-tree comes from a callback handler (see
`trace_view.py`). This module owns the bespoke reliability machinery that has no
clean LangChain equivalent, kept as raw calls to a local vLLM server's
OpenAI-compatible /v1/chat/completions (serving openai/gpt-oss-20b):

  * a single thin model client (`chat_complete`) — no streaming, errors bubble;
  * the cognitive substrate: think_then_answer, difficulty/type routing,
    self-consistency, verifier asymmetry, adaptive_think;
  * the hardening stack: architect→editor, self-refine, adversarial probe.

These hit vLLM directly (raw) because they need behaviour the loop does not:
per-call token counts, json-constrained output, and cheap parallel sampling.
gpt-oss returns its reasoning out-of-band; `chat_complete` folds it back inline as
<think>...</think> so the substrate's split_think/parse_json work uniformly. The
LangChain agent loop (ChatOpenAI) is the *other* plumbing; both point at the same
vLLM server.

Observability: every model call goes through `tracer`. By default `tracer` is a
no-op; `trace_view.py` rebinds it to the `RunRecorder`, which records these calls
into the same run tree as the LangChain agent loop. Behaviour never changes.
"""
import contextlib
import json
import logging
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

# --- logging -----------------------------------------------------------------
AGENT_LOG_LEVEL = os.environ.get("AGENT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=AGENT_LOG_LEVEL,
    format="%(asctime)s | %(name)-16s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reasoning")
log_llm = logging.getLogger("reasoning.llm")

# --- configuration -----------------------------------------------------------
# Local vLLM server, OpenAI-compatible API (serving openai/gpt-oss-20b). Locally
# :8000; in the container the host is host.docker.internal (start_container.sh).
# The api_key is unused by vLLM but the protocol requires a non-empty bearer token.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
ACTIVE_ENDPOINT = VLLM_BASE_URL  # for logging / the sidebar health line

# Model per role. gpt-oss is a single model, so the reasoning/fast split collapses
# onto it by default; the roles are kept so the substrate's call sites stay
# self-documenting (and could later differ by reasoning_effort).
#   reasoning: architect, verifier, adversary — the hard thinking steps.
#   fast:      classifiers, refinement, parallel samples — high-volume work.
MODELS = {
    "reasoning": os.environ.get("AGENT_REASONING_MODEL", "openai/gpt-oss-20b"),
    "fast":      os.environ.get("AGENT_FAST_MODEL",      "openai/gpt-oss-20b"),
}
MODEL_REASONING = MODELS["reasoning"]
MODEL_FAST = MODELS["fast"]
# The tool-calling lead (used by the LangChain loop in tools.py); defaults to the
# same gpt-oss model as the substrate.
LEAD_MODEL = os.environ.get("AGENT_LEAD_MODEL", MODEL_REASONING)

REQUEST_TIMEOUT_S = int(os.environ.get("REQUEST_TIMEOUT_S", "900"))  # 32b can be slow
MAX_TOOL_OUTPUT = int(os.environ.get("MAX_TOOL_OUTPUT", "8000"))


# =============================================================================
# Observability hook surface.
# =============================================================================
class _NoopTracer:
    """Default tracer: the engine runs standalone with tracing off. `trace_view`
    rebinds `tracer` (below) to the real RunRecorder, which implements this same
    interface, so the raw substrate's calls land in the shared run tree."""

    @contextlib.contextmanager
    def span(self, kind: str, title: str, meta: Optional[str] = None):
        yield

    def event(self, title: str, body: str = "", style: str = "") -> None: ...
    def model_request(self, **kw) -> None: ...
    def model_response(self, **kw) -> None: ...


# Rebound by trace_view.install(). chat_complete and the substrate look this name
# up at call time, so rebinding the module attribute switches tracing on globally.
tracer: Any = _NoopTracer()


def _truncate(s: Any, limit: int = MAX_TOOL_OUTPUT) -> str:
    s = str(s)
    return s if len(s) <= limit else s[:limit] + f"\n... [truncated {len(s) - limit} chars]"


# =============================================================================
# Model client — the single wrapper every RAW model call goes through.
# =============================================================================
def chat_complete(*, model: str, messages: List[Dict[str, Any]],
                  temperature: float = 0.2, json_mode: bool = False,
                  max_tokens: Optional[int] = None, label: str = "substrate",
                  timeout: Optional[float] = None) -> Dict[str, Any]:
    """Single thin model client — vLLM's OpenAI-compatible /v1/chat/completions
    (stream=False). Logs latency + tokens, hands the call to the tracer, and returns
    a `message` dict ({role, content, eval_count}). No retry, no streaming.

    gpt-oss delivers its analysis (reasoning) channel out-of-band in
    `reasoning_content` (or `reasoning`) with a clean `content`; we fold the
    reasoning back inline as <think>...</think> so the substrate's split_think /
    strip_think / parse_json work unchanged. json_mode maps to
    response_format=json_object.
    """
    log_llm.info(f"-> {model:18s} [{label}] msgs={len(messages)} json={json_mode}")
    tracer.model_request(label=label, model=model, messages=messages, json_mode=json_mode)

    payload: Dict[str, Any] = {"model": model, "messages": messages,
                               "stream": False, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    t0 = time.time()
    resp = requests.post(f"{VLLM_BASE_URL}/chat/completions", json=payload,
                         headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                         timeout=timeout or REQUEST_TIMEOUT_S)
    dt = time.time() - t0
    if resp.status_code != 200:
        log_llm.error(f"<- HTTP {resp.status_code}: {resp.text[:400]}")
        resp.raise_for_status()

    data = resp.json()
    m = (data.get("choices") or [{}])[0].get("message", {}) or {}
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    if reasoning:
        content = f"<think>{reasoning}</think>{content}"
    tokens = (data.get("usage") or {}).get("completion_tokens", 0)

    msg: Dict[str, Any] = {"role": "assistant", "content": content, "eval_count": tokens}
    log_llm.info(f"<- {model:18s} [{label}] {dt:5.1f}s text={len(content)}ch tokens={tokens}")
    tracer.model_response(label=label, model=model, content=content, tokens=tokens, dt=dt)
    return msg


# --- parsing helpers ---------------------------------------------------------
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove the <think> block (folded in by chat_complete), returning the answer."""
    return _THINK_RE.sub("", text or "").strip()


def split_think(text: str):
    """Return (thinking, answer) by splitting on the <think> block."""
    m = _THINK_RE.search(text or "")
    return (m.group(1).strip() if m else "", strip_think(text))


def parse_json(text: str) -> Any:
    """Tolerant JSON parse: strip <think>, then fall back to the first {...} span."""
    text = strip_think(text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def backend_healthcheck() -> Dict[str, Any]:
    """Return {ok, models, missing} for the configured roles by querying vLLM's
    /v1/models. Never raises."""
    try:
        r = requests.get(f"{VLLM_BASE_URL}/models", timeout=5,
                         headers={"Authorization": f"Bearer {VLLM_API_KEY}"})
        r.raise_for_status()
        tags = [m["id"] for m in r.json().get("data", [])]
        # vLLM serves one model; flag a configured role only if its id is absent.
        missing = [f"{role}={name}" for role, name in
                   {**MODELS, "lead": LEAD_MODEL}.items() if name not in tags]
        log_llm.info(f"healthcheck OK -- {len(tags)} model(s); missing={missing}")
        return {"ok": True, "models": tags, "missing": missing}
    except Exception as e:
        log_llm.error(f"healthcheck FAILED: {e}")
        return {"ok": False, "models": [], "missing": [], "error": str(e)}


# =============================================================================
# Cognitive substrate — think-before-answer, difficulty/type routing.
# =============================================================================
DIRECT_SYSTEM_PROMPT = (
    "You are Reasoning Machine, a careful, senior research analyst. You think "
    "before you answer and you name your assumptions before you commit to them.\n\n"
    "RULES OF ENGAGEMENT:\n"
    "1. Prefer evidence over recollection; distinguish what you know from what you infer.\n"
    "2. If you do not know something, say so. Do not invent facts or citations.\n"
    "3. Be concise and structured. Lead with the answer, then the support."
)


@dataclass
class ThoughtfulResponse:
    thinking: str
    answer: str
    output_tokens: int


def think_then_answer(query: str, model: str = MODEL_FAST,
                      temperature: float = 0.3, max_tokens: int = 2048,
                      system: str = DIRECT_SYSTEM_PROMPT) -> ThoughtfulResponse:
    """One deliberate pass; the thinking arrives as a <think> block (gpt-oss's
    reasoning channel, folded inline by chat_complete)."""
    msg = chat_complete(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": query}],
        temperature=temperature, max_tokens=max_tokens, label="think",
    )
    thinking, answer = split_think(msg.get("content", ""))
    return ThoughtfulResponse(thinking=thinking, answer=answer,
                              output_tokens=msg.get("eval_count", 0))


def estimate_difficulty(query: str) -> str:
    """Cheap JSON classifier on the fast model (json_mode suppresses <think>)."""
    msg = chat_complete(
        model=MODEL_FAST, json_mode=True, temperature=0.0, max_tokens=64, label="difficulty",
        messages=[
            {"role": "system", "content":
             'Classify the difficulty of the task. Output JSON: '
             '{"difficulty": one of "trivial","easy","medium","hard","extreme"}.'},
            {"role": "user", "content": query},
        ],
    )
    try:
        return str(parse_json(msg["content"]).get("difficulty", "medium")).lower()
    except Exception:
        return "medium"


# Budgets scaled up vs the article — a thinking model spends tokens on <think> too.
THINKING_BUDGETS = {"trivial": 256, "easy": 512, "medium": 1500,
                    "hard": 3000, "extreme": 6000}

PROBLEM_TYPES = ["convergent", "divergent", "exploratory", "structural"]


def classify_problem(query: str) -> dict:
    """Classify the KIND of problem (orthogonal to difficulty). One cheap JSON call."""
    msg = chat_complete(
        model=MODEL_FAST, json_mode=True, temperature=0.0, max_tokens=128, label="classify",
        messages=[
            {"role": "system", "content":
             "Classify the KIND of problem (not its difficulty). Output JSON: "
             '{"type": one of "convergent","divergent","exploratory","structural", '
             '"reason": str (one sentence)}.\n'
             "convergent  = one correct/defensible answer (a fact, a calculation).\n"
             "divergent   = many valid answers (designs, strategies, recommendations).\n"
             "exploratory = open-ended or under-specified (research, 'implications of').\n"
             "structural  = needs decomposition into parts before answering."},
            {"role": "user", "content": query},
        ],
    )
    try:
        d = parse_json(msg["content"])
        ptype = str(d.get("type", "convergent")).lower()
        if ptype not in PROBLEM_TYPES:
            ptype = "convergent"
        return {"type": ptype, "reason": str(d.get("reason", ""))}
    except Exception:
        return {"type": "convergent", "reason": "fallback (classifier parse failed)"}


TYPE_STRATEGY = {
    "convergent":  "self_consistency",   # majority vote over independent samples
    "divergent":   "asymmetric_solve",   # diverse candidates + a strong verifier ranks them
    "exploratory": "wide_pass",          # one higher-temperature, higher-budget pass
    "structural":  "decompose",          # plan-first single pass
}


def self_consistency(query: str, k: int = 3, model: str = MODEL_FAST) -> dict:
    """Sample k answers (parallel), return the majority bucket (first 60 chars)."""
    with tracer.span("strategy", f"self-consistency (k={k})"):
        with ThreadPoolExecutor(max_workers=k) as ex:
            samples = list(ex.map(
                lambda _: think_then_answer(query, model=model, temperature=0.7).answer, range(k)))
        keys = [s[:60].lower() for s in samples]
        winner_key, votes = Counter(keys).most_common(1)[0]
        winner = next(s for s in samples if s[:60].lower() == winner_key)
        tracer.event(f"self-consistency: {votes}/{k} agree ({votes / k:.0%})",
                     f"winner: {winner[:200]}")
        return {"winner": winner, "votes": votes, "k": k,
                "agreement": votes / k, "all_samples": samples}


VERIFIER_SYSTEM = (
    "You are a careful, structured verifier. Score the candidate on a 1-10 scale "
    "(10 = perfect, 1 = unusable). Score FACTS and correctness, not style. "
    'Output JSON: {"score": int, "reason": str (one sentence)}.'
)


def verifier_score(question: str, candidate: str,
                   verifier_model: str = MODEL_REASONING) -> dict:
    msg = chat_complete(
        model=verifier_model, json_mode=True, temperature=0.0, max_tokens=300, label="verify",
        messages=[{"role": "system", "content": VERIFIER_SYSTEM},
                  {"role": "user", "content": f"QUESTION:\n{question}\n\nCANDIDATE:\n{candidate}"}],
    )
    try:
        verdict = parse_json(msg["content"])
    except Exception as e:
        verdict = {"score": 0, "reason": f"verifier parse error: {e}"}
    tracer.event(f"verifier score: {verdict.get('score')}/10", str(verdict.get("reason", "")))
    return verdict


def asymmetric_solve(query: str, n_candidates: int = 3) -> dict:
    """Verifier asymmetry: cheap generator (n candidates) + one strong ranking call."""
    with tracer.span("strategy", f"verifier-asymmetry (best of {n_candidates})"):
        with ThreadPoolExecutor(max_workers=n_candidates) as ex:
            candidates = list(ex.map(
                lambda _: think_then_answer(query, model=MODEL_FAST, temperature=0.7).answer,
                range(n_candidates)))
        rank_prompt = (
            'Rank these candidates best-to-worst. Output JSON: '
            '{"ranking": [{"rank": 1, "index": 0, "reason": "..."}, ...]}.\n\n'
            + "\n\n".join(f"CANDIDATE {i}:\n{c}" for i, c in enumerate(candidates)))
        msg = chat_complete(model=MODEL_REASONING, json_mode=True, max_tokens=600, label="rank",
                            messages=[{"role": "user", "content": rank_prompt}])
        try:
            ranking = parse_json(msg["content"])["ranking"]
            idx = ranking[0]["index"]
            tracer.event(f"asymmetry: picked candidate #{idx} of {n_candidates}",
                         str(ranking[0].get("reason", "")))
            return {"winner": candidates[idx], "winner_reason": ranking[0].get("reason", ""),
                    "ranking": ranking}
        except Exception:
            tracer.event("asymmetry: ranking parse failed -- falling back to candidate #0")
            return {"winner": candidates[0], "winner_reason": "fallback (ranking parse failed)",
                    "ranking": []}


def adaptive_think(query: str, route: bool = True) -> dict:
    """Compute-adaptive effort on both axes: difficulty -> token budget, problem
    type -> solving strategy. This 'direct reasoning' path does NOT use tools."""
    difficulty = estimate_difficulty(query)
    budget = THINKING_BUDGETS.get(difficulty, 1500)
    problem = classify_problem(query) if route else {"type": "convergent", "reason": "routing off"}
    ptype = problem["type"]
    strategy = TYPE_STRATEGY.get(ptype, "self_consistency") if route else "single_pass"
    log.info(f"[adaptive] difficulty={difficulty} type={ptype} -> budget {budget}, strategy {strategy}")

    with tracer.span("strategy", f"adaptive route: {difficulty} / {ptype} -> {strategy}",
                     meta=f"budget={budget} tokens; {problem.get('reason', '')}"):
        if strategy == "self_consistency":
            r = self_consistency(query, k=3)
            answer, extra = r["winner"], {"agreement": r["agreement"], "votes": r["votes"]}
        elif strategy == "asymmetric_solve":
            r = asymmetric_solve(query, n_candidates=3)
            answer, extra = r["winner"], {"winner_reason": r["winner_reason"]}
        elif strategy == "decompose":
            r = think_then_answer("Break this into parts, solve each, then assemble the result:\n\n"
                                  + query, max_tokens=budget)
            answer, extra = r.answer, {"actual_tokens": r.output_tokens}
        elif strategy == "wide_pass":
            r = think_then_answer(query, temperature=0.7, max_tokens=max(budget, 3000))
            answer, extra = r.answer, {"actual_tokens": r.output_tokens}
        else:  # single_pass (route=False)
            r = think_then_answer(query, max_tokens=budget)
            answer, extra = r.answer, {"actual_tokens": r.output_tokens}

    return {"difficulty": difficulty, "type": ptype, "reason": problem["reason"],
            "budget": budget, "strategy": strategy, "answer": answer, **extra}


# =============================================================================
# Hardening stack — architect→editor, self-refine, adversarial probe.
# =============================================================================
ARCHITECT_SYSTEM = (
    "You are a senior architect. Given a task, produce a STRUCTURED PLAN the editor will "
    "execute. Do NOT produce the final answer -- produce the plan. Output JSON: "
    '{"plan": [{"section": str, "intent": str, "key_constraints": [str]}], '
    '"design_decisions": [{"decision": str, "rationale": str}]}. Be ruthless about constraints.'
)
EDITOR_SYSTEM = (
    # /no_think: the architect already deliberated -- the editor just executes the plan.
    "/no_think You are an editor. The architect produced a structured plan. Execute it "
    "precisely: produce the final output that satisfies the plan. Do NOT redesign, add, "
    "or skip sections. Output the final result only."
)


def architect_editor_solve(task: str, editor_max_tokens: int = 3072) -> dict:
    with tracer.span("strategy", "architect -> editor"):
        arch = chat_complete(model=MODEL_REASONING, json_mode=True, temperature=0.2,
                             max_tokens=1200, label="architect",
                             messages=[{"role": "system", "content": ARCHITECT_SYSTEM},
                                       {"role": "user", "content": f"TASK:\n{task}"}])
        try:
            plan = parse_json(arch["content"])
        except Exception:
            plan = {"plan": [], "design_decisions": []}
        sections = [s.get("section", "?") for s in plan.get("plan", [])]
        tracer.event(f"architect plan: {len(sections)} section(s)",
                     ", ".join(sections) if sections else "(plan parse failed)")
        edit = chat_complete(model=MODEL_FAST, temperature=0.3, max_tokens=editor_max_tokens,
                             label="editor",
                             messages=[{"role": "system", "content": EDITOR_SYSTEM},
                                       {"role": "user", "content":
                                        f"TASK:\n{task}\n\nARCHITECT PLAN:\n{json.dumps(plan, indent=2)}"
                                        "\n\nProduce the final output now."}])
        return {"plan": plan, "output": strip_think(edit.get("content", "")),
                "architect_tokens": arch.get("eval_count", 0), "editor_tokens": edit.get("eval_count", 0)}


def self_refine(query: str, iterations: int = 2, model: str = MODEL_FAST) -> dict:
    """Generate, then critique-and-rewrite `iterations` times."""
    with tracer.span("strategy", f"self-refine ({iterations} iter)"):
        current = think_then_answer(query, model=model, max_tokens=1500).answer
        history = [{"iteration": 0, "output": current, "critique": None}]
        for k in range(1, iterations + 1):
            crit = chat_complete(model=model, temperature=0.3, max_tokens=600, label="critique",
                                 messages=[{"role": "user", "content": query},
                                           {"role": "assistant", "content": current},
                                           {"role": "user", "content":
                                            "Critique your output as a strict reviewer. List 2-5 "
                                            "specific issues. If it is already excellent, say so."}])
            critique = strip_think(crit.get("content", ""))
            ref = chat_complete(model=model, temperature=0.3, max_tokens=1500, label="refine",
                                messages=[{"role": "user", "content": query},
                                          {"role": "user", "content":
                                           f"Previous output:\n{current}\n\nYour critique:\n{critique}"
                                           "\n\nProduce a refined version addressing every point."}])
            current = strip_think(ref.get("content", ""))
            history.append({"iteration": k, "output": current, "critique": critique})
            tracer.event(f"self-refine iteration {k} complete")
    return {"final": current, "history": history, "iterations_run": iterations}


ADVERSARIAL_SYSTEM = (
    "You are a hostile critic. Find ways the candidate answer is WRONG, incomplete, or "
    "misleading: unstated assumptions, factual errors, missing caveats, weak evidence. "
    'Output JSON: {"attacks": [{"category": str, "scenario": str, "why_it_breaks": str, '
    '"severity": "critical"|"major"|"minor"}]}.'
)


def adversarial_probe(target_description: str, candidate_output: str, n_max: int = 4) -> list:
    msg = chat_complete(model=MODEL_REASONING, json_mode=True, temperature=0.4, max_tokens=800,
                        label="adversary",
                        messages=[{"role": "system", "content": ADVERSARIAL_SYSTEM},
                                  {"role": "user", "content":
                                   f"QUESTION:\n{target_description}\n\nCANDIDATE ANSWER:\n{candidate_output}"
                                   f"\n\nFind up to {n_max} weaknesses."}])
    try:
        attacks = parse_json(msg["content"]).get("attacks", [])
    except Exception:
        attacks = []
    if attacks:
        tracer.event(f"adversary found {len(attacks)} weakness(es)",
                     "\n".join(f"[{a.get('severity', '?')}] {a.get('scenario', '')}" for a in attacks))
    else:
        tracer.event("adversary found no weaknesses")
    return attacks


log.info(f"Reasoning engine ready. vLLM={ACTIVE_ENDPOINT} "
         f"lead={LEAD_MODEL} reasoning={MODEL_REASONING} fast={MODEL_FAST}")
