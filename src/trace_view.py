"""Unified run-tree recorder + LangSmith-style HTML viewer.

This module is the seam that makes the project's two model plumbings show up in
ONE trace. A single `RunRecorder` object is simultaneously:

  * a LangChain `BaseCallbackHandler` — it receives on_chain/llm/tool/agent
    callbacks from the AgentExecutor loop (see tools.py) and turns them into
    tree nodes keyed by LangChain's run_id / parent_run_id; and

  * the engine tracer — it implements `span` / `event` / `model_request` /
    `model_response`, the interface `reasoning_engine`'s raw substrate calls.
    `install(recorder)` rebinds `reasoning_engine.tracer` to it.

Both feed the same `root` node, so the LangChain agent loop and the raw
reliability substrate (self-consistency, architect→editor, verifier, …) render
as one nested waterfall. `render_trace(recorder)` returns an HTML string for
`st.components.v1.html(...)`.

Concurrency note: the substrate fans out model calls across a ThreadPoolExecutor
(self_consistency / asymmetric_solve). The span stack is mutated only on the
main thread; worker threads merely append model nodes under the currently-open
span, and pair their own request/response via thread-local state. A lock guards
all mutation.
"""
import contextlib
import html
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from reasoning_engine import split_think

# LangChain emits a forest of generic Runnable wrappers (RunnableSequence,
# RunnableAssign, prompt templates, output parsers). They carry no signal and
# would bury the real steps, so we pass through them transparently: their
# children re-parent onto the nearest meaningful ancestor.
_SKIP_CHAIN_PREFIXES = ("Runnable", "ChatPromptTemplate", "PromptTemplate")
_SKIP_CHAIN_NAMES = {"agent", "AgentExecutor"}  # AgentExecutor == our root already

_FIELD_LIMIT = 6000


def _clip(s: Any, limit: int = _FIELD_LIMIT) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= limit else s[:limit] + f"\n… [+{len(s) - limit} chars]"


class _Node:
    __slots__ = ("id", "kind", "title", "meta", "t0", "t1", "children",
                 "fields", "tokens", "model")

    def __init__(self, kind: str, title: str, meta: Optional[str] = None):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.title = title
        self.meta = meta
        self.t0 = time.time()
        self.t1: Optional[float] = None
        self.children: List["_Node"] = []
        self.fields: Dict[str, str] = {}
        self.tokens = 0
        self.model: Optional[str] = None

    def dur(self) -> float:
        return (self.t1 or time.time()) - self.t0

    def total_tokens(self) -> int:
        return self.tokens + sum(c.total_tokens() for c in self.children)


def _fmt_messages(messages: List[Dict[str, Any]]) -> str:
    out = []
    for m in messages or []:
        role = m.get("role", "?") if isinstance(m, dict) else getattr(m, "type", "?")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        out.append(f"[{role}]\n{content}")
    return "\n\n".join(out)


class RunRecorder(BaseCallbackHandler):
    """One object, two roles: LangChain callback handler + engine tracer."""

    # let LangChain hand us raw nested run ids even on errors
    raise_error = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tls = threading.local()
        self.reset()

    # -- lifecycle ------------------------------------------------------------
    def reset(self) -> None:
        """Start a fresh tree for a new turn."""
        with self._lock:
            self.root = _Node("agent", "turn")
            self._span_stack: List[_Node] = [self.root]
            self._lc_attach: Dict[Any, _Node] = {}   # run_id -> node children attach to
            self._lc_owned: set = set()               # run_ids that created their own node
            self._tls.open = None

    def finish(self) -> None:
        with self._lock:
            self.root.t1 = time.time()

    # -- shared tree mutation -------------------------------------------------
    def _current_parent(self) -> _Node:
        return self._span_stack[-1]

    def _add(self, parent: _Node, node: _Node) -> _Node:
        with self._lock:
            parent.children.append(node)
        return node

    # =====================================================================
    # Engine tracer API (raw substrate in reasoning_engine.py).
    # =====================================================================
    @contextlib.contextmanager
    def span(self, kind: str, title: str, meta: Optional[str] = None):
        with self._lock:
            node = _Node(kind, title, meta)
            self._current_parent().children.append(node)
            self._span_stack.append(node)
        try:
            yield node
        finally:
            with self._lock:
                node.t1 = time.time()
                if node in self._span_stack:
                    idx = self._span_stack.index(node)
                    del self._span_stack[idx:]   # defensive: pop node + any stragglers

    def event(self, title: str, body: str = "", style: str = "") -> None:
        node = _Node("event", title, meta=style or None)
        node.t1 = node.t0
        if body:
            node.fields["detail"] = _clip(body)
        self._add(self._current_parent(), node)

    def model_request(self, *, label: str, model: str,
                      messages: List[Dict[str, Any]], json_mode: bool = False,
                      **_: Any) -> None:
        with self._lock:
            node = _Node("llm", f"{label} · {model}")
            node.model = model
            node.fields["request"] = _clip(_fmt_messages(messages))
            if json_mode:
                node.fields["format"] = "json"
            self._current_parent().children.append(node)
            self._tls.open = node

    def model_response(self, *, label: str, model: str, content: str,
                       tokens: int = 0, dt: float = 0.0, **_: Any) -> None:
        node = getattr(self._tls, "open", None)
        if node is None:
            return
        node.t1 = time.time()
        node.tokens = tokens or 0
        thinking, answer = split_think(content or "")
        if thinking:
            node.fields["thinking"] = _clip(thinking)
        node.fields["answer"] = _clip(answer)
        self._tls.open = None

    # =====================================================================
    # LangChain BaseCallbackHandler API (the AgentExecutor loop in tools.py).
    # =====================================================================
    def _lc_parent(self, parent_run_id: Any) -> _Node:
        return self._lc_attach.get(parent_run_id) or self._current_parent()

    def on_chain_start(self, serialized, inputs, *, run_id,
                       parent_run_id=None, **kw):
        name = (serialized or {}).get("name") or kw.get("name") or "chain"
        parent = self._lc_parent(parent_run_id)
        if name in _SKIP_CHAIN_NAMES or name.startswith(_SKIP_CHAIN_PREFIXES):
            self._lc_attach[run_id] = parent          # transparent passthrough
            return
        node = _Node("chain", name)
        if isinstance(inputs, dict) and "input" in inputs:
            node.fields["input"] = _clip(inputs["input"])
        self._add(parent, node)
        self._lc_attach[run_id] = node
        self._lc_owned.add(run_id)

    def on_chain_end(self, outputs, *, run_id, **kw):
        if run_id in self._lc_owned:
            node = self._lc_attach.get(run_id)
            if node:
                node.t1 = time.time()

    def on_chat_model_start(self, serialized, messages, *, run_id,
                            parent_run_id=None, **kw):
        # messages: List[List[BaseMessage]] (one list per generation prompt)
        flat = messages[0] if messages else []
        self._lc_llm_start(serialized, flat, run_id, parent_run_id, kw)

    def on_llm_start(self, serialized, prompts, *, run_id,
                     parent_run_id=None, **kw):
        flat = [{"role": "user", "content": p} for p in (prompts or [])]
        self._lc_llm_start(serialized, flat, run_id, parent_run_id, kw)

    def _lc_llm_start(self, serialized, msgs, run_id, parent_run_id, kw):
        name = (serialized or {}).get("name") or kw.get("name") or "llm"
        node = _Node("llm", f"{name}")
        node.fields["request"] = _clip(_fmt_messages(
            [{"role": getattr(m, "type", "?"), "content": getattr(m, "content", m)}
             if not isinstance(m, dict) else m for m in msgs]))
        self._add(self._lc_parent(parent_run_id), node)
        self._lc_attach[run_id] = node
        self._lc_owned.add(run_id)

    def on_llm_end(self, response, *, run_id, **kw):
        node = self._lc_attach.get(run_id)
        if not node:
            return
        node.t1 = time.time()
        text, tokens = self._lc_extract(response)
        thinking, answer = split_think(text)
        if thinking:
            node.fields["thinking"] = _clip(thinking)
        node.fields["answer"] = _clip(answer)
        node.tokens = tokens

    @staticmethod
    def _lc_extract(response) -> tuple:
        text, tokens = "", 0
        try:
            for gens in response.generations:
                for gen in gens:
                    msg = getattr(gen, "message", None)
                    text = (getattr(msg, "content", None) or getattr(gen, "text", "") or text)
                    meta = getattr(msg, "usage_metadata", None) or {}
                    tokens = (meta.get("output_tokens") or tokens)
            llm_out = getattr(response, "llm_output", None) or {}
            tokens = tokens or llm_out.get("eval_count", 0)
        except Exception:
            pass
        return text or "", tokens or 0

    def on_tool_start(self, serialized, input_str, *, run_id,
                      parent_run_id=None, **kw):
        name = (serialized or {}).get("name") or kw.get("name") or "tool"
        node = _Node("tool", name)
        node.fields["args"] = _clip(input_str)
        self._add(self._lc_parent(parent_run_id), node)
        self._lc_attach[run_id] = node
        self._lc_owned.add(run_id)

    def on_tool_end(self, output, *, run_id, **kw):
        node = self._lc_attach.get(run_id)
        if node:
            node.t1 = time.time()
            node.fields["result"] = _clip(output)

    def on_tool_error(self, error, *, run_id, **kw):
        node = self._lc_attach.get(run_id)
        if node:
            node.t1 = time.time()
            node.fields["error"] = _clip(error)

    def on_agent_action(self, action, *, run_id, parent_run_id=None, **kw):
        ev = _Node("event", f"action → {getattr(action, 'tool', '?')}")
        ev.t1 = ev.t0
        log = getattr(action, "log", "")
        if log:
            ev.fields["detail"] = _clip(log)
        self._add(self._lc_parent(parent_run_id), ev)

    def on_llm_error(self, error, *, run_id, **kw):
        node = self._lc_attach.get(run_id)
        if node:
            node.t1 = time.time()
            node.fields["error"] = _clip(error)


# =============================================================================
# Installation: rebind the engine's module-level tracer to this recorder.
# =============================================================================
def install(recorder: RunRecorder) -> RunRecorder:
    """Point reasoning_engine's raw substrate at this recorder. Idempotent."""
    import reasoning_engine as E
    E.tracer = recorder
    return recorder


# =============================================================================
# HTML viewer — a LangSmith-style collapsible run tree with a timing waterfall.
# =============================================================================
_KIND_STYLE = {
    "agent":    ("🧩", "#7C4DFF"),
    "chain":    ("🔗", "#6B7280"),
    "llm":      ("🧠", "#3B82F6"),
    "tool":     ("🛠️", "#10B981"),
    "strategy": ("🎯", "#F59E0B"),
    "iter":     ("🔁", "#F59E0B"),
    "subagent": ("🤖", "#EC4899"),
    "event":    ("•",  "#9CA3AF"),
}

_FIELD_ORDER = ["input", "request", "format", "args", "thinking",
                "answer", "result", "detail", "error"]

_CSS = """
<style>
.rmtrace { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12.5px; color: #E5E7EB; background: #0E1117; padding: 6px 4px; }
.rmtrace details { margin: 0; }
.rmtrace .node { border-left: 2px solid #262730; margin-left: 6px; padding-left: 8px; }
.rmtrace summary { list-style: none; cursor: pointer; padding: 3px 4px;
  border-radius: 5px; display: flex; align-items: center; gap: 8px;
  white-space: nowrap; }
.rmtrace summary::-webkit-details-marker { display: none; }
.rmtrace summary:hover { background: #1A1D26; }
.rmtrace .badge { font-size: 10px; padding: 1px 6px; border-radius: 8px;
  background: #1F2430; color: #cbd5e1; }
.rmtrace .title { font-weight: 600; overflow: hidden; text-overflow: ellipsis; }
.rmtrace .meta { color: #6B7280; font-size: 11px; }
.rmtrace .wf { flex: 1; min-width: 60px; height: 7px; background: #161A22;
  border-radius: 4px; position: relative; }
.rmtrace .wf > span { position: absolute; top: 0; height: 7px; border-radius: 4px;
  opacity: 0.85; }
.rmtrace .fields { margin: 2px 0 6px 18px; }
.rmtrace .field { margin: 3px 0; }
.rmtrace .flabel { color: #9CA3AF; font-size: 10.5px; text-transform: uppercase;
  letter-spacing: .04em; }
.rmtrace pre { margin: 2px 0; padding: 6px 8px; background: #11151C;
  border-radius: 5px; white-space: pre-wrap; word-break: break-word;
  max-height: 280px; overflow: auto; color: #D1D5DB; }
.rmtrace pre.thinking { color: #93C5FD; background: #0F1622; }
.rmtrace pre.answer { color: #E5E7EB; }
.rmtrace pre.error { color: #FCA5A5; background: #1F1416; }
.rmtrace .empty { color: #6B7280; padding: 8px; }
</style>
"""


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _render_node(node: _Node, root_t0: float, total: float, depth: int) -> str:
    icon, color = _KIND_STYLE.get(node.kind, ("•", "#9CA3AF"))
    dur = node.dur()
    left = max(0.0, (node.t0 - root_t0) / total * 100) if total > 0 else 0
    width = min(100.0, max(1.5, dur / total * 100)) if total > 0 else 1.5

    badges = [f'<span class="badge">{dur * 1000:.0f} ms</span>']
    toks = node.total_tokens()
    if toks:
        badges.append(f'<span class="badge">{toks} tok</span>')

    meta = f'<span class="meta">{_esc(node.meta)}</span>' if node.meta else ""
    wf = (f'<span class="wf"><span style="left:{left:.1f}%;width:{width:.1f}%;'
          f'background:{color}"></span></span>')

    summary = (
        f'<summary><span style="color:{color}">{icon}</span>'
        f'<span class="title">{_esc(node.title)}</span>{meta}{wf}'
        f'{"".join(badges)}</summary>')

    # fields
    body = []
    for key in _FIELD_ORDER:
        if key not in node.fields:
            continue
        val = node.fields[key]
        cls = key if key in ("thinking", "answer", "error") else ""
        if key in ("format",):
            body.append(f'<div class="field"><span class="flabel">{key}: {_esc(val)}</span></div>')
        else:
            body.append(
                f'<div class="field"><div class="flabel">{key}</div>'
                f'<pre class="{cls}">{_esc(val)}</pre></div>')
    fields_html = f'<div class="fields">{"".join(body)}</div>' if body else ""

    children = "".join(_render_node(c, root_t0, total, depth + 1) for c in node.children)
    open_attr = " open" if depth < 1 else ""
    return (f'<div class="node"><details{open_attr}>{summary}'
            f'{fields_html}{children}</details></div>')


def render_trace(recorder: RunRecorder) -> str:
    """Return a self-contained HTML string for the recorder's current tree."""
    root = recorder.root
    if not root.children:
        return _CSS + '<div class="rmtrace"><div class="empty">No steps recorded yet.</div></div>'
    root_t0 = root.t0
    total = max(root.dur(), 1e-6)
    inner = "".join(_render_node(c, root_t0, total, 0) for c in root.children)
    summary = (f'<div class="meta" style="padding:4px 6px">'
               f'total {total * 1000:.0f} ms · {root.total_tokens()} tokens · '
               f'{len(root.children)} top-level step(s)</div>')
    return _CSS + f'<div class="rmtrace">{summary}{inner}</div>'
