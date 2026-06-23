"""Reasoning Machine — a Streamlit research agent over a local vLLM server
(openai/gpt-oss-20b).

Two answer paths share one model server and one trace:
  • Research agent — a LangChain tool-calling loop (Tavily / arXiv).
  • Direct reasoning — the raw reliability substrate (difficulty/type routing,
    self-consistency, asymmetric verification) with no tools.

Optional post-processing (verify / self-refine / adversarial probe) runs the raw
substrate on the produced answer. Every step — LangChain loop and raw substrate
alike — is captured by one RunRecorder and shown in a LangSmith-style run tree.
"""
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

import reasoning_engine as engine
import trace_view
from auth_config import require_authentication
from tools import build_agent_executor, research_agent

load_dotenv()

MODE_RESEARCH = "🔬 Research agent (tools)"
MODE_DIRECT = "🧠 Direct reasoning (no tools)"


def init_page() -> None:
    st.set_page_config(page_title="Reasoning Machine", page_icon="🧠", layout="wide")
    st.header("🧠 Reasoning Machine")
    st.sidebar.title("Options")
    st.sidebar.caption(
        f"Lead: `{engine.LEAD_MODEL}` · reasoning: `{engine.MODEL_REASONING}` · "
        f"fast: `{engine.MODEL_FAST}`")


@st.cache_data(ttl=30, show_spinner=False)
def _health():
    return engine.backend_healthcheck()


def render_sidebar_status() -> None:
    h = _health()
    if h["ok"] and not h["missing"]:
        st.sidebar.success(f"vLLM OK · {len(h['models'])} model(s)")
    elif h["ok"]:
        st.sidebar.warning("vLLM up, missing: " + ", ".join(h["missing"]))
    else:
        st.sidebar.error(f"vLLM unreachable @ {engine.ACTIVE_ENDPOINT}")


@st.cache_resource(show_spinner="Building research agent …")
def get_agent_executor():
    return build_agent_executor()


def get_recorder() -> trace_view.RunRecorder:
    if "recorder" not in st.session_state:
        st.session_state.recorder = trace_view.RunRecorder()
    # Rebind the engine's tracer to this session's recorder every run (the engine
    # tracer is a module global; reinstalling is cheap and keeps it pointed here).
    trace_view.install(st.session_state.recorder)
    return st.session_state.recorder


def init_state() -> None:
    if st.sidebar.button("Clear Conversation", key="clear") or "history" not in st.session_state:
        st.session_state.history = []      # display dicts
        st.session_state.chat_history = []  # LangChain messages (clean answers)


def run_turn(mode: str, query: str, post: dict, recorder) -> dict:
    """Produce an answer for one turn, then apply opted-in post-processing."""
    if mode == MODE_RESEARCH:
        executor = get_agent_executor()
        res = research_agent(executor, query, st.session_state.chat_history, recorder)
        answer = res["answer"]
        entry = {"thinking": res["thinking"], "sources": res["sources"],
                 "routing": None}
    else:
        with recorder.span("strategy", "direct reasoning (adaptive_think)"):
            r = engine.adaptive_think(query)
        answer = r["answer"]
        entry = {"thinking": "", "sources": [],
                 "routing": {"difficulty": r["difficulty"], "type": r["type"],
                             "strategy": r["strategy"], "reason": r.get("reason", "")}}

    # ----- post-processing (raw substrate, traced) -----------------------
    if post.get("refine"):
        with recorder.span("strategy", "post: self-refine"):
            answer = engine.self_refine(query, iterations=1)["final"]
    if post.get("verify"):
        entry["verdict"] = engine.verifier_score(query, answer)
    if post.get("adversarial"):
        entry["attacks"] = engine.adversarial_probe(query, answer)

    entry["answer"] = answer
    return entry


def render_message(entry: dict) -> None:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.markdown(entry["content"])
        return

    with st.chat_message("assistant"):
        main_col, side_col = st.columns([3, 1])
        with main_col:
            st.markdown(entry["answer"])

            if entry.get("routing"):
                r = entry["routing"]
                st.caption(f"route · {r['difficulty']} / {r['type']} → {r['strategy']}"
                           + (f" — {r['reason']}" if r.get("reason") else ""))

            if entry.get("verdict"):
                v = entry["verdict"]
                st.info(f"**Verifier:** {v.get('score', '?')}/10 — {v.get('reason', '')}")

            if entry.get("attacks"):
                with st.expander(f"⚔️ Adversarial probe ({len(entry['attacks'])})", expanded=False):
                    for a in entry["attacks"]:
                        st.markdown(f"- **[{a.get('severity', '?')}] {a.get('category', '')}** — "
                                    f"{a.get('scenario', '')} → _{a.get('why_it_breaks', '')}_")

            if entry.get("sources"):
                st.markdown("**Sources**")
                for s in entry["sources"]:
                    tag = f" · `{s['tool']}`" if s.get("tool") else ""
                    if s.get("link"):
                        st.markdown(f"- [{s['title']}]({s['link']}){tag}")
                    else:
                        st.markdown(f"- {s['title']}{tag}")

        with side_col:
            if entry.get("thinking"):
                with st.expander("🧠 Thinking", expanded=False):
                    st.markdown(entry["thinking"])

        if entry.get("trace_html"):
            with st.expander("🔍 Run trace", expanded=False):
                components.html(entry["trace_html"], height=600, scrolling=True)


def main() -> None:
    init_page()
    if not require_authentication():
        st.stop()

    render_sidebar_status()
    init_state()

    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Mode", [MODE_RESEARCH, MODE_DIRECT], index=0)
    st.sidebar.markdown("**Post-processing**")
    post = {
        "refine": st.sidebar.checkbox("Self-refine answer", value=False),
        "verify": st.sidebar.checkbox("Verify (score 1–10)", value=False),
        "adversarial": st.sidebar.checkbox("Adversarial probe", value=False),
    }

    if user_input := st.chat_input("Ask a research question…"):
        recorder = get_recorder()
        recorder.reset()
        with st.spinner("reasoning …"):
            try:
                entry = run_turn(mode, user_input, post, recorder)
            except Exception as e:  # surface failures instead of a blank screen
                entry = {"answer": f"⚠️ Error: {e}", "thinking": "", "sources": [],
                         "routing": None}
            recorder.finish()
        entry["role"] = "assistant"
        entry["trace_html"] = trace_view.render_trace(recorder)

        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append(entry)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=entry["answer"]))

    for entry in st.session_state.get("history", []):
        render_message(entry)


if __name__ == "__main__":
    main()
