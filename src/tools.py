"""LangChain agent loop + research tools (the LangChain half of the split).

This module owns everything LangChain does cleanly: the tool-calling loop and the
tools themselves. Tavily (web) and arXiv are wrapped as LangChain
tools; an OpenAI-compatible `ChatOpenAI` pointed at the local vLLM server
(openai/gpt-oss-20b) is the lead model; `create_tool_calling_agent` +
`AgentExecutor` run the loop. The raw reliability substrate lives in
`reasoning_engine.py`; the shared trace comes from the `RunRecorder` in
`trace_view.py`.

`research_agent(query, chat_history, recorder)` runs one turn, passing the recorder
as a callback so the loop's chains/llm/tool steps land in the same run tree as the
raw substrate. It returns the answer, the gpt-oss reasoning trace, and the sources
pulled from tool observations.
"""
import os
from typing import Any, Dict, List

try:
    # LangChain < 1.0
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    # LangChain >= 1.0 moved the legacy agent API to langchain-classic
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

import reasoning_engine as engine
from reasoning_engine import split_think

# The lead/loop model and the raw substrate share one vLLM server. The loop uses
# ChatOpenAI (LangChain); the substrate uses raw HTTP — same OpenAI-compatible API.
LEAD_MODEL = engine.LEAD_MODEL

AGENT_SYSTEM_PROMPT = (
    "You are Reasoning Machine, a meticulous research analyst with live tools.\n\n"
    "You have two tools:\n"
    "  • web_search   — current events, recent facts, anything time-sensitive.\n"
    "  • arxiv_search — scientific / technical papers and preprints.\n\n"
    "Process: decide which tool(s) the question needs, gather evidence, then answer.\n"
    "Cite what you used. Prefer primary evidence over recollection; if the tools do "
    "not support a claim, say so rather than inventing it. Lead with the answer, then "
    "the supporting detail."
)


# =============================================================================
# Tools — each wraps a LangChain community utility and returns a string the
# agent reads back. trace_view records args + results via on_tool_start/end.
# =============================================================================
def build_tools() -> list:
    from langchain_community.tools import ArxivQueryRun
    from langchain_community.utilities import ArxivAPIWrapper

    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=4, doc_content_chars_max=3000))

    # Tavily: prefer the dedicated package (langchain-tavily, in the container);
    # fall back to the community tool for local dev where it may not be installed.
    try:
        from langchain_tavily import TavilySearch
        tavily = TavilySearch(max_results=5, api_key=os.environ.get("TAVILY_API_KEY"))

        def _tavily_run(query: str) -> str:
            return str(tavily.invoke({"query": query}))
    except ImportError:
        from langchain_community.tools import TavilySearchResults
        tavily = TavilySearchResults(max_results=5,
                                     tavily_api_key=os.environ.get("TAVILY_API_KEY"))

        def _tavily_run(query: str) -> str:
            return str(tavily.invoke(query))

    @tool
    def web_search(query: str) -> str:
        """Search the live web (Tavily) for current information and recent events."""
        return _tavily_run(query)

    @tool
    def arxiv_search(query: str) -> str:
        """Search arXiv for scientific and technical papers (titles, authors, abstracts)."""
        return arxiv.run(query)

    return [web_search, arxiv_search]


def _build_lead_model(temperature: float):
    """The tool-calling lead: an OpenAI-compatible ChatOpenAI on the local vLLM
    server (gpt-oss). gpt-oss delivers its analysis channel out-of-band in a
    `reasoning`/`reasoning_content` field that langchain-openai drops;
    ReasoningChatOpenAI recovers it into additional_kwargs['reasoning'] (mirrors
    personal_assistant). disable_streaming is required: the drop happens only on
    the streaming path, which AgentExecutor uses for the final turn, so streaming
    would lose the trace.
    """
    from langchain_openai import ChatOpenAI

    class ReasoningChatOpenAI(ChatOpenAI):
        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            choices = getattr(response, "choices", None)
            if choices is None and isinstance(response, dict):
                choices = response.get("choices", [])
            for gen, choice in zip(result.generations, choices or []):
                msg = getattr(choice, "message", None)
                if msg is None and isinstance(choice, dict):
                    msg = choice.get("message", {})
                reasoning = getattr(msg, "reasoning", None) or \
                    getattr(msg, "reasoning_content", None)
                if reasoning is None and hasattr(msg, "model_extra"):
                    extra = msg.model_extra or {}
                    reasoning = extra.get("reasoning") or extra.get("reasoning_content")
                if reasoning is None and isinstance(msg, dict):
                    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
                if reasoning:
                    gen.message.additional_kwargs["reasoning"] = reasoning
            return result

    return ReasoningChatOpenAI(
        model=LEAD_MODEL,
        base_url=engine.VLLM_BASE_URL,
        api_key=engine.VLLM_API_KEY,
        temperature=temperature,
        disable_streaming=True,
    )


def build_agent_executor(temperature: float = 0.0) -> AgentExecutor:
    """Build the tool-calling agent once (cache this in the app)."""
    tools = build_tools()
    model = _build_lead_model(temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(model, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )


# =============================================================================
# Source extraction from tool observations.
# =============================================================================
def _coerce_results(observation: Any) -> List[dict]:
    """Best-effort: turn a tool observation into a list of {title?, url?} dicts."""
    import ast
    import json

    obj = observation
    if isinstance(obj, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                obj = parse(observation)
                break
            except Exception:
                continue
    if isinstance(obj, dict):
        # TavilySearch returns {"results": [...]} (or similar).
        for key in ("results", "items", "documents"):
            if isinstance(obj.get(key), list):
                return obj[key]
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def extract_sources(intermediate_steps: list) -> List[dict]:
    """Pull {title, link} from tool observations, de-duplicated, across all tools."""
    sources: List[dict] = []
    seen: set = set()
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", "")
        for item in _coerce_results(observation):
            if not isinstance(item, dict):
                continue
            link = item.get("url") or item.get("link") or item.get("source") or ""
            title = item.get("title") or item.get("Title") or link or tool_name
            key = link or title
            if not key or key in seen:
                continue
            seen.add(key)
            sources.append({"title": str(title)[:140], "link": link, "tool": tool_name})
    return sources


from langchain_core.callbacks import BaseCallbackHandler


class _ReasoningCollector(BaseCallbackHandler):
    """Capture the reasoning trace from the final LLM turn of the agent loop.

    On the vLLM/gpt-oss path the thinking arrives out-of-band in the message's
    `reasoning` field (surfaced into additional_kwargs by ReasoningChatOpenAI),
    not inline in the answer, so split_think on the output finds nothing. The last
    LLM call in the loop is the final-answer turn, so its reasoning is what we keep.
    """

    def __init__(self) -> None:
        self.reasoning = ""

    def on_llm_end(self, response, **kwargs) -> None:
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                trace = (getattr(message, "additional_kwargs", {}) or {}).get("reasoning")
                if trace:
                    self.reasoning = trace


def research_agent(agent_executor: AgentExecutor, query: str,
                   chat_history: list, recorder=None) -> Dict[str, Any]:
    """Run one research turn. The recorder (if given) traces the whole loop."""
    collector = _ReasoningCollector()
    callbacks = [collector] + ([recorder] if recorder is not None else [])
    result = agent_executor.invoke(
        {"input": query, "chat_history": chat_history}, config={"callbacks": callbacks})
    # Two possible thinking sources:
    #   - vLLM (gpt-oss), normal: thinking arrives out-of-band via the collector; answer is clean.
    #   - fallback: <think> is inline in the answer.
    inline_thinking, answer = split_think(result.get("output", ""))
    thinking = collector.reasoning or inline_thinking
    sources = extract_sources(result.get("intermediate_steps", []))
    return {"answer": answer, "thinking": thinking, "sources": sources,
            "intermediate_steps": result.get("intermediate_steps", [])}
