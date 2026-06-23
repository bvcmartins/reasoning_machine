# 🧠 Reasoning Machine

A Streamlit research agent over a local Ollama model, with a **LangSmith-style run
tree** so you can inspect every step the model takes. It pairs a LangChain
tool-calling loop (Tavily / arXiv) with a raw reliability *substrate*
(self-consistency, verifier asymmetry, architect→editor, adversarial probe) ported
from the `claude_code_from_scratch_v2` notebook — minus the coding tools.

## Two answer paths

| Mode | What runs | Tools |
|------|-----------|-------|
| 🔬 **Research agent** | LangChain `AgentExecutor` loop | Tavily, arXiv |
| 🧠 **Direct reasoning** | raw `adaptive_think` (difficulty/type routing → self-consistency / asymmetric-solve / decompose) | none |

Optional **post-processing** (self-refine · verify 1–10 · adversarial probe) runs
the raw substrate over the produced answer. Both paths and all post-processing feed
**one** run tree, shown in a collapsible HTML waterfall.

## Quick start

```bash
cp .env.example .env       # fill in ADMIN_PASSWORD_HASH, COOKIE_KEY, TAVILY_API_KEY
ollama pull qwen3:32b && ollama pull qwen3:8b

cd src
pip install -r ../deploy/requirements.txt
streamlit run streamlit_app.py        # http://localhost:8501
```

Generate the password hash:
```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'YOURPASS', bcrypt.gensalt()).decode())"
```

## Container (port 8503)

```bash
podman build -f deploy/Containerfile -t localhost/reasoning-machine:latest .
bash src/start_container.sh
```

See [`CLAUDE.md`](CLAUDE.md) for the architecture and the LangChain/raw split.
