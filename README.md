# Agentic Research Assistant

<div align="center">

[![Live Demo](https://img.shields.io/badge/🔬%20Try%20the%20Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://agentic-research-assistant-jxbypfktqrsbe2sw7wjvm9.streamlit.app)

</div>

![LangGraph](https://img.shields.io/badge/LangGraph-1.x-blueviolet?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=chainlink)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude%20Sonnet-orange?logo=anthropic)
![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41+-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)

A **production-quality multi-agent AI system** that autonomously researches any topic end-to-end — no human intervention required. Give it a topic and it plans subtopics, searches the web, synthesizes findings, identifies knowledge gaps, and writes a structured markdown report with citations.

Built with **LangGraph's supervisor pattern** to demonstrate stateful multi-agent orchestration, conditional routing, web search integration, and a live Streamlit UI.

---

## How It Works

The system runs a loop of five agents, each returning control to the Supervisor after every step:

```
                        ┌─────────────────────┐
   Research Topic ───►  │     SUPERVISOR      │ ◄──────────────────┐
                        │  (pure router node) │                    │
                        └──────────┬──────────┘                    │
                                   │  conditional edges            │
              ┌────────────────────┼───────────────┬────────────┐  │
              ▼                    ▼               ▼            ▼  │
        ┌──────────┐       ┌────────────┐   ┌──────────┐  ┌────────┤ │
        │ PLANNER  │       │ RESEARCHER │   │SYNTHESIZER│  │ WRITER │ │
        │          │       │            │   │           │  │        │ │
        │ Breaks   │       │ Tavily web │   │ Combines  │  │ Final  │ │
        │ topic    │       │ search per │   │ findings, │  │ report │ │
        │ into 3-5 │       │ subtopic + │   │ IDs gaps  │  │ w/     │ │
        │ subtopics│       │ Claude     │   │           │  │ cites  │ │
        └────┬─────┘       └─────┬──────┘   └─────┬─────┘  └───┬────┘ │
             └───────────────────┴─────────────────┴────────────┘      │
                              all return to SUPERVISOR ─────────────────┘
```

### Routing Logic

```
No plan yet                              → PLANNER
Plan exists + unsearched subtopics       → RESEARCHER
All subtopics searched + not synthesized → SYNTHESIZER
Gaps found + iterations < max           → RESEARCHER  (gap-fill loop)
No gaps remaining / max iterations hit  → WRITER
Report written                          → END
```

### Agent Roles

| Agent | What it does |
|---|---|
| **Supervisor** | Inspects state and routes — no LLM call, pure conditional logic |
| **Planner** | Asks Claude to decompose the topic into 3–5 focused subtopics |
| **Researcher** | Picks the next unsearched subtopic, runs a Tavily search, extracts insights via Claude |
| **Synthesizer** | Combines all findings, identifies knowledge gaps for follow-up |
| **Writer** | Produces a structured markdown report with inline `[Source N]` citations |

---

## Project Structure

```
agentic-research-assistant/
├── src/
│   ├── agents/
│   │   ├── supervisor.py      # Stateless routing function
│   │   ├── planner.py         # Topic decomposition
│   │   ├── researcher.py      # Web search + insight extraction
│   │   ├── synthesizer.py     # Findings synthesis + gap analysis
│   │   └── writer.py          # Markdown report generation
│   ├── tools/
│   │   └── search.py          # Tavily web search LangChain tool
│   ├── graph/
│   │   ├── state.py           # ResearchState TypedDict + reducers
│   │   └── workflow.py        # LangGraph StateGraph construction
│   └── api/
│       └── main.py            # FastAPI REST API
├── ui/
│   └── app.py                 # Streamlit UI
├── tests/
│   ├── test_state.py          # State reducer + routing unit tests
│   └── test_workflow.py       # Graph compilation tests
├── .streamlit/
│   └── secrets.toml           # Streamlit Cloud secrets template (gitignored)
├── config.py                  # Pydantic Settings
├── requirements.txt
├── Dockerfile                 # API container
├── Dockerfile.ui              # Streamlit UI container
├── docker-compose.yml
└── .env.example
```

---

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/) — for Claude
- [Tavily API key](https://app.tavily.com/) — free tier (1,000 searches/month)

### 1. Clone and install

```bash
git clone https://github.com/Viraj-Pathak/agentic-research-assistant.git
cd agentic-research-assistant
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
LANGCHAIN_API_KEY=         # optional — LangSmith tracing
LANGCHAIN_TRACING_V2=false
```

### 3. Run the Streamlit UI

```bash
streamlit run ui/app.py
# → http://localhost:8501
```

### 4. Run the REST API (optional)

```bash
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs
```

---

## Deploy to Streamlit Cloud

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**
3. Set **Main file path** to `ui/app.py`
4. Under **Advanced settings → Secrets**, add:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
TAVILY_API_KEY    = "tvly-..."
```

5. Click **Deploy** — done.

The app uses `MemorySaver` (in-memory checkpointing) when running on Streamlit Cloud, so no file system persistence is needed.

---

## REST API

### POST `/research` — run a full research pipeline

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Impact of large language models on scientific research"}'
```

```json
{
  "thread_id": "550e8400-...",
  "topic": "Impact of large language models on scientific research",
  "plan": ["LLM applications in drug discovery", "AI-assisted literature review", "..."],
  "report": "# Impact of LLMs on Scientific Research\n\n...",
  "sources": [{"title": "...", "url": "...", "content": "...", "subtopic": "..."}],
  "iterations": 4
}
```

### POST `/research` with `thread_id` — resume a session

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "quantum computing", "thread_id": "your-thread-id"}'
```

### GET `/research/{thread_id}/state` — inspect progress

```bash
curl http://localhost:8000/research/550e8400-.../state
```

---

## Docker

```bash
cp .env.example .env   # fill in your keys
docker-compose up --build
# API → http://localhost:8000
# UI  → http://localhost:8501
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API key (required) |
| `TAVILY_API_KEY` | — | Tavily search key (required) |
| `LANGCHAIN_API_KEY` | — | LangSmith tracing key (optional) |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `MODEL_NAME` | `claude-sonnet-4-6` | Anthropic model |
| `MAX_ITERATIONS` | `3` | Max research loop iterations |
| `MAX_SEARCH_RESULTS` | `5` | Tavily results per query |

---

## Development

```bash
pytest tests/ -v          # run tests
ruff check src/ tests/    # lint
ruff format src/ tests/   # format
```

CI runs automatically on push via `.github/workflows/ci.yml`.

---

## Design Notes

**Supervisor pattern over linear chain** — routing is state-driven, not hardcoded. The Researcher can be called multiple times (once per subtopic, then again per gap), and the loop exits gracefully once all gaps are covered or `MAX_ITERATIONS` is reached.

**MemorySaver for Streamlit Cloud, SqliteSaver for the API** — the Streamlit UI uses in-memory checkpointing (no file system needed), while the FastAPI backend uses SQLite for persistent, resumable sessions.

**Annotated reducers on shared state** — `sources` and `findings` use `operator.add` (append on each agent update); `messages` uses LangGraph's `add_messages` (deduplication + ordering). All other fields are last-write-wins.

---

## License

MIT
