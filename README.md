# Agentic Research Assistant

![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-blueviolet?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=chainlink)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude%20Sonnet-orange?logo=anthropic)
![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41+-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)

A **production-quality multi-agent AI system** that autonomously researches any topic end-to-end. Given a topic, the system plans subtopics, searches the web, synthesizes findings, identifies knowledge gaps, and produces a structured markdown report — all without human intervention.

Built with **LangGraph's supervisor pattern**, this project demonstrates advanced agentic AI architecture: stateful multi-agent orchestration, conditional routing, SQLite checkpointing for resumability, web search integration, and both a REST API and Streamlit UI.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Research Topic                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   SUPERVISOR    │ ◄─────────────────────┐
              │  (Router Node)  │                       │
              └────────┬────────┘                       │
                       │ conditional edges              │
         ┌─────────────┼──────────────┬─────────────┐  │
         ▼             ▼              ▼             ▼  │
   ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌────────┐ │
   │ PLANNER  │ │ RESEARCHER │ │SYNTHESIZR│ │ WRITER │ │
   │          │ │            │ │          │ │        │ │
   │ Breaks   │ │ Tavily web │ │ Combines │ │ Final  │ │
   │ topic    │ │ search per │ │ findings │ │ report │ │
   │ into     │ │ subtopic   │ │ IDs gaps │ │ with   │ │
   │ subtopics│ │            │ │          │ │cites   │ │
   └────┬─────┘ └─────┬──────┘ └────┬─────┘ └───┬────┘ │
        │             │             │            │      │
        └─────────────┴─────────────┴────────────┘      │
                       all return to SUPERVISOR ────────┘
```

### Agent Descriptions

| Agent | Role | Key Behavior |
|-------|------|--------------|
| **Supervisor** | Router node | Inspects state, routes via conditional edges — no LLM call needed |
| **Planner** | Topic decomposer | Uses Claude to break the topic into 3-5 focused, non-overlapping subtopics |
| **Researcher** | Web searcher | Picks one unsearched subtopic, calls Tavily, extracts key insights via Claude |
| **Synthesizer** | Gap analyst | Combines all findings, identifies knowledge gaps for follow-up research |
| **Writer** | Report author | Produces a structured markdown report with inline citations and References section |

### Routing Logic

```
No plan → PLANNER
Plan exists + unsearched subtopics → RESEARCHER
All searched + not synthesized → SYNTHESIZER
Synthesized + gaps exist + iterations < max → RESEARCHER (gap fill)
Synthesized + no gaps (or max iterations reached) + no report → WRITER
Report exists → END
```

### State Schema

The shared `ResearchState` uses LangGraph's reducer annotations for safe concurrent updates:

- `sources` and `findings` use `operator.add` (append semantics across agent updates)
- `messages` uses LangChain's `add_messages` reducer (deduplication + ordering)
- All other fields are last-write-wins

---

## Project Structure

```
agentic-research-assistant/
├── .github/workflows/ci.yml      # GitHub Actions: lint + test
├── src/
│   ├── agents/
│   │   ├── supervisor.py         # Stateless routing function
│   │   ├── planner.py            # Topic decomposition agent
│   │   ├── researcher.py         # Web search + insight extraction
│   │   ├── synthesizer.py        # Findings synthesis + gap analysis
│   │   └── writer.py             # Markdown report generation
│   ├── tools/
│   │   └── search.py             # Tavily web search LangChain tool
│   ├── graph/
│   │   ├── state.py              # ResearchState TypedDict + reducers
│   │   └── workflow.py           # LangGraph StateGraph construction
│   └── api/
│       └── main.py               # FastAPI REST API
├── ui/app.py                     # Streamlit demo UI
├── tests/
│   ├── test_state.py             # State reducer and routing unit tests
│   └── test_workflow.py          # Graph compilation and routing tests
├── config.py                     # Pydantic Settings configuration
├── requirements.txt
├── Dockerfile                    # API container
├── Dockerfile.ui                 # Streamlit UI container
├── docker-compose.yml            # Orchestrates API + UI
├── .ruff.toml                    # Ruff linter configuration
└── .env.example                  # Environment variable template
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/)
- [Tavily API key](https://app.tavily.com/) (free tier available)
- (Optional) [LangSmith API key](https://smith.langchain.com/) for tracing

### 1. Clone and install

```bash
git clone https://github.com/Viraj-Pathak/agentic-research-assistant.git
cd agentic-research-assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
LANGCHAIN_API_KEY=ls__...      # optional, for LangSmith tracing
LANGCHAIN_TRACING_V2=false     # set to true to enable tracing
LANGCHAIN_PROJECT=agentic-research-assistant
```

### 3. Run the API

```bash
uvicorn src.api.main:app --reload
```

API docs available at: http://localhost:8000/docs

### 4. Run the Streamlit UI

```bash
streamlit run ui/app.py
```

UI available at: http://localhost:8501

---

## API Reference

### POST /research

Run the full multi-agent research pipeline on a topic.

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "The impact of large language models on scientific research"}'
```

**Response:**
```json
{
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "topic": "The impact of large language models on scientific research",
  "plan": ["LLM applications in drug discovery", "AI-assisted literature review", "..."],
  "report": "# The Impact of Large Language Models on Scientific Research\n\n...",
  "sources": [
    {
      "title": "GPT-4 in Drug Discovery",
      "url": "https://example.com/article",
      "content": "...",
      "subtopic": "LLM applications in drug discovery"
    }
  ],
  "iterations": 3
}
```

### Resume a research session

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "quantum computing", "thread_id": "your-thread-id"}'
```

### GET /research/{thread_id}/state

Inspect the current state of a research thread.

```bash
curl http://localhost:8000/research/550e8400-e29b-41d4-a716-446655440000/state
```

**Response:**
```json
{
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "topic": "quantum computing",
  "plan": ["quantum hardware", "quantum algorithms", "..."],
  "searches_completed": ["quantum hardware"],
  "synthesized": false,
  "has_report": false,
  "iteration": 1,
  "source_count": 5
}
```

---

## Docker

### Run with Docker Compose (API + UI)

```bash
# Copy and fill in your .env file
cp .env.example .env

# Start both services
docker-compose up --build

# API: http://localhost:8000
# UI:  http://localhost:8501
```

### Run API only

```bash
docker build -t research-assistant-api .
docker run -p 8000:8000 --env-file .env research-assistant-api
```

---

## LangSmith Tracing

Enable full agent trace visualization in [LangSmith](https://smith.langchain.com/):

```env
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-research-assistant
```

This captures every LLM call, tool invocation, and state transition in the multi-agent graph, making it easy to debug and optimize agent behavior.

---

## Development

### Run tests

```bash
pytest tests/ -v
```

### Lint

```bash
ruff check src/ tests/
```

### Format

```bash
ruff format src/ tests/
```

---

## Configuration

All settings are managed via `config.py` using `pydantic-settings`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | `` | Claude API key (required) |
| `TAVILY_API_KEY` | `` | Tavily search API key (required) |
| `LANGCHAIN_API_KEY` | `` | LangSmith tracing key (optional) |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `MODEL_NAME` | `claude-sonnet-4-6` | Anthropic model to use |
| `MAX_ITERATIONS` | `3` | Max research loop iterations |
| `MAX_SEARCH_RESULTS` | `5` | Tavily results per query |

---

## Key Design Decisions

**Why supervisor pattern over linear chain?** The supervisor gives us dynamic routing based on state — the researcher can be called multiple times (once per subtopic), and the synthesizer can trigger additional research loops to fill gaps, all without hardcoding a fixed pipeline.

**Why SQLite checkpointing?** LangGraph's `SqliteSaver` persists state after every node execution. This means research sessions survive process restarts, can be resumed by thread ID, and support fault-tolerant long-running research tasks.

**Why separate Dockerfile for UI?** The API and UI have different runtime concerns. Separating them allows independent scaling and lets the Streamlit UI be deployed on services like Streamlit Cloud while the API runs on a dedicated server.

---

## License

MIT
