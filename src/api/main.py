import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph.state import ResearchState
from src.graph.workflow import build_graph

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the research graph once at startup."""
    global _graph
    _graph = build_graph()
    yield
    # Cleanup (if needed) on shutdown


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic Research Assistant",
    description=(
        "A multi-agent AI system that autonomously plans, researches, synthesizes, "
        "and writes comprehensive research reports on any topic."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="The research topic to investigate.")
    thread_id: str | None = Field(
        default=None,
        description="Optional thread ID for resuming a previous research session.",
    )


class SourceOut(BaseModel):
    title: str
    url: str
    content: str
    subtopic: str


class ResearchResponse(BaseModel):
    thread_id: str
    topic: str
    plan: list[str]
    report: str
    sources: list[SourceOut]
    iterations: int


class StateResponse(BaseModel):
    thread_id: str
    topic: str | None
    plan: list[str]
    searches_completed: list[str]
    synthesized: bool
    has_report: bool
    iteration: int
    source_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Service health check."""
    return {"status": "ok", "service": "agentic-research-assistant"}


@app.post("/research", response_model=ResearchResponse, tags=["Research"])
async def run_research(request: ResearchRequest) -> ResearchResponse:
    """Run the full multi-agent research pipeline on the given topic.

    If a ``thread_id`` is provided, the graph will attempt to resume from the
    last saved checkpoint for that thread. Otherwise a new thread is created.

    Returns the generated report, source list, and research plan once the
    pipeline completes.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Research graph not initialized.")

    thread_id = request.thread_id or str(uuid.uuid4())
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    initial_state: ResearchState = {
        "topic": request.topic,
        "plan": [],
        "searches_completed": [],
        "sources": [],
        "findings": [],
        "synthesized": False,
        "gaps": [],
        "report": "",
        "iteration": 0,
        "messages": [],
    }

    try:
        final_state: ResearchState = await _graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Research pipeline failed: {str(exc)}",
        ) from exc

    sources_out = [
        SourceOut(
            title=s.get("title", ""),
            url=s.get("url", ""),
            content=s.get("content", ""),
            subtopic=s.get("subtopic", ""),
        )
        for s in final_state.get("sources", [])
    ]

    return ResearchResponse(
        thread_id=thread_id,
        topic=final_state.get("topic", request.topic),
        plan=final_state.get("plan", []),
        report=final_state.get("report", ""),
        sources=sources_out,
        iterations=final_state.get("iteration", 0),
    )


@app.get("/research/{thread_id}/state", response_model=StateResponse, tags=["Research"])
async def get_research_state(thread_id: str) -> StateResponse:
    """Retrieve the current (or last known) state for a research thread.

    Useful for polling progress when running the graph asynchronously or
    for inspecting a completed research session.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Research graph not initialized.")

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot = _graph.get_state(config)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Thread '{thread_id}' not found or state unavailable: {str(exc)}",
        ) from exc

    if snapshot is None or snapshot.values is None:
        raise HTTPException(
            status_code=404,
            detail=f"No state found for thread '{thread_id}'.",
        )

    state_values: dict[str, Any] = snapshot.values

    return StateResponse(
        thread_id=thread_id,
        topic=state_values.get("topic"),
        plan=state_values.get("plan", []),
        searches_completed=state_values.get("searches_completed", []),
        synthesized=state_values.get("synthesized", False),
        has_report=bool(state_values.get("report", "")),
        iteration=state_values.get("iteration", 0),
        source_count=len(state_values.get("sources", [])),
    )
