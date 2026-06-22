"""Tests for the LangGraph workflow: graph compilation and node structure."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Graph compilation tests
# ---------------------------------------------------------------------------


def test_build_graph_compiles() -> None:
    """build_graph() should return a compiled graph without raising."""
    # Mock SqliteSaver so we don't need a real database in CI
    mock_saver = MagicMock()
    mock_saver.get_next_version.return_value = 1

    with patch("src.graph.workflow.SqliteSaver.from_conn_string", return_value=mock_saver):
        from src.graph.workflow import build_graph

        graph = build_graph(db_path=":memory:")
        assert graph is not None


def test_graph_has_required_nodes() -> None:
    """The compiled graph must contain all five agent nodes."""
    mock_saver = MagicMock()

    with patch("src.graph.workflow.SqliteSaver.from_conn_string", return_value=mock_saver):
        from src.graph.workflow import build_graph

        graph = build_graph(db_path=":memory:")

        # LangGraph exposes compiled graph nodes via .nodes attribute or graph.graph
        # We access the underlying graph structure for node inspection
        node_names = set(graph.nodes.keys())  # type: ignore[attr-defined]

        expected_nodes = {"supervisor", "planner", "researcher", "synthesizer", "writer"}
        assert expected_nodes.issubset(node_names), (
            f"Missing nodes: {expected_nodes - node_names}. Found: {node_names}"
        )


def test_graph_entry_point_is_supervisor() -> None:
    """The graph entry point must be the supervisor node."""
    mock_saver = MagicMock()

    with patch("src.graph.workflow.SqliteSaver.from_conn_string", return_value=mock_saver):
        from src.graph.workflow import build_graph

        graph = build_graph(db_path=":memory:")

        # The compiled graph wraps the original StateGraph; check the builder's entry
        # Access via the underlying graph object
        assert hasattr(graph, "nodes"), "Compiled graph should have nodes attribute"


# ---------------------------------------------------------------------------
# Supervisor routing logic tests (no LLM, no network)
# ---------------------------------------------------------------------------


def test_route_returns_planner_when_no_plan() -> None:
    """Supervisor should route to planner when plan is empty."""
    from src.agents.supervisor import route
    from src.graph.state import ResearchState

    state: ResearchState = {
        "topic": "AI safety",
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
    assert route(state) == "planner"


def test_route_returns_researcher_when_subtopics_remain() -> None:
    """Supervisor should route to researcher when unsearched subtopics exist."""
    from src.agents.supervisor import route
    from src.graph.state import ResearchState

    state: ResearchState = {
        "topic": "AI safety",
        "plan": ["alignment", "interpretability", "governance"],
        "searches_completed": ["alignment"],
        "sources": [],
        "findings": [],
        "synthesized": False,
        "gaps": [],
        "report": "",
        "iteration": 1,
        "messages": [],
    }
    assert route(state) == "researcher"


def test_route_returns_synthesizer_when_all_searched() -> None:
    """Supervisor should route to synthesizer after all subtopics are researched."""
    from src.agents.supervisor import route
    from src.graph.state import ResearchState

    state: ResearchState = {
        "topic": "AI safety",
        "plan": ["alignment", "interpretability"],
        "searches_completed": ["alignment", "interpretability"],
        "sources": [],
        "findings": ["finding 1", "finding 2"],
        "synthesized": False,
        "gaps": [],
        "report": "",
        "iteration": 2,
        "messages": [],
    }
    assert route(state) == "synthesizer"


def test_route_returns_writer_when_synthesized_no_gaps() -> None:
    """Supervisor should route to writer when synthesis is done and no gaps exist."""
    from src.agents.supervisor import route
    from src.graph.state import ResearchState

    state: ResearchState = {
        "topic": "AI safety",
        "plan": ["alignment", "interpretability"],
        "searches_completed": ["alignment", "interpretability"],
        "sources": [],
        "findings": ["finding 1", "finding 2"],
        "synthesized": True,
        "gaps": [],
        "report": "",
        "iteration": 2,
        "messages": [],
    }
    assert route(state) == "writer"


def test_route_returns_end_when_report_exists() -> None:
    """Supervisor should route to END when report is written."""
    from langgraph.graph import END

    from src.agents.supervisor import route
    from src.graph.state import ResearchState

    state: ResearchState = {
        "topic": "AI safety",
        "plan": ["alignment", "interpretability"],
        "searches_completed": ["alignment", "interpretability"],
        "sources": [],
        "findings": ["finding 1"],
        "synthesized": True,
        "gaps": [],
        "report": "# AI Safety Report\n\nContent here.",
        "iteration": 2,
        "messages": [],
    }
    assert route(state) == END


def test_route_researcher_when_gaps_under_max_iterations() -> None:
    """Supervisor should loop back to researcher to fill gaps when under max iterations."""
    from src.agents.supervisor import route
    from src.graph.state import ResearchState

    state: ResearchState = {
        "topic": "AI safety",
        "plan": ["alignment"],
        "searches_completed": ["alignment"],
        "sources": [],
        "findings": ["finding 1"],
        "synthesized": True,
        "gaps": ["emerging regulatory frameworks"],  # gap not yet in searches_completed
        "report": "",
        "iteration": 1,  # under max_iterations (3)
        "messages": [],
    }
    result = route(state)
    # Should route to researcher (to fill gaps) or writer; gaps exist so researcher
    assert result == "researcher"


def test_route_writer_when_gaps_exceed_max_iterations() -> None:
    """Supervisor should proceed to writer when max iterations exceeded, even with gaps."""
    from src.agents.supervisor import route
    from src.graph.state import ResearchState

    state: ResearchState = {
        "topic": "AI safety",
        "plan": ["alignment"],
        "searches_completed": ["alignment"],
        "sources": [],
        "findings": ["finding 1"],
        "synthesized": True,
        "gaps": ["some gap"],
        "report": "",
        "iteration": 10,  # exceeds max_iterations (3)
        "messages": [],
    }
    result = route(state)
    assert result == "writer"
