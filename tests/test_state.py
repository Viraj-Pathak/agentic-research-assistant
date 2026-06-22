"""Tests for ResearchState reducers and type correctness."""

import operator

import pytest

from src.graph.state import ResearchState, Source


# ---------------------------------------------------------------------------
# Source TypedDict
# ---------------------------------------------------------------------------


def test_source_typeddict_fields() -> None:
    """Source TypedDict should accept all required fields."""
    source: Source = {
        "title": "Test Article",
        "url": "https://example.com/article",
        "content": "This is the article content.",
        "subtopic": "machine learning basics",
    }
    assert source["title"] == "Test Article"
    assert source["url"] == "https://example.com/article"
    assert source["content"] == "This is the article content."
    assert source["subtopic"] == "machine learning basics"


def test_source_partial_construction() -> None:
    """Source fields are individually accessible."""
    source: Source = {
        "title": "",
        "url": "https://example.com",
        "content": "",
        "subtopic": "test",
    }
    assert isinstance(source["url"], str)


# ---------------------------------------------------------------------------
# ResearchState reducers
# ---------------------------------------------------------------------------


def test_sources_reducer_appends() -> None:
    """The Annotated[list[Source], operator.add] reducer should concatenate lists."""
    existing: list[Source] = [
        {"title": "A", "url": "https://a.com", "content": "a", "subtopic": "sub1"}
    ]
    new_sources: list[Source] = [
        {"title": "B", "url": "https://b.com", "content": "b", "subtopic": "sub2"}
    ]
    merged = operator.add(existing, new_sources)
    assert len(merged) == 2
    assert merged[0]["title"] == "A"
    assert merged[1]["title"] == "B"


def test_findings_reducer_appends() -> None:
    """The Annotated[list[str], operator.add] reducer on findings should concatenate."""
    existing_findings = ["## Subtopic 1\n\nFinding one."]
    new_findings = ["## Subtopic 2\n\nFinding two."]
    merged = operator.add(existing_findings, new_findings)
    assert len(merged) == 2
    assert "Finding one." in merged[0]
    assert "Finding two." in merged[1]


def test_sources_reducer_empty_lists() -> None:
    """Adding empty lists should yield empty list."""
    assert operator.add([], []) == []


def test_sources_reducer_left_empty() -> None:
    """Adding sources to empty list."""
    source: Source = {"title": "X", "url": "https://x.com", "content": "x", "subtopic": "s"}
    result = operator.add([], [source])
    assert len(result) == 1
    assert result[0]["title"] == "X"


# ---------------------------------------------------------------------------
# ResearchState construction
# ---------------------------------------------------------------------------


def test_research_state_initial_values() -> None:
    """ResearchState should hold all expected fields with correct types."""
    state: ResearchState = {
        "topic": "quantum computing",
        "plan": ["basics", "hardware", "algorithms"],
        "searches_completed": ["basics"],
        "sources": [],
        "findings": ["## basics\n\nSome content."],
        "synthesized": False,
        "gaps": [],
        "report": "",
        "iteration": 1,
        "messages": [],
    }
    assert state["topic"] == "quantum computing"
    assert len(state["plan"]) == 3
    assert state["searches_completed"] == ["basics"]
    assert state["synthesized"] is False
    assert state["iteration"] == 1
    assert state["report"] == ""


def test_research_state_boolean_fields() -> None:
    """synthesized field should support True/False correctly."""
    state: ResearchState = {
        "topic": "test",
        "plan": [],
        "searches_completed": [],
        "sources": [],
        "findings": [],
        "synthesized": True,
        "gaps": ["gap 1"],
        "report": "# Report\n\nContent.",
        "iteration": 2,
        "messages": [],
    }
    assert state["synthesized"] is True
    assert len(state["gaps"]) == 1
    assert state["report"].startswith("# Report")


def test_research_state_plan_is_list_of_strings() -> None:
    """Plan should be a list of string subtopics."""
    plan = ["neural networks", "transformers", "reinforcement learning"]
    state: ResearchState = {
        "topic": "deep learning",
        "plan": plan,
        "searches_completed": [],
        "sources": [],
        "findings": [],
        "synthesized": False,
        "gaps": [],
        "report": "",
        "iteration": 0,
        "messages": [],
    }
    assert all(isinstance(s, str) for s in state["plan"])
    assert state["plan"][1] == "transformers"


# ---------------------------------------------------------------------------
# Supervisor routing logic (unit-level, no LLM calls)
# ---------------------------------------------------------------------------


def test_remaining_subtopics_calculation() -> None:
    """set(plan) - set(searches_completed) should give remaining subtopics."""
    plan = ["sub1", "sub2", "sub3"]
    searches_completed = ["sub1"]
    remaining = set(plan) - set(searches_completed)
    assert "sub2" in remaining
    assert "sub3" in remaining
    assert "sub1" not in remaining
    assert len(remaining) == 2


def test_all_subtopics_completed() -> None:
    """When all subtopics are searched, remaining should be empty."""
    plan = ["sub1", "sub2"]
    searches_completed = ["sub1", "sub2"]
    remaining = set(plan) - set(searches_completed)
    assert len(remaining) == 0
