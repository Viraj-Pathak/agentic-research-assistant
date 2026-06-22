import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.graph.state import ResearchState

_SYSTEM_PROMPT = """You are an expert research planner with deep experience in breaking down complex topics \
into well-structured investigation subtopics. Your role is to analyze a research topic and decompose it into \
3 to 5 focused, non-overlapping subtopics that together provide comprehensive coverage of the subject. \
Each subtopic should be specific enough to guide targeted web searches, yet broad enough to yield meaningful \
results. Prioritize subtopics that address fundamentals, current developments, practical applications, \
challenges, and future directions where applicable. Return ONLY a valid JSON array of strings with no \
additional text, markdown, or explanation."""


def run(state: ResearchState) -> dict:
    """Planner agent: decomposes the research topic into 3-5 focused subtopics."""
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        max_tokens=1024,
    )

    topic = state["topic"]
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research topic: {topic}\n\n"
                "Generate 3-5 specific subtopics for comprehensive research. "
                "Return ONLY a JSON array of strings, e.g.: "
                '["subtopic 1", "subtopic 2", "subtopic 3"]'
            )
        ),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # Extract JSON array from response robustly
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if match:
        raw = match.group(0)
        plan: list[str] = json.loads(raw)
    else:
        # Fallback: attempt direct parse
        plan = json.loads(content)

    # Sanitize: ensure list of non-empty strings, max 5 items
    plan = [str(item).strip() for item in plan if str(item).strip()][:5]

    return {"plan": plan, "searches_completed": [], "iteration": 0}
