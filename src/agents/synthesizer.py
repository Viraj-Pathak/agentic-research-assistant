import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.graph.state import ResearchState

_SYSTEM_PROMPT = """You are an expert research synthesizer with a talent for identifying patterns, \
contradictions, and gaps across multiple information sources. Your role is to critically evaluate a \
set of research findings, identify overarching themes, note any contradictions or inconsistencies, \
and—most importantly—identify specific knowledge gaps that warrant further investigation. Be rigorous \
and intellectually honest: if important aspects of the topic are not covered by the current research, \
call them out explicitly. Return your response as a JSON object with two keys: "summary" (a brief \
2-3 sentence synthesis of the overall findings) and "gaps" (an array of strings, each describing a \
specific knowledge gap or area needing further research; return an empty array if coverage is complete)."""


def run(state: ResearchState) -> dict:
    """Synthesizer agent: combines findings, identifies gaps, and flags areas for further research."""
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        max_tokens=1024,
    )

    findings = state.get("findings", [])
    plan = state.get("plan", [])
    topic = state["topic"]

    findings_text = "\n\n---\n\n".join(findings) if findings else "No findings available."

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research topic: {topic}\n"
                f"Research subtopics covered: {', '.join(plan)}\n\n"
                f"Research findings:\n{findings_text}\n\n"
                "Analyze these findings. Return a JSON object with:\n"
                '- "summary": 2-3 sentence overall synthesis\n'
                '- "gaps": array of specific knowledge gaps (empty array [] if coverage is complete)\n\n'
                "Be specific about gaps — name the exact topic or angle that needs more research."
            )
        ),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # Extract JSON from response robustly
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
    else:
        parsed = json.loads(content)

    gaps: list[str] = [str(g).strip() for g in parsed.get("gaps", []) if str(g).strip()]

    return {
        "synthesized": True,
        "gaps": gaps,
    }
