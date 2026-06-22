from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.graph.state import ResearchState, Source
from src.tools.search import web_search

_SYSTEM_PROMPT = """You are an expert research analyst skilled at extracting key insights from web search \
results. Given a research subtopic and a set of search results, your task is to synthesize the most \
relevant and accurate information into a concise, factual research finding. Focus on concrete facts, \
statistics, expert opinions, and recent developments. Write in clear, professional prose that can be \
used as a building block for a comprehensive research report. Always attribute important claims to their \
sources where possible. Aim for 2-4 paragraphs that thoroughly cover the subtopic."""


def run(state: ResearchState) -> dict:
    """Researcher agent: picks the next unresearched subtopic, searches the web, and extracts findings."""
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        max_tokens=2048,
    )

    # Identify the next subtopic to research
    searches_completed = set(state.get("searches_completed", []))
    plan = state.get("plan", [])

    # Check for gaps from synthesizer that haven't been searched yet
    gaps = state.get("gaps", [])
    all_topics = list(dict.fromkeys(plan + gaps))  # preserve order, deduplicate
    remaining = [t for t in all_topics if t not in searches_completed]

    if not remaining:
        # Nothing left to research
        return {}

    subtopic = remaining[0]

    # Perform web search
    search_results: list[dict] = web_search.invoke({"query": f"{state['topic']} {subtopic}"})

    # Build sources list
    sources: list[Source] = [
        Source(
            title=r.get("title", "Untitled"),
            url=r.get("url", ""),
            content=r.get("content", ""),
            subtopic=subtopic,
        )
        for r in search_results
        if r.get("url")
    ]

    # Format search results for the LLM
    results_text = "\n\n".join(
        f"**Source {i + 1}: {r.get('title', 'Untitled')}**\n"
        f"URL: {r.get('url', 'N/A')}\n"
        f"Content: {r.get('content', 'No content available')}"
        for i, r in enumerate(search_results)
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research topic: {state['topic']}\n"
                f"Current subtopic: {subtopic}\n\n"
                f"Search results:\n{results_text}\n\n"
                "Synthesize the key findings from these search results into a clear, "
                "factual research summary for this subtopic. Include specific facts, "
                "statistics, and insights. Reference the sources where appropriate."
            )
        ),
    ]

    response = llm.invoke(messages)
    finding = response.content.strip()

    return {
        "sources": sources,
        "findings": [f"## {subtopic}\n\n{finding}"],
        "searches_completed": list(searches_completed) + [subtopic],
        "iteration": state.get("iteration", 0) + 1,
    }
