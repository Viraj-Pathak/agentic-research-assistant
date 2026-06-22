from datetime import date

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.graph.state import ResearchState, Source

_SYSTEM_PROMPT = """You are an expert technical writer who produces clear, authoritative, and well-structured \
research reports. Your reports are known for their logical flow, precise language, and proper citation of \
sources. Given a set of research findings and source materials, write a comprehensive markdown report that \
would satisfy both a general audience and subject matter experts. Structure the report with a compelling \
executive summary, clearly delineated sections for each major subtopic, and a conclusion that synthesizes \
key takeaways. Every significant claim should be supported by inline citations using [Source N] notation, \
with a full References section at the end. Use markdown formatting effectively: headers, bullet points, \
bold text for key terms, and horizontal rules where appropriate."""


def _format_sources_section(sources: list[Source]) -> str:
    """Build a deduplicated References section from sources."""
    seen_urls: set[str] = set()
    refs: list[str] = []
    idx = 1

    for source in sources:
        url = source.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            title = source.get("title", "Untitled")
            refs.append(f"{idx}. [{title}]({url})")
            idx += 1

    if not refs:
        return "## References\n\n*No sources available.*"

    return "## References\n\n" + "\n".join(refs)


def run(state: ResearchState) -> dict:
    """Writer agent: generates the final structured markdown research report with citations."""
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )

    topic = state["topic"]
    findings = state.get("findings", [])
    sources = state.get("sources", [])
    plan = state.get("plan", [])

    findings_text = "\n\n---\n\n".join(findings) if findings else "No findings available."

    # Build source list for the LLM to reference
    source_list = "\n".join(
        f"{i + 1}. {s.get('title', 'Untitled')} — {s.get('url', 'N/A')}"
        for i, s in enumerate(sources)
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research topic: {topic}\n"
                f"Subtopics researched: {', '.join(plan)}\n"
                f"Today's date: {date.today().isoformat()}\n\n"
                f"Available sources:\n{source_list}\n\n"
                f"Research findings:\n{findings_text}\n\n"
                "Write a comprehensive, well-structured markdown research report. Requirements:\n"
                "1. Start with a # title and a brief executive summary\n"
                "2. Include a ## section for each major subtopic with detailed analysis\n"
                "3. Use inline citations like [Source 1], [Source 2], etc., referencing the source list above\n"
                "4. End with a ## Key Takeaways section with 4-6 bullet points\n"
                "5. End with a ## Conclusion section\n"
                "6. Do NOT include the References section — it will be appended automatically\n"
                "7. Use bold, bullet points, and subheadings to improve readability"
            )
        ),
    ]

    response = llm.invoke(messages)
    report_body = response.content.strip()

    # Append the references section
    references = _format_sources_section(sources)
    full_report = f"{report_body}\n\n---\n\n{references}"

    return {"report": full_report}
