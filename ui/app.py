"""Streamlit UI for the Agentic Research Assistant."""

from __future__ import annotations

import os
import sys
import time
import uuid

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or ui/ directory
# ---------------------------------------------------------------------------
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (after path fix)
# ---------------------------------------------------------------------------
try:
    from config import settings
    from src.graph.state import ResearchState
    from src.graph.workflow import build_graph

    _IMPORTS_OK = True
except ImportError as _err:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(_err)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init_session() -> None:
    defaults: dict = {
        "thread_id": str(uuid.uuid4()),
        "research_done": False,
        "final_state": None,
        "agent_log": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.markdown(
        "This app uses a **LangGraph supervisor pattern** to autonomously research "
        "any topic. It plans subtopics, searches the web with Tavily, synthesizes "
        "findings, identifies gaps, and writes a structured report — all without "
        "human intervention."
    )
    st.divider()

    # Environment variable status
    st.subheader("Configuration Status")
    env_checks = {
        "ANTHROPIC_API_KEY": bool(settings.anthropic_api_key) if _IMPORTS_OK else False,
        "TAVILY_API_KEY": bool(settings.tavily_api_key) if _IMPORTS_OK else False,
        "LANGCHAIN_API_KEY (optional)": bool(settings.langchain_api_key) if _IMPORTS_OK else False,
    }
    for var, ok in env_checks.items():
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} `{var}`")

    st.divider()

    if not _IMPORTS_OK:
        st.error(f"Import error: {_IMPORT_ERROR}")
        st.stop()

    if not settings.anthropic_api_key:
        st.warning("Set `ANTHROPIC_API_KEY` in your `.env` file to use this app.")

    if not settings.tavily_api_key:
        st.warning("Set `TAVILY_API_KEY` in your `.env` file to enable web search.")

    # Show research plan if available
    if st.session_state.final_state:
        plan: list[str] = st.session_state.final_state.get("plan", [])
        if plan:
            st.subheader("Research Plan")
            for i, subtopic in enumerate(plan, 1):
                st.markdown(f"**{i}.** {subtopic}")

    st.divider()
    st.caption("Built with LangGraph · LangChain · Anthropic · Tavily")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🔬 Agentic Research Assistant")
st.markdown(
    "Enter any research topic and the multi-agent system will autonomously plan, "
    "search the web, synthesize findings, and generate a structured report."
)

col1, col2 = st.columns([4, 1])
with col1:
    topic_input = st.text_input(
        "Research Topic",
        placeholder="e.g., The impact of large language models on scientific research",
        label_visibility="collapsed",
    )
with col2:
    run_btn = st.button("🚀 Research", type="primary", use_container_width=True)

# Reset button
if st.session_state.research_done:
    if st.button("🔄 New Research"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.research_done = False
        st.session_state.final_state = None
        st.session_state.agent_log = []
        st.rerun()

# ---------------------------------------------------------------------------
# Run research
# ---------------------------------------------------------------------------
if run_btn and topic_input:
    if not settings.anthropic_api_key:
        st.error("ANTHROPIC_API_KEY is not set. Please add it to your .env file.")
        st.stop()
    if not settings.tavily_api_key:
        st.error("TAVILY_API_KEY is not set. Please add it to your .env file.")
        st.stop()

    graph = build_graph()
    thread_id = st.session_state.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: ResearchState = {
        "topic": topic_input,
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

    agent_display_names = {
        "supervisor": "Supervisor (routing)",
        "planner": "Planner (breaking topic into subtopics)",
        "researcher": "Researcher (searching the web)",
        "synthesizer": "Synthesizer (combining findings)",
        "writer": "Writer (generating report)",
    }

    log_placeholder = st.empty()
    progress_bar = st.progress(0, text="Starting research pipeline...")

    agent_log: list[str] = []
    agent_weights = {"supervisor": 2, "planner": 10, "researcher": 20, "synthesizer": 15, "writer": 25}
    cumulative_progress = 0
    total_weight = 100

    with st.status("Running research pipeline...", expanded=True) as status_box:
        final_state: ResearchState | None = None

        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                display_name = agent_display_names.get(node_name, node_name)
                timestamp = time.strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] **{display_name}** completed"

                # Enrich log with details
                if node_name == "planner" and node_output.get("plan"):
                    n = len(node_output["plan"])
                    log_entry += f" → generated {n} subtopics"
                elif node_name == "researcher":
                    completed = node_output.get("searches_completed", [])
                    if completed:
                        log_entry += f" → searched: *{completed[-1]}*"
                elif node_name == "synthesizer":
                    gaps = node_output.get("gaps", [])
                    log_entry += f" → found {len(gaps)} gap(s)"
                elif node_name == "writer" and node_output.get("report"):
                    wc = len(node_output["report"].split())
                    log_entry += f" → report ~{wc} words"

                agent_log.append(log_entry)
                st.session_state.agent_log = agent_log

                # Update progress
                weight = agent_weights.get(node_name, 5)
                cumulative_progress = min(cumulative_progress + weight, 99)
                progress_bar.progress(cumulative_progress / 100, text=f"Active: {display_name}")

                # Display log in status box
                for entry in agent_log:
                    st.write(entry)

                # Capture final state
                if node_name == "writer" and node_output.get("report"):
                    final_state = graph.get_state(config).values  # type: ignore[assignment]

        status_box.update(label="Research complete!", state="complete", expanded=False)

    progress_bar.progress(1.0, text="Done!")

    if final_state is None:
        # Fallback: fetch state from checkpointer
        final_state = graph.get_state(config).values  # type: ignore[assignment]

    st.session_state.final_state = final_state
    st.session_state.research_done = True
    st.rerun()


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if st.session_state.research_done and st.session_state.final_state:
    final: ResearchState = st.session_state.final_state
    report: str = final.get("report", "")
    sources: list[dict] = final.get("sources", [])
    plan: list[str] = final.get("plan", [])

    st.success(
        f"Research complete! Covered {len(plan)} subtopics, "
        f"collected {len(sources)} sources, "
        f"completed {final.get('iteration', 0)} search iteration(s)."
    )
    st.divider()

    if report:
        st.markdown(report, unsafe_allow_html=False)
    else:
        st.warning("No report was generated. Check logs for errors.")

    # Sources expander
    if sources:
        with st.expander(f"📚 Sources ({len(sources)} total)"):
            seen_urls: set[str] = set()
            idx = 1
            for source in sources:
                url = source.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                subtopic = source.get("subtopic", "General")
                title = source.get("title", "Untitled")
                st.markdown(f"**{idx}. [{title}]({url})**  \n*Subtopic: {subtopic}*")
                if source.get("content"):
                    st.caption(source["content"][:300] + "...")
                st.divider()
                idx += 1

    # Agent activity log
    if st.session_state.agent_log:
        with st.expander("🤖 Agent Activity Log"):
            for entry in st.session_state.agent_log:
                st.markdown(entry)
