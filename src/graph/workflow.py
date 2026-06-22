from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.agents import planner, researcher, synthesizer, writer
from src.agents.supervisor import route
from src.graph.state import ResearchState


def build_graph(db_path: str = "checkpoints.db") -> StateGraph:
    """Construct and compile the multi-agent research graph.

    The graph uses a supervisor pattern: every agent node returns to the
    supervisor, which uses conditional edges to route to the appropriate
    next agent based on the current research state.

    Args:
        db_path: Path to the SQLite database used for LangGraph checkpointing.

    Returns:
        A compiled LangGraph StateGraph ready to invoke or stream.
    """
    graph = StateGraph(ResearchState)

    # Register all nodes
    graph.add_node("supervisor", lambda s: {})  # pure router — no state mutation
    graph.add_node("planner", planner.run)
    graph.add_node("researcher", researcher.run)
    graph.add_node("synthesizer", synthesizer.run)
    graph.add_node("writer", writer.run)

    # Supervisor is the entry point
    graph.set_entry_point("supervisor")

    # Conditional routing FROM supervisor
    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "planner": "planner",
            "researcher": "researcher",
            "synthesizer": "synthesizer",
            "writer": "writer",
            END: END,
        },
    )

    # Every agent returns to supervisor after completing its work
    for node in ["planner", "researcher", "synthesizer", "writer"]:
        graph.add_edge(node, "supervisor")

    # SQLite-backed checkpointing for state persistence and resumability
    memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    return graph.compile(checkpointer=memory)
