from langgraph.graph import END

from config import settings
from src.graph.state import ResearchState


def route(state: ResearchState) -> str:
    """Routing function used as the conditional edge from the supervisor node.

    Inspects the current research state and returns the name of the next agent
    to invoke, or END if the workflow is complete.
    """
    # Step 1: if we have no plan, we must plan first
    if not state.get("plan"):
        return "planner"

    # Step 2: if there are subtopics that haven't been searched yet, research them
    searches_completed = set(state.get("searches_completed", []))
    remaining = set(state["plan"]) - searches_completed
    if remaining:
        return "researcher"

    # Step 3: if all subtopics are searched but we haven't synthesized, synthesize
    if not state.get("synthesized"):
        return "synthesizer"

    # Step 4: if there are unsearched knowledge gaps and we haven't exceeded max
    # iterations, loop back to researcher to fill them.
    # We check for *unsearched* gaps specifically to avoid an infinite loop when
    # the researcher has already covered all gaps but the gaps list is non-empty.
    gaps = state.get("gaps", [])
    iteration = state.get("iteration", 0)
    unsearched_gaps = [g for g in gaps if g not in searches_completed]
    if unsearched_gaps and iteration < settings.max_iterations:
        return "researcher"

    # Step 5: if synthesis is done but no report has been written, write it
    if not state.get("report"):
        return "writer"

    # Step 6: report exists — we're done
    return END
