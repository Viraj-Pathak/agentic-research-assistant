import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, add_messages


class Source(TypedDict):
    title: str
    url: str
    content: str
    subtopic: str


class ResearchState(TypedDict):
    topic: str
    plan: list[str]
    searches_completed: list[str]
    sources: Annotated[list[Source], operator.add]
    findings: Annotated[list[str], operator.add]
    synthesized: bool
    gaps: list[str]
    report: str
    iteration: int
    messages: Annotated[list[AnyMessage], add_messages]
