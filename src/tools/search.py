from langchain_core.tools import tool
from tavily import TavilyClient

from config import settings

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    """Lazily initialize the Tavily client so import doesn't fail without API key."""
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


@tool
def web_search(query: str) -> list[dict]:
    """Search the web for information on a topic. Returns list of results with title, url, content."""
    client = _get_client()
    resp = client.search(
        query=query,
        max_results=settings.max_search_results,
        search_depth="advanced",
    )
    return [
        {
            "title": r["title"],
            "url": r["url"],
            "content": r.get("content", "")[:1500],
        }
        for r in resp.get("results", [])
    ]
