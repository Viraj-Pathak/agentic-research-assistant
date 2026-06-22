from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    tavily_api_key: str = ""
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "false"
    langchain_project: str = "agentic-research-assistant"
    model_name: str = "claude-sonnet-4-6"
    max_iterations: int = 3
    max_search_results: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
