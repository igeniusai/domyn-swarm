from .openai import OpenAIClient
from .base import LLMClient


def create_llm_client(provider: str, endpoint: str, timeout: float = 600) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(endpoint=endpoint, timeout=timeout)
    raise ValueError(f"Unsupported provider: {provider}")
