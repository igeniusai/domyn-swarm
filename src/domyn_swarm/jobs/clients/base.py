from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str: ...
