from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from .base import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self, endpoint: str, timeout: float = 600):
        self.client = AsyncOpenAI(
            base_url=f"{endpoint}/v1",
            api_key="-",
            organization="-",
            project="-",
            timeout=timeout,
        )

    async def chat(self, messages: list[dict], **kwargs) -> str:
        resp: ChatCompletion = await self.client.chat.completions.create(
            model=kwargs.get("model"), messages=messages, extra_body=kwargs
        )
        return resp.choices[0].message.content
