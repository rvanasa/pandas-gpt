from dataclasses import dataclass

from pandas_gpt.completers.openai import OpenAI

__all__ = ["OpenRouter"]


@dataclass
class OpenRouter(OpenAI):
    def __init__(self, model: str, **config):
        super().__init__(self, model, base_url="https://openrouter.ai/api/v1", **config)
