from dataclasses import dataclass
import os
from typing import Any

__all__ = ["OpenAI"]

default_system_prompt = "Write the function in a Python code block with all necessary imports and no example usage."


@dataclass
class AzureOpenAI:
    completion_config: dict[str, Any]
    client_config: dict[str, Any]
    system_prompt: str
    _cache: dict[str, str]
    _client: Any

    def __init__(self, model: str, **client_config):
        self.completion_config = {"model": model}
        self.client_config = client_config
        self.system_prompt = default_system_prompt
        self._cache = {}
        self._client = None

    def __call__(self, prompt: str) -> str:
        completion = self._cache.get(prompt) or self.run_completion_function(
            **self.completion_config,
            messages=[
                dict(role="system", content=self.system_prompt),
                dict(role="user", content=prompt),
            ],
        )
        self._cache[prompt] = completion
        return completion.choices[0].message.content

    def run_completion_function(self, **kw):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise Exception(
                    "The package `openai` could not be found. You can fix this error by running `pip install pandas-gpt[openai]` or passing a custom `completer` argument."
                )
            client_config = dict(self.client_config)
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", openai.api_key)
            if api_key is not None and "api_key" not in client_config:
                client_config["api_key"] = api_key
            self._client = openai.AzureOpenAI(**client_config)
        return self._client.chat.completions.create(**kw)
