from pandas_gpt.completers.openai import OpenAI

__all__ = ["LiteLLM"]


class LiteLLM(OpenAI):
    def run_completion_function(self, **kw):
        try:
            import litellm
        except ImportError:
            raise Exception(
                "The package `litellm` could not be found. You can fix this error by running `pip install pandas-gpt[litellm]` or passing a custom `completer` argument."
            )
        return litellm.completion(**kw)
