from pandas_gpt.completers.openai import OpenAI

class AzureOpenAI(OpenAI):
    def _create_client(openai, **kw):
        return openai.AzureOpenAI(**kw)
