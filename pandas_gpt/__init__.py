from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd

__all__ = [
    "verbose",
    "mutable",
    "completer",
    "system_prompt",
    "template",
    "Ask",
    "AskAccessor",
    "OpenAI",
    "LiteLLM",
]


@dataclass
class OpenAI:
    config: dict[str, Any]
    cache: dict[str, str] = {}

    def __init__(self, model: str, **config):
        self.config = config
        self.config["model"] = model

    def __call__(self, prompt: str) -> str:
        completion = self.cache.get(prompt) or self.run_completion_function(
            **self.config,
            messages=[
                dict(
                    role="system",
                    content=system_prompt,
                ),
                dict(role="user", content=prompt),
            ],
        )
        self.cache[prompt] = completion
        return completion.choices[0].message.content

    def run_completion_function(self, **kw):
        try:
            import openai
        except ImportError:
            raise Exception(
                "The package `openai` could not be found. You can fix this error by running `pip install pandas-gpt[openai]` or passing a custom `completer` argument."
            )
        return openai.chat.completions.create(**kw)


class LiteLLM(OpenAI):
    def run_completion_function(self, **kw):
        try:
            import litellm
        except ImportError:
            raise Exception(
                "The package `litellm` could not be found. You can fix this error by running `pip install pandas-gpt[litellm]` or passing a custom `completer` argument."
            )
        return litellm.completion(**kw)


# Override with `pandas_gpt.verbose = True`
verbose: bool = False
# Override with `pandas_gpt.mutable = True`
mutable: bool = False
# Override with `pandas_gpt.completer = ...`
completer: Callable[[str], str] = OpenAI("gpt-3.5-turbo")
system_prompt = "Write the function in a Python code block with all necessary imports and no example usage."
# Override with `pandas_gpt.template = ...`
template = """
Write a Python function `process({arg_name})` which takes the following input value:

{arg_name} = {arg}

This is the function's purpose: {goal}
"""


@dataclass
class Ask:
    verbose: bool
    mutable: bool
    completer: Callable[[str], str]

    def __init__(
        self,
        verbose: bool | None = None,
        mutable: bool | None = None,
        completer: Callable[[str], str] = None,
    ):
        self.verbose = verbose if verbose is not None else globals()["verbose"]
        self.mutable = mutable if mutable is not None else globals()["mutable"]
        self.completer = completer if completer is not None else globals()["completer"]

    @staticmethod
    def _fill_template(template: str, **kw) -> str:
        import re
        from textwrap import dedent

        result = dedent(template.lstrip("\n").rstrip())
        for k, v in kw.items():
            result = result.replace(f"{{{k}}}", v)
        m = re.match(r"\{[a-zA-Z0-9_]*\}", result)
        if m:
            raise Exception(f"Expected variable: {m.group(0)}")
        return result

    def _get_prompt(self, goal: str, arg: Any) -> str:
        if isinstance(arg, pd.DataFrame) or isinstance(arg, pd.Series):
            import io

            buf = io.StringIO()
            arg.info(buf=buf)
            arg_summary = buf.getvalue()
        else:
            arg_summary = repr(arg)
        arg_name = (
            "df"
            if isinstance(arg, pd.DataFrame)
            else "index" if isinstance(arg, pd.Index) else "data"
        )

        return self._fill_template(
            template, arg_name=arg_name, arg=arg_summary.strip(), goal=goal.strip()
        )

    def _run_prompt(self, prompt: str) -> str:
        return self.completer(prompt)

    def _extract_code_block(self, text: str) -> str:
        import re

        pattern = r"```(\s*(py|python)\s*\n)?([\s\S]*?)```"
        m = re.search(pattern, text)
        if not m:
            return text
        return m.group(3)

    def _eval(self, source: str, *args):
        scope = dict(_args_=args)
        exec(
            self._fill_template(
                """
                {source}
                _result_ = process(*_args_)
                """,
                source=source,
            ),
            scope,
        )
        return scope["_result_"]

    def _code(self, goal: str, arg: Any):
        prompt = self._get_prompt(goal, arg)
        result = self._run_prompt(prompt)
        if self.verbose:
            print()
            print(result)
        return self._extract_code_block(result)

    def code(self, *args):
        print(self._code(*args))

    def prompt(self, *args):
        print(self._get_prompt(*args))

    def __call__(self, goal: str, *args):
        source = self._code(goal, *args)
        return self._eval(source, *args)


@pd.api.extensions.register_dataframe_accessor("ask")
@pd.api.extensions.register_series_accessor("ask")
@pd.api.extensions.register_index_accessor("ask")
class AskAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        pass

    def _ask(self, **kw):
        return Ask(**kw)

    def _data(self, **kw):
        if not mutable and not kw.get("mutable") and hasattr(self._obj, "copy"):
            return self._obj.copy()
        return self._obj

    def __call__(self, goal: str, *args, **kw):
        ask = self._ask(**kw)
        data = self._data(**kw)
        return ask(goal, data, *args)

    def code(self, goal: str, *args, **kw):
        ask = self._ask(**kw)
        data = self._data(**kw)
        return ask.code(goal, data, *args)

    def prompt(self, goal: str, *args, **kw):
        ask = self._ask(**kw)
        data = self._data(**kw)
        return ask.prompt(goal, data, *args)
