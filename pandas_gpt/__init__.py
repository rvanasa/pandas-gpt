import pandas as pd

_ask_cache = {}

class Ask:
  def __init__(self, *, verbose=False):
    import os
    self.verbose = verbose

  @staticmethod
  def _fill_template(template, **kw):
    import re
    from textwrap import dedent
    result = dedent(template.lstrip('\n').rstrip())
    for k, v in kw.items():
      result = result.replace(f'{{{k}}}', v)
    m = re.match(r'\{[a-zA-Z0-9_]*\}', result)
    if m:
      raise Exception(f'Expected variable: {m.group(0)}')
    return result

  def _get_prompt(self, goal, arg):
    import io
    buf = io.StringIO()
    arg.info(buf=buf)
    arg_summary = buf.getvalue()
    arg_name = 'df' if isinstance(arg, pd.DataFrame) else 'data'

    return self._fill_template('''
      Write a Python function `process({arg_name})` which takes the following input value:

      {arg_name} = {arg}

      This is the function's purpose: {goal}

      Write the function in a Python code block with all necessary imports and no example usage:
    ''', arg_name=arg_name, arg=arg_summary.strip(), goal=goal.strip())

  def _run_prompt(self, prompt):
    import openai
    cache = _ask_cache
    completion = cache.get(prompt) or openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages=[
        # dict(role='system', content=''),
        dict(role='user', content=prompt),
      ]
    )
    cache[prompt] = completion
    return completion['choices'][0]['message']['content']

  def _extract_code_block(self, text):
    import re
    pattern = r'```(\s*(py|python)\s*\n)?([\s\S]*?)```'
    m = re.search(pattern, text)
    if not m:
      return text
    return m.group(3)

  def _eval(self, source, *args):
    _args_ = args
    scope = dict(_args_=args)
    exec(self._fill_template('''
      {source}
      _result_ = process(*_args_)
    ''', source=source), scope)
    return scope['_result_']

  def _code(self, goal, arg):
    prompt = self._get_prompt(goal, arg)
    result = self._run_prompt(prompt)
    if self.verbose:
      print()
      print(result)
    return self._extract_code_block(result)

  def code(self, *args):
    print(self._code(*args))

  def __call__(self, goal, *args):
    source = self._code(goal, *args)
    return self._eval(source, *args)


@pd.api.extensions.register_dataframe_accessor('ask')
class AskAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        pass

    def _ask(self, **kw):
      return Ask(**kw)

    def _data(self):
      return self._obj.copy() # TODO: possibly `deep=False`

    def __call__(self, goal, *args, **kw):
        ask = self._ask(**kw)
        data = self._data()
        return ask(goal, data, *args)

    def code(self, goal, *args, **kw):
        ask = self._ask(**kw)
        data = self._data()
        return ask.code(goal, data, *args)
