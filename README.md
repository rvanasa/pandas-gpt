# `pandas-gpt` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rvanasa/pandas-gpt/blob/main/notebooks/pandas_gpt_demo.ipynb)

> ### Power up your data science workflow with ChatGPT.

---

`pandas-gpt` is a Python library for doing almost anything with a [pandas](https://pandas.pydata.org/) DataFrame using ChatGPT prompts. 

## Installation

```bash
pip install pandas-gpt
```

Set the `OPENAI_API_KEY` environment variable to your [OpenAI API key](https://platform.openai.com/account/api-keys), or use the following code snippet:

```python
import openai
openai.api_key = '<API Key>'
```

## Examples

Setup and usage examples are available in this **[Google Colab notebook](https://colab.research.google.com/github/rvanasa/pandas-gpt/blob/main/notebooks/pandas_gpt_demo.ipynb)**.

```python
import pandas as pd
import pandas_gpt

df = pd.DataFrame('https://gist.githubusercontent.com/bluecoconut/9ce2135aafb5c6ab2dc1d60ac595646e/raw/c93c3500a1f7fae469cba716f09358cfddea6343/sales_demo_with_pii_and_all_states.csv')

# Data transformation
df = df.ask('drop purchases from Laurenchester, NY')
df = df.ask('add a new Category column with values "cheap", "regular", or "expensive"')

# Queries
weekday = df.ask('which day of the week had the largest number of orders?')
top_10 = df.ask('what are the top 10 most popular products, as a table')

# Plotting
df.ask('plot monthly and hourly sales')
top_10.ask('horizontal bar plot with pastel colors')

# Allow changes to original dataset
df.ask('do something interesting', mutable=True)

# Show source code before running
df.ask('convert prices from USD to GBP', verbose=True)
```

## Other Hosts

If you want to use a different API host such as [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service):

```python
import openai
openai.api_type = 'azure'
openai.api_base = '<Endpoint>'
openai.api_version = '<Version>'
openai.api_key = '<API Key>'

import pandas_gpt
# pandas_gpt.model = '<Model>' # Default is 'gpt-3.5-turbo'
pandas_gpt.completion_config = {
  'engine': '<Engine>',
  # 'deployment_id': '<Deployment ID>',
}
```

## Alternatives

- [GitHub Copilot](https://github.com/features/copilot): General-purpose code completion (paid subscription)
- [Sketch](https://github.com/approximatelabs/sketch): AI-powered data summarization and code suggestions (works without an API key)

## Disclaimer

Please note that the [limitations](https://github.com/openai/gpt-3/blob/master/model-card.md#limitations) of ChatGPT also apply to this library. I would recommend using `pandas-gpt` in a sandboxed environment such as [Google Colab](https://colab.research.google.com), [Kaggle](https://www.kaggle.com/docs/notebooks), or [GitPod](https://www.gitpod.io/).
