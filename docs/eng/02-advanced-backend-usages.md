# Advanced Backend Usages

In this tutorial, we explore advanced methods for leveraging the Haerae Evaluation Toolkit using LiteLLM and OpenAI-compatible APIs.

## 1. LiteLLM Integration

### 1.1 What is LiteLLM?

* **LiteLLM** is a library that provides a unified interface to multiple LLM providers (OpenAI, Azure, Anthropic, Claude, Cohere, and more).
* It allows seamless switching between over 30 LLM vendors via a single API.
* Haerae supports LiteLLM to streamline access to various models.

### 1.2 Setting Up and Using LiteLLM

```python
from llm_eval.evaluator import Evaluator

# 1) Create an Evaluator instance
evaluator = Evaluator()

# 2) Evaluate using LiteLLM backend
results = evaluator.run(
    model="litellm",                       # Use LiteLLM backend
    model_params={
        "model_name": "gpt-4",            # OpenAI model name
        # or another provider, e.g. "anthropic/claude-3-opus-20240229"
        "api_key": "your-api-key-here",
        "max_tokens": 512,
        "temperature": 0.7
    },
    dataset="haerae_bench",
    subset=["csat_math"],
    split="test",
    evaluation_method="string_match"
)

print(results)
df = results.to_dataframe()
print(df)
```

## 2. OpenAI-Compatible API Usage

### 2.1 What is an OpenAI-Compatible API?

* Many LLM services offer endpoints compatible with OpenAI’s API interface.
* You can switch to these services without changing your existing OpenAI client code.
* Haerae provides support for connecting to such OpenAI-compatible endpoints.

### 2.2 Using an OpenAI-Compatible API

```python
from llm_eval.evaluator import Evaluator

# 1) Create an Evaluator instance
evaluator = Evaluator()

# 2) Evaluate using OpenAI-compatible endpoint
results = evaluator.run(
    model="openai",                        # Use OpenAI backend
    model_params={
        "model_name": "text-davinci-003",  # Model name
        "api_key": "your-api-key-here",
        "api_base": "https://your-custom-endpoint.com/v1",  # Custom endpoint
        "max_tokens": 256,
        "temperature": 0.2
    },
    dataset="haerae_bench",
    subset=["csat_eng"],
    split="test",
    evaluation_method="string_match"
)

print(results)
df = results.to_dataframe()
print(df)
```

### 2.3 Using a Self-Hosted OpenAI-Compatible Server

When hosting your own LLM server (e.g., vLLM, FastChat, text-generation-inference), connect as follows:

```python
# Connect to a self-hosted OpenAI-compatible server
results = evaluator.run(
    model="openai",
    model_params={
        "model_name": "local-llama-2-13b",   # Local model name
        "api_base": "http://localhost:8000/v1",  # Local server URL
        "api_key": "not-needed",              # API key may be optional
        "max_tokens": 512
    },
    dataset="haerae_bench",
    subset=["csat_kor"],
    split="test",
    evaluation_method="string_match"
)
```
