# Adding a New Model

> How to evaluate a new model with Horangi.
> For installation and environment setup, see the [root README](../README_en.md#-installation) first.

---

## 📋 Step 1: Collect model information

Check the model's [HuggingFace model card](https://huggingface.co/models) or official API docs for:

| Item | Example | Note |
|---|---|---|
| Model ID | `LGAI-EXAONE/EXAONE-4.0-32B`, `gpt-4o-2024-11-20` | Use the exact name issued by the provider for API models |
| Parameter count / MoE | 32B, plus `active_params` if MoE | Used for `size_category` |
| Context window | 32k, 128k, ... | Sets `max_model_len` / `context_window` |
| Reasoning (thinking) model? | Uses `enable_thinking`, `reasoning_effort`, etc. | If yes, bump timeout and token limits significantly |
| Tool calling support | Parser type (`hermes`, `llama3_json`, ...) | Required for BFCL |
| Release date | `2025-07-15` | Shown on the leaderboard |
| Custom fork required | e.g., specific `transformers` / `vllm` branch | See troubleshooting in README_swebench_en |

---

## 🎯 Step 2: Pick a template

| Situation | Template | File |
|---|---|---|
| **API models** (OpenAI / Anthropic / Google / xAI / OpenRouter, etc.) | API template | [`configs/models/_template_api.yaml`](../configs/models/_template_api.yaml) |
| **Local models** served via vLLM | vLLM template | [`configs/models/_template_vllm.yaml`](../configs/models/_template_vllm.yaml) |

> With the vLLM template, `run_eval.py` **automatically starts and stops** the vLLM server. You do not need to run it separately.

```bash
# API model
cp configs/models/_template_api.yaml configs/models/my-model.yaml

# Local vLLM model
cp configs/models/_template_vllm.yaml configs/models/my-model.yaml
```

The config filename (without extension) becomes the `--config` argument. For example, `configs/models/my-model.yaml` → `--config my-model`.

---

## 🔧 Step 3: Edit the config

### Shared blocks

```yaml
wandb:
  run_name: "My-Model-v1"          # Name shown in the W&B run and leaderboard

metadata:
  release_date: "2026-04-01"
  size_category: "Large (30B<)"    # "Small (<10B)" | "Medium (10B-30B)" | "Large (30B<)"
  model_size: 32000000000          # Total parameter count
  active_params: 32000000000       # Active params for MoE; same as model_size for dense
  context_window: 32768
```

### API model

```yaml
model:
  name: claude-opus-4-5-20251101   # Exact name used by the provider
  client: litellm                  # openai | litellm
  provider: anthropic              # anthropic | openai | google | xai | openrouter | ...
  api_key_env: ANTHROPIC_API_KEY   # Must exist in .env

  params:
    max_tokens: 4096
    temperature: 0.6
    top_p: 0.95
    timeout: 7200                  # Set high for reasoning models
    max_retries: 2
    max_connections: 30            # Concurrent requests

    # For reasoning models, add ONE of:
    # reasoning_effort: high       # OpenAI reasoning (low/medium/high/xhigh)
    # effort: high                 # Anthropic
    # extra_body:                  # OpenRouter thinking
    #   reasoning:
    #     enabled: true
```

| Provider | `client` | `api_key_env` example |
|---|---|---|
| OpenAI (direct) | `openai` | `OPENAI_API_KEY` |
| Anthropic | `litellm` | `ANTHROPIC_API_KEY` |
| Google Gemini | `litellm` | `GEMINI_API_KEY` |
| xAI Grok | `litellm` | `XAI_API_KEY` |
| OpenRouter | `litellm` | `OPENROUTER_API_KEY` |

### vLLM model

```yaml
vllm:
  model_path: "LGAI-EXAONE/EXAONE-4.0-32B"   # HuggingFace ID or local path
  tensor_parallel_size: 2                    # Number of GPUs
  port: 8000
  host: "0.0.0.0"
  max_model_len: 32768
  trust_remote_code: true
  served_model_name: "exaone-4.0-32b"        # Must match model.name

  # If the model supports tool calling:
  enable_auto_tool_choice: true
  tool_call_parser: "hermes"                 # hermes | llama3_json | ...

  # For thinking models:
  reasoning_parser: "deepseek_r1"

model:
  name: exaone-4.0-32b
  client: litellm
  provider: hosted_vllm
  api_key_env: HOSTED_VLLM_API_KEY

  params:
    max_tokens: 16384
    temperature: 0.6
    top_p: 0.95
    timeout: 7200
    max_retries: 2
    max_connections: 30

    # For thinking models:
    # extra_body:
    #   chat_template_kwargs:
    #     enable_thinking: true
```

### Benchmark-specific overrides (shared)

Use these to change parameters only for specific benchmarks.

```yaml
benchmarks:
  bfcl:
    use_native_tools: false              # Prefer text-based for OSS models; API models can often use true
  swebench_verified_official_80:
    max_tokens: 64000                    # Code generation needs long outputs
  ko_arc_agi:
    extra_body:
      repetition_penalty: 1.05           # Suppress repetition
```

---

## ▶️ Step 4: Run

```bash
# Smoke test with a single benchmark at low sample count
uv run python run_eval.py --config my-model --only kmmlu --limit 5

# If OK, run everything
uv run python run_eval.py --config my-model

# Re-run only some benchmarks
uv run python run_eval.py --config my-model --only bfcl,kobbq

# Resume an interrupted run
uv run python run_eval.py --config my-model --resume <wandb_run_id>
```

SWE-bench requires a separate server. See the [SWE-bench guide](./README_swebench_en.md).

---

## ✅ Checklist

Before submitting to the leaderboard, confirm:

- [ ] Smoke test (`--limit 5 --only kmmlu`) succeeds and prints a Weave URL
- [ ] `metadata.release_date`, `size_category`, `model_size` are correct
- [ ] For API models, `api_key_env` actually exists in `.env`
- [ ] For vLLM models, `served_model_name` matches `model.name`
- [ ] For reasoning models, `timeout` ≥ 7200 and `max_retries` ≥ 5
- [ ] `use_native_tools` in BFCL is set appropriately for the model

---

## 🛠 Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `litellm.Timeout` | Reasoning model. Set `params.timeout: 7200`, `max_retries: 10` |
| `model type "xxx" but Transformers does not recognize` | Custom `transformers` fork required. Check the HuggingFace model card for install instructions |
| `swebench score 0.0000` | `SWE_SERVER_URL` missing in `.env`, or server health check failing |
| GPU OOM | Increase `tensor_parallel_size` or decrease `max_model_len` |
| Thinking model is slow | Expected — long reasoning traces are part of a fair evaluation. Just wait |
| `WANDB_API_KEY`-related errors | Confirm all three W&B env vars (`API_KEY` / `ENTITY` / `PROJECT`) are set in `.env` |

---

## 📚 Related docs

- [Adding a new benchmark](./README_benchmark_en.md)
- [SWE-bench server setup](./README_swebench_en.md)
- [Weave results analysis](./README_weave_en.md)
