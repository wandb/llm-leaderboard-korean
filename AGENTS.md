# Project Overview

Horangi (호랑이) is a Korean LLM benchmark evaluation framework built on top of Inspect AI. It evaluates 30+ benchmarks across two categories (GLP: General Language Performance, ALT: Alignment Performance) and publishes results to a W&B/Weave leaderboard.

# Common Commands

```bash
# Install dependencies
uv sync

# Run all benchmarks for a model
uv run python run_eval.py --config <model_config_name>

# Run specific benchmarks only
uv run python run_eval.py --config <model_config_name> --only kmmlu,kobbq,swebench_verified_official_80

# Limit samples (for testing)
uv run python run_eval.py --config <model_config_name> --limit 10

# Resume a failed W&B run
uv run python run_eval.py --config <model_config_name> --resume <wandb_run_id>

# Custom log directory (needed for parallel execution)
uv run python run_eval.py --config <model_config_name> --log-dir /tmp/my_logs

# Resume swebench for multiple models (parallel workers + vLLM GPU allocation)
uv run python resume_swebench.py --workers 4 --dry-run
uv run python resume_swebench.py --workers 4

# Start swebench evaluation server (requires Docker host)
uv run python src/server/swebench_server.py --port 8000
```

Config name is the YAML filename without extension from `configs/models/` (e.g., `gpt-5.4-2026-03-05_xhigh-effort`).

# Initial Setup: W&B (Weights & Biases)

All benchmark results are logged to W&B and its Weave leaderboard. Before running evaluations, users must:

1. **Sign up** at [wandb.ai](https://wandb.ai) and get an API key from [wandb.ai/authorize](https://wandb.ai/authorize).
2. **Create a project** in the W&B dashboard (e.g., `my-korean-llm-bench`). A project is a container for all your evaluation runs.
3. **Set environment variables** in `.env`:
   - `WANDB_API_KEY`: Your API key for authentication.
   - `WANDB_ENTITY`: Your W&B username or team name. This is the account that owns the project.
   - `WANDB_PROJECT`: The project name you created above.

# Required Environment Variables

Set in `.env` file at project root (copy from `.env.sample`):
```
WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
DEEPSEEK_API_KEY, OPENROUTER_API_KEY, HF_TOKEN
HOSTED_VLLM_API_KEY, SWE_API_KEY
SWE_SERVER_URL  # swebench evaluation server URL
```

# Architecture

For benchmark architecture and inheritance patterns, see `docs/README_benchmark_en.md`.

For SWE-bench server setup, see `docs/swebench_server_setup.md`.

Models with a `vllm` section in their config auto-start a local vLLM server via `VLLMServerManager`.

# Adding and Evaluating a New Model (Step-by-Step)

Full workflow for adding a new model and running all benchmarks.

## Step 1: Gather Model Information

Fetch the HuggingFace model page and extract:
- Model ID (e.g., `LGAI-EXAONE/EXAONE-4.5-33B`)
- Parameter count, MoE status (active_params if applicable)
- Context window size
- Whether it is a thinking/reasoning model
- Tool calling support and parser type (hermes, llama3_json, etc.)
- Release date
- **Special requirements**: custom vLLM/transformers fork needed or not

## Step 2: Create Config YAML

Create `configs/models/<ModelName>.yaml`. Reference the template: `configs/models/_template_vllm.yaml` and `configs/models/_template_api.yaml`

## Step 3: Prepare SWE-bench Server

SWE-bench evaluation requires a separate Docker-based server. Follow `docs/swebench_server_setup.md` to start the server and set `SWE_SERVER_URL` in `.env`.

## Step 4: Run Benchmarks

```bash
# Standard models
uv run python run_eval.py --config <ModelName>

# Models with custom forks
.venv/bin/python run_eval.py --config <ModelName>

# Run specific benchmarks only
.venv/bin/python run_eval.py --config <ModelName> --only kmmlu,kobbq

# Limit samples (for quick testing)
.venv/bin/python run_eval.py --config <ModelName> --limit 5
```

## Step 5: Troubleshooting

- **Timeout errors** (`litellm.Timeout`): For thinking models, ensure `timeout: 7200` and `max_retries: 10` are set in config.
- **Architecture not recognized** (`model type "xxx" but Transformers does not recognize`): Custom transformers fork required. Check HuggingFace model page for install instructions.
- **swebench score 0.0000**: Verify `SWE_SERVER_URL` is set in `.env` and server health check passes.
- **GPU OOM**: Increase `tensor_parallel_size` or reduce `max_model_len`.
- **Slow generation**: Expected with thinking models (long reasoning traces). Wait it out for fair evaluation.
