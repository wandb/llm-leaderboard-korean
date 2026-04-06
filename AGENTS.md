# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Horangi (호랑이) is a Korean LLM benchmark evaluation framework built on top of Inspect AI. It evaluates 30+ benchmarks across two categories (GLP: General Language Performance, ALT: Alignment Performance) and publishes results to a W&B/Weave leaderboard.

## Common Commands

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

## Required Environment Variables

Set in `.env` file at project root:
```
WANDB_ENTITY, WANDB_PROJECT, WANDB_API_KEY
OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
OPENROUTER_API_KEY, HF_TOKEN
SWE_SERVER_URL  # swebench evaluation server URL
```

## Architecture

### Evaluation Pipeline

```
configs/models/<name>.yaml          # Model config (client, params, benchmark overrides)
        ↓
run_eval.py                         # Orchestrator: loads config, iterates benchmarks
        ↓
src/benchmarks/horangi.py           # @task functions call create_benchmark()
        ↓
src/core/factory.py                 # Factory: loads data, creates Inspect AI Task
  ├── loads BenchmarkConfig from src/benchmarks/<name>.py
  ├── loads data from Weave refs or JSONL
  ├── if `base` exists: inherits solver/scorer from inspect_evals
  └── otherwise: uses custom solver/scorer from src/solvers/ and src/scorers/
        ↓
inspect_ai.eval()                   # Runs evaluation (inspect-wandb auto-logs to Weave)
        ↓
src/core/models_leaderboard.py      # Aggregates scores into GLP/ALT categories
src/core/weave_leaderboard.py       # Publishes Weave leaderboard
```

### Key Pattern: Benchmark Inheritance

Many benchmarks inherit from `inspect_evals` (official implementations). Horangi replaces only the dataset (Korean-translated) and solver while keeping the original scorer. This is done via the `base` field in BenchmarkConfig:

```python
# src/benchmarks/ko_hellaswag.py
CONFIG = BenchmarkConfig(
    base="inspect_evals.hellaswag.hellaswag",  # Inherit scorer from official
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoHellaSwag_mini:...",
    solver="korean_multiple_choice",
)
```

### Model Config Structure

```yaml
# configs/models/<name>.yaml
model:
  name: gpt-5.4-2026-03-05       # Model identifier
  client: openai | litellm        # API client
  provider: openai | anthropic | openrouter | hosted_vllm
  api_key_env: OPENAI_API_KEY
  params:                          # Global generation params
    max_tokens: 128000
    temperature: 1
    reasoning_effort: xhigh

benchmarks:                        # Per-benchmark overrides
  swebench_verified_official_80:
    max_tokens: 128000
    temperature: 0
```

`config_loader.py` merges base_config.yaml → model params → benchmark overrides.

### SWE-bench Evaluation

SWE-bench uses a separate server architecture:
1. **Solver** (`swebench_patch_solver.py`): LLM generates unified diff patch
2. **Scorer** (`swebench_server_scorer.py`): Client-side patch normalization pipeline (extract_diff → repair_patch → fix_split_headers → extract_minimal_patch), then submits to server
3. **Server** (`swebench_server.py`): Receives patch, runs swebench official harness in Docker, returns resolved/unresolved

Server setup guide: `docs/swebench_server_setup.md`

### Parallel Execution

When running multiple models in parallel, each process needs isolated directories to avoid `inspect_ai` log_dir lock conflicts and wandb state corruption:
- `--log-dir` per process (for inspect_ai eval logs)
- `WANDB_DIR` per process (for wandb metadata)
- `CUDA_VISIBLE_DEVICES` per vLLM model (GPU isolation)

### vLLM Local Models

Models with a `vllm` section in their config auto-start a local vLLM server via `VLLMServerManager`. These models require GPU access and cannot run in parallel with each other unless given dedicated GPUs.

## Adding a New Benchmark

1. Create `src/benchmarks/<name>.py` with a `CONFIG = BenchmarkConfig(...)` 
2. Add `@task` function in `src/benchmarks/horangi.py`
3. Register in `run_eval.py`'s `ALL_BENCHMARKS` list
4. If custom scoring needed, add scorer in `src/scorers/` and register in `factory.py`

## Adding a New Model

Create `configs/models/<name>.yaml` following the structure above. The config name (filename without `.yaml`) is used as the `--config` argument.
