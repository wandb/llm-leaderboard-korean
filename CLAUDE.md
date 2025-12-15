# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the HAE-RAE Evaluation Toolkit - a comprehensive framework for evaluating Korean Large Language Models (LLMs). The codebase uses a modular, registry-based architecture where datasets, models, and evaluators are pluggable components registered globally for dynamic loading.

## Key Architecture

The system follows a pipeline architecture: Config → Dataset Loading → Model Inference → Evaluation → Post-Processing → W&B/Weave Logging

Main components:
- **Registry Pattern**: All components (datasets, models, evaluators) use decorators like `@register_dataset()` for automatic registration
- **Config-Driven**: YAML files control the entire pipeline behavior
- **Singleton W&B**: Single W&B run shared across multiple dataset evaluations via `WandbConfigSingleton`

## Essential Commands

### Setup and Installation
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install with optional vLLM support
uv sync --extra vllm

# Install dev/test tools
uv sync --extra dev,test
```

### Running Evaluations
```bash
# Single model evaluation
uv run python run_eval.py --config gpt-4o-2024-11-20

# Single dataset evaluation
uv run python run_eval.py --dataset kmmlu

# Multiple models by provider
python experiment.py --provider openai
python experiment.py --provider anthropic

# Custom dataset list
uv run python run_eval.py --config claude-sonnet-4-5-20250929 --dataset mt_bench,kmmlu,squad_kor_v1
```

### Testing
```bash
# Run all tests
pytest llm_eval/test/ --cache-clear

# Specific test suites
pytest llm_eval/test/test_datasets.py
pytest llm_eval/test/test_evaluations.py
pytest llm_eval/test/test_scaling.py

# Run single test
pytest llm_eval/test/test_datasets.py::test_dataset_loading[kmmlu]
```

### Code Quality
```bash
# Run pre-commit hooks
pre-commit run --all-files

# Auto-format code (autopep8, line length 80)
autopep8 --in-place --max-line-length=80 <file>

# Sort imports
isort <file>
```

## Core File Structure

Key files and their purposes:
- `run_eval.py` - Main CLI entry point for evaluations
- `experiment.py` - Batch evaluation runner for multiple models
- `configs/base_config.yaml` - Master configuration with dataset settings and model overrides
- `configs/*.yaml` - Individual model configurations (54 models)
- `llm_eval/runner.py` - PipelineRunner class that orchestrates the evaluation pipeline
- `llm_eval/evaluator.py` - High-level Evaluator CLI interface
- `llm_eval/wandb_singleton.py` - Manages shared W&B run across datasets
- `llm_eval/datasets/` - Dataset loaders (37 datasets), each registered with `@register_dataset()`
- `llm_eval/models/` - Model backends (13 implementations), each registered with `@register_model()`
- `llm_eval/evaluation/` - Evaluators (18 scorers), each registered with `@register_evaluator()`

## Configuration System

The system uses a two-level configuration hierarchy:

1. **base_config.yaml**: Contains global settings, dataset configurations, and evaluation methods
2. **Model configs** (e.g., `gpt-4o-2024-11-20.yaml`): Model-specific settings that can override base config

Dataset configuration in base_config.yaml:
```yaml
dataset_name:
  split: train/test/validation
  subset: [list_of_subsets]  # Optional
  params:
    num_samples: N  # Number of samples to evaluate
    limit: M  # Number of batches (optional)
  evaluation:
    method: evaluator_name
    params: {...}  # Evaluator-specific parameters
  model_params: {...}  # Optional model parameter overrides
```

## Adding New Components

### New Dataset
1. Create class in `llm_eval/datasets/` extending `BaseDataset`
2. Implement `load()` method returning `List[Dict[str, Any]]` with keys: `instruction`, `reference_answer`, etc.
3. Add `@register_dataset("name")` decorator
4. Add configuration to `base_config.yaml`
5. Add test in `test_datasets.py`

### New Evaluator
1. Create class in `llm_eval/evaluation/` extending `BaseEvaluator`
2. Implement `score(predictions, references, **kwargs)` method
3. Add `@register_evaluator("name")` decorator
4. Reference in dataset config's `evaluation.method`
5. Add test in `test_evaluations.py`

### New Model Backend
1. Create class in `llm_eval/models/` extending `BaseModel`
2. Implement `generate_batch(prompts, **kwargs)` method
3. Add `@register_model("name")` decorator
4. Create model config YAML in `configs/`
5. Set required API keys in `.env`

## Environment Variables

Required API keys in `.env`:
- `OPENAI_API_KEY` - For OpenAI models
- `ANTHROPIC_API_KEY` - For Claude models
- `WANDB_API_KEY` - For W&B logging
- `GOOGLE_API_KEY` - For Gemini models

See `.env.example` for complete list of 50+ optional provider keys.

## W&B Integration

The system uses a singleton pattern for W&B runs:
- One run encompasses all dataset evaluations for a model
- Results are logged to tables and artifacts
- Leaderboard tables are automatically generated
- Configuration: `wandb.params` in base_config.yaml sets entity/project

## Common Debugging

```python
# Debug single evaluation
from llm_eval.runner import PipelineRunner, PipelineConfig

config = PipelineConfig(
    dataset_name='kmmlu',
    subset=['Chemistry'],
    model_backend_name='openai',
    model_backend_params={'model_name': 'gpt-4o'}
)
runner = PipelineRunner(config)
result = runner.run()
print(result.metrics)
```

## Performance Considerations

- Dataset loading uses caching when possible
- API calls respect `inference_interval` to avoid rate limits
- Batch sizes are configurable per model/dataset
- vLLM backend supports auto server management for local models
- Test mode (`testmode: true`) reduces sample counts for quick testing

## Current Development Status

- Active branch: horangi4-dev
- Python 3.10+ required
- Package manager: uv (ultra-fast Python package manager)
- CI/CD: GitHub Actions running tests on Python 3.10 and 3.12
- Pre-commit hooks enforce code quality (flake8, autopep8, isort, mypy)