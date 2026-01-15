# Benchmark Addition Guide

> This document explains how to add new benchmarks.
> For installation, usage, and model configuration, see [Root README](../README.md).

---

## 🎯 Adding a New Benchmark

### Step 1: Create Config File

```python
# src/benchmarks/my_benchmark.py
from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    # Data source (required)
    data_type="weave",  # "weave" or "jsonl"
    data_source="weave:///entity/project/object/DatasetName:latest",
    
    # Field mapping
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",  # MCQA only
    },
    
    # Evaluation settings
    answer_format="index_0",
    solver="multiple_choice",
    scorer="choice",
    system_message="System prompt",
)
```

### Step 2: Register

```python
# src/benchmarks/__init__.py
from benchmarks.my_benchmark import CONFIG as my_benchmark

BENCHMARKS = {
    ...
    "my_benchmark": my_benchmark,
}

BENCHMARK_DESCRIPTIONS = {
    ...
    "my_benchmark": "Benchmark description",
}
```

### Step 3: Add Task Function

```python
# src/benchmarks/horangi.py
@task
def my_benchmark(shuffle: bool = False, limit: int | None = None) -> Task:
    """My Benchmark"""
    return create_benchmark(name="my_benchmark", shuffle=shuffle, limit=limit)
```

### Step 4: Test

```bash
uv run horangi my_benchmark --model openai/gpt-4o -T limit=5
```

---

## 📋 BenchmarkConfig Field Reference

### Required Fields

| Field | Description |
|------|------|
| `data_type` | `"weave"` or `"jsonl"` |
| `data_source` | Weave URI or JSONL filename (relative to `src/data/`) |

### Main Optional Fields

| Field | Default | Description |
|------|--------|------|
| `field_mapping` | `{}` | Dataset → Sample field mapping |
| `solver` | `"multiple_choice"` | Solver |
| `scorer` | `"choice"` | Scorer |
| `answer_format` | `"index_0"` | Answer conversion method |
| `system_message` | `None` | System prompt |

### `answer_format` Options

| Value | Description | Example |
|----|------|------|
| `identity` | No conversion | `"answer"` → `"answer"` |
| `index_0` | 0-indexed → A/B/C | `0` → `"A"` |
| `index_1` | 1-indexed → A/B/C | `1` → `"A"` |
| `text` | Text → Index | `"apple"` → `"A"` |
| `letter` | Keep as-is | `"A"` → `"A"` |
| `to_string` | Convert to string | `123` → `"123"` |
| `boolean` | True/False → A/B | `True` → `"A"` |

### Solver / Scorer Options

| Solver | Purpose |
|--------|------|
| `multiple_choice` | MCQA |
| `generate` | Free generation |
| `bfcl_solver` | Tool calling (Native) |
| `bfcl_text_solver` | Tool calling (Text-based) |
| `mtbench_solver` | MT-Bench multi-turn conversation |
| `swebench_patch_solver` | SWE-bench |

| Scorer | Purpose |
|--------|------|
| `choice` | MCQA accuracy |
| `match` | Exact match |
| `match_numeric` | Numeric match |
| `model_graded_qa` | LLM grading |
| `hle_grader` | HLE dedicated grading |
| `grid_match` | Grid match (ARC-AGI) |
| `macro_f1` | Macro F1 |
| `kobbq_scorer` | KoBBQ bias |
| `hallulens_qa_scorer` | HalluLens QA |
| `refusal_scorer` | HalluLens refusal response evaluation |
| `bfcl_scorer` | BFCL function calling |
| `mtbench_scorer` | MT-Bench evaluation |
| `swebench_server_scorer` | SWE-bench server grading |

---

## 🔧 Adding Custom Scorers

### Step 1: Create Scorer File

```python
# src/scorers/my_scorer.py
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, CORRECT, INCORRECT
from inspect_ai.solver import TaskState

@scorer(metrics=[accuracy()])
def my_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        expected = target.text
        is_correct = response.strip() == expected.strip()
        
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=response[:100],
        )
    return score
```

### Step 2: Register

```python
# src/scorers/__init__.py
from scorers.my_scorer import my_scorer

__all__ = [..., "my_scorer"]
```

---

## 📝 Checklist

When adding a new benchmark:

- [ ] Create config file in `src/benchmarks/`
- [ ] Register in `src/benchmarks/__init__.py`
- [ ] Add `@task` function in `src/benchmarks/horangi.py`
- [ ] Run tests

---

## 🔗 References

- [Inspect AI Docs](https://inspect.ai-safety-institute.org.uk/)
- [inspect_evals GitHub](https://github.com/UKGovernmentBEIS/inspect_evals)
