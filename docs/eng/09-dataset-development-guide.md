# Dataset Development Guide

Welcome! This guide explains how to add a new dataset to the Haerae Evaluation Toolkit. It covers design philosophy, the registry-based architecture and facade usage, core interfaces, and step-by-step instructions with best practices.

## Design Philosophy

We aim for a flexible, extensible evaluation stack. Two key principles underpin our design:

1) Registry Pattern (Extensibility)
   - Each dataset, model backend, scaling method, and evaluator is registered via a decorator.
   - Users add new components by implementing a small class and annotating it with a register_* decorator.
   - This minimizes invasive code changes and keeps integration simple.

2) Facade Pattern (Usability)
   - The Evaluator and PipelineRunner act as facades over multiple subsystems (dataset loading, inference, scaling, and evaluation).
   - Contributors focus on implementing a single component correctly; the facade orchestrates it within the pipeline.

These principles let different teams contribute independently with low coupling and high cohesion.

## Dataset Interface at a Glance

- Base class: `llm_eval.datasets.base.BaseDataset`
- Required method:
  - `load() -> List[Dict[str, Any]]`: returns a standardized list of samples
    - Each sample must contain at least:
      - `input`: string given to the model
      - `reference`: expected answer string
    - Optional keys (common): `_subset_name`, `options`, `metadata`
- Optional methods:
  - `get_raw_samples()`: return the raw HF dataset or data frame
  - `info() -> Dict[str, Any]`: metadata about the dataset

## Registry Usage

- File: `llm_eval/datasets/__init__.py`
- Decorator: `@register_dataset("your_key")`
- The registry maps a string key to the dataset class. The loader is then accessible via:

```python
from llm_eval.datasets import load_datasets
loader = load_datasets(name="your_key", subset="your_subset", split="test")
samples = loader.load()
```

## Standard Output Schema

Return a list of dicts like:

```python
[
  {
    "input": "...",               # the full prompt presented to the model
    "reference": "...",           # gold answer string
    "_subset_name": "subset_a",   # optional: used to segment metrics
    "options": ["(A)", "(B)", ...],  # optional: MCQA
    "metadata": { ... }            # optional: anything extra
  },
  ...
]
```

Keep `input` self-contained (include instructions or a base prompt template if necessary).

## Prompt Templates

- Many datasets benefit from a `base_prompt_template`: a format string used when building `input`.
- Provide sensible defaults to minimize config burden. Allow override via `dataset_params`.

Example:
```python
default_template = "Answer succinctly and put final answer after 'Answer:'.\n\n{question}"
formatted = default_template.format(question=raw)
```

## Subsets and Splits

- `subset` can be None | str | List[str]. Support all three when reasonable.
- When pulling from Hugging Face Hub, some datasets expose only certain splits (e.g., test only). Implement light fallbacks, e.g.: try `self.split`, then `test`, `validation`, `train`.
- Always populate `_subset_name` when subsets exist so the analysis module can compute per-subset metrics.

## Allowed Evaluations

- If your dataset requires a specific evaluation method, return it via `info()`:

```python
return { "evaluation_only": ["string_match", "math_match"] }
```

- If unrestricted, return `None` or omit the key. The PipelineRunner validates this.

## Step-by-Step: Adding a New Dataset

1) Create a new file: `llm_eval/datasets/my_dataset.py`
2) Implement a class inheriting `BaseDataset`, add `@register_dataset("my_key")`
3) Implement `load()` to return standardized samples
4) Wire it into registry imports: add `from .my_dataset import MyDataset` to `llm_eval/datasets/__init__.py`
5) (Optional) Add an example config under `examples/`
6) (Optional) Add docs and update contribution guides if necessary

## Example: Minimal Loader (Local Files)

If your data is CSV/Parquet/XLSX, use the built-in `generic_file` dataset (see `dataset_loader.py`). Or implement your own:

```python
@register_dataset("my_local")
class MyLocalDataset(BaseDataset):
    def load(self):
        # read file, map columns
        return [
            {"input": row["question"], "reference": row["answer"]}
            for row in rows
        ]
```

## Example: Hugging Face Dataset

```python
from datasets import load_dataset

@register_dataset("my_hf")
class MyHFDataset(BaseDataset):
    def load(self):
        ds = load_dataset("owner/name", split=self.split)
        out = []
        for it in ds:
            q = it.get("question", "")
            a = str(it.get("answer", ""))
            inp = self.base_prompt_template.format(question=q) if self.base_prompt_template else q
            out.append({"input": inp, "reference": a, "_subset_name": "default"})
        return out
```

## Best Practices

- Minimal, readable code. Prefer small helpers instead of large functions.
- Avoid leaking non-essential fields; keep schema tight.
- Provide deterministic ordering (e.g., stable iteration) when possible.
- Make normalization choices explicit in the prompt or evaluator (e.g., tell models to end with "Answer:").
- Reuse existing evaluators when possible: `string_match`, `math_match`, `partial_match`, `llm_judge`, etc.

## Testing

- We provide `llm_eval/test/test_datasets.py` to sanity-check registry and `load()` shape.
- For networked datasets (HF Hub), tests are tolerant to transient errors.

## Where to Ask Questions

- See `docs/eng/07-contribution-guide.md` for contribution workflow.
- Open a GitHub issue or start a discussion if you need clarifications.
