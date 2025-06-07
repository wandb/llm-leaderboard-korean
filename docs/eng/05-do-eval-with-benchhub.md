# BenchHub Dataset Usage Guide

HRET (HaeRae Evaluation Toolkit) supports **BenchHub**, a unified benchmark repository that lets you easily assemble evaluation datasets according to your criteria. In this guide, we'll cover how to load BenchHub in HRET, perform customized evaluations using filtering, and generate citation reports for academic papers.

## Table of Contents

1. [What is BenchHub?](#what-is-benchhub)
2. [Using BenchHub in HRET](#using-benchhub-in-hret)

   * 2.1 [Basic Usage](#basic-usage)
   * 2.2 [Customizing with Filters](#customizing-with-filters)
3. [Inspecting Results & Metadata](#inspecting-results--metadata)
4. [Generating Citation Reports](#generating-citation-reports)

---

## 1. What is BenchHub?

* A **unified benchmark repository** that aggregates datasets scattered across multiple sources, solving management challenges.
* **Key Concepts:**

  * **Integration & Auto-Classification:** Datasets are automatically categorized by skill, subject, target, and more.
  * **Custom Evaluation Sets:** Users can extract only the data relevant to their evaluation objectives via simple filters.
  * **Dynamic Scalability:** New benchmarks are automatically formatted and classified upon addition.

## 2. Using BenchHub in HRET

Specify `dataset="benchhub"` in `Evaluator.run()`, and supply your filter criteria under `dataset_params`.

### 2.1 Basic Usage

Load Korean test samples without additional filters:

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.run(
    model="openai",
    model_params={
        "api_base": "http://0.0.0.0:8000/v1/chat/completions",
        "model_name": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "batch_size": 3
    },
    dataset="benchhub",
    dataset_params={"language": "ko"},  # Korean data
    evaluation_method="string_match",
    split="test"
)
print(results)
```

### 2.2 Customizing with Filters

Building on `simple_tutorial.ipynb`, combine multiple filters to assemble your evaluation set:

```python
evaluator = Evaluator()
results = evaluator.run(
    model="openai",
    model_params={
        "api_base": "http://0.0.0.0:8000/v1/chat/completions",
        "model_name": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "batch_size": 3
    },
    dataset="benchhub",
    dataset_params={
        "language": "ko",                       # Language
        "split": "test",                       # Data split
        "benchmark_names": ["KMMLU"],          # Benchmark IDs
        "problem_types": ["MCQA"],             # Question formats
        "task_types": ["knowledge"],           # Task categories
        "target_types": ["General"],           # Target audience
        "subject_types": ["tech/electrical/Electrical Eng"]  # Subjects
    },
    evaluation_method="string_match"
)
print(results)
```

| Filter Parameter  | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `language`        | 'ko' or 'en'                                         |
| `benchmark_names` | List of original benchmark identifiers               |
| `problem_types`   | Problem formats (e.g., MCQA, Open-ended)             |
| `task_types`      | Ability categories (knowledge, reasoning, alignment) |
| `target_types`    | Cultural orientation (General, Cultural)             |
| `subject_types`   | Subject taxonomy (e.g., science/math, tech/IT)       |

## 3. Inspecting Results & Metadata

* View summary metrics with `results.metrics`
* Convert to a DataFrame via `results.to_dataframe()`
* Retrieve applied filters and config via `results.info()`

```python
print(results.metrics)
df = results.to_dataframe()
print(df.head())
info = results.info()
print(info)
```

## 4. Generating Citation Reports

After evaluating with BenchHub, automatically generate LaTeX tables and BibTeX entries:

```python
# Assuming `results` is from a BenchHub evaluation
try:
    results.benchhub_citation_report(output_path='benchhub_citation_report.tex')
    print("Citation report saved to benchhub_citation_report.tex")
except ValueError as e:
    print(e)
```

Example of `benchhub_citation_report.tex`:

```latex
The evaluation datasets were sampled using BenchHub~\cite{kim2025benchhub}, and the evaluation was conducted using HRET~\cite{lee2025hret}.

% Table of included datasets
\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Dataset} & \textbf{Number of Samples} \\
\midrule
\cite{son-etal-2025-kmmlu} & 30499 \\
\cite{son-etal-2024-hae}   & 4900 \\
\bottomrule
\end{tabular}
\caption{Breakdown of datasets included in the evaluation set.}
\label{tab:eval-dataset}
\end{table}

% --- BibTeX Entries ---

@article{lee2025hret, ...}
@misc{kim2025benchhub, ...}
@inproceedings{son-etal-2025-kmmlu, ...}
```

---

Use this guide to quickly build and document your BenchHub-based evaluation pipelines!"}
