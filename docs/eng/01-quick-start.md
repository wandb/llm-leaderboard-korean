# Introduction

HRET (HaeRae Evaluation Toolkit) is an open-source library designed to provide comprehensive validation capabilities for Korean large language models (LLMs) in a standardized evaluation environment.

The HRET framework addresses inconsistencies in existing Korean LLM evaluation methods—where direct comparison was difficult—by aiming to:

## Features

* **Unified Benchmarks**: Integrates major Korean benchmarks (HAE-RAE Bench, KMMLU, KUDGE, HRM8K, etc.).
* **Evaluation Techniques**: Supports string matching, language inconsistency penalties, log‐probability‐based evaluation, and LLM-as-a-judge. Because it provides token-level probabilities via logits, you can assess model confidence, and penalize outputs in languages other than Korean when they appear.
* **Test‐Time Scaling**: Offers Beam Search, Best‐of‐N, and Self‐Consistency Voting to evaluate model performance from multiple angles.
* **Flexible Backends**: Designed to work on‐premise via HuggingFace, as well as integrate with 100+ online inference endpoints via litellm or OpenAI-compatible APIs.
* **Research Reproducibility**: Aims to enhance transparency and consistency for large‐scale experiments in Korean NLP.

---

# Installation

It is recommended to create and activate a Python (≥3.10) virtual environment before installing. The following steps outline the setup process:

1. Create and activate a virtual environment (Conda or venv)
2. Clone the HRET GitHub repository
3. Install required packages

## Example: Conda Environment

1. Download and install Anaconda: [https://www.anaconda.com/download](https://www.anaconda.com/download) (click “skip registration” to install without signing up)
2. Open the Anaconda Prompt
3. Create and activate a Conda environment (e.g., Python 3.11):

   ```bash
   conda create -n hret python=3.11 -y && conda activate hret
   ```
4. Clone the repository:

   ```bash
   git clone https://github.com/HAE-RAE/haerae-evaluation-toolkit.git
   ```
5. Change into the project directory:

   ```bash
   cd haerae-evaluation-toolkit
   ```
6. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

# Usage

Before running evaluations, choose your model and (if necessary) request access permissions: [https://huggingface.co/models](https://huggingface.co/models)

## Command-Line Interface (CLI) Example

```bash
python -m llm_eval.evaluator \
  --model huggingface \
  --model_params '{"model_name_or_path": "google/gemma-3-1b-it"}' \
  --dataset haerae_bench \
  --subset standard_nomenclature \
  --split test \
  --evaluation_method string_match \
  --output_file results.json
```

This command will:

* Load the `haerae_bench` dataset (subset: `standard_nomenclature`, split: `test`)
* Use the HuggingFace model `google/gemma-3-1b-it`
* Evaluate outputs via string matching
* Save results to `results.json`

---

## Evaluator API Usage

The `Evaluator` class lets you load datasets, apply models, scale outputs, and evaluate results programmatically. You can also specify separate judge or reward models for LLM-as-judge functionality.

### Python Usage

```python
from llm_eval.evaluator import Evaluator

# 1) Initialize the evaluator
evaluator = Evaluator()

# 2) Run the evaluation pipeline
results = evaluator.run(
    model="huggingface",  
    model_params={
        "model_name_or_path": "kakaocorp/kanana-nano-2.1b-instruct",
        "device": "cuda:0",
        "batch_size": 2,
        "max_new_tokens": 128
    },
    dataset="haerae_bench",
    subset=["standard_nomenclature"],  # optional
    split="test",
    judge_model=None,           # or specify a separate judge model
    reward_model=None,          # or specify a separate reward model
    scaling_method=None,        # e.g., "beam_search", "best_of_n"
    evaluation_method="string_match"
)

print(results)
# Example output:
# EvaluationResult(
#   metrics={'accuracy': 0.0, 'language_penalizer_average': 0.8733},
#   info={
#     'dataset_name': 'haerae_bench',
#     'subset': ['standard_nomenclature'],
#     'split': 'test',
#     'model_backend_name': 'huggingface',
#     'scaling_method_name': None,
#     'evaluation_method_name': 'string_match',
#     'elapsed_time_sec': 1119.5
#   },
#   samples=[...]
#)

df = results.to_dataframe()
print(df)  # DataFrame with inputs, references, predictions, logits, etc.
```

### Changing Backend to vLLM or OpenAI-Compatible API

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.run(
    model="openai",
    model_params={
        "api_base": "http://localhost:8000/v1/chat/completions",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "batch_size": 1
    },
    dataset="haerae_bench",
    subset=["standard_nomenclature"],
    split="test",
    evaluation_method="string_match"
)

print(results)
# Example:
# EvaluationResult(metrics={'accuracy': 0.34, 'language_penalizer_average': 0.4533}, ...)
```

---

## 🔍 Evaluation Methods

### String / Partial Match

Evaluates whether the model’s prediction matches the reference exactly or partially.

#### Partial Match

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.run(
    model="huggingface",
    model_params={"model_name_or_path": "Qwen/Qwen2.5-3B-Instruct", "device": "cuda:0", "batch_size": 2, "cot": True, "max_new_tokens": 1024},
    dataset="haerae_bench",
    split="test",
    subset=["standard_nomenclature"],
    evaluation_method="partial_match"
)

print(results)
# e.g. metrics={'accuracy': 0.5867}
```

#### String Match

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.run(
    model="huggingface",
    model_params={"model_name_or_path": "Qwen/Qwen2.5-3B-Instruct", "device": "cuda:0", "batch_size": 2, "cot": True, "max_new_tokens": 1024},
    dataset="haerae_bench",
    split="test",
    subset=["standard_nomenclature"],
    evaluation_method="string_match"
)
```

### Log-Probability Evaluation

Uses the model’s token-level log probabilities to choose the highest-likelihood answer and compare against the reference.

```python
answer_template = "{query} ### Answer:"
results = evaluator.run(
    model="huggingface",
    model_params={
        "model_name_or_path": "kakaocorp/kanana-nano-2.1b-instruct",
        "device": "cuda:0",
        "batch_size": 4,
        "max_new_tokens": 128
    },
    dataset="haerae_bench",
    split="test",
    subset=["standard_nomenclature"],
    evaluation_method="log_likelihood",
    dataset_params={"base_prompt_template": answer_template}
)

print(results)
# e.g. metrics={'accuracy': 0.2533, 'language_penalizer_average': 0.0}
```

---

## Scaling Method (Optional)

Note: Repeated runs may increase runtime significantly.

### Self-Consistency

Runs multiple generations for the same input and selects the most frequent answer.

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.run(
    model="huggingface",
    model_params={"model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "device": "cuda", "batch_size": 1},
    dataset="haerae_bench",
    split="test",
    scaling_method="self_consistency"
)
print(results["metrics"])
# e.g. {'accuracy': 0.0004}
```

---

## Chain-of-Thought (CoT)

Encourages the model to generate reasoning steps before the final answer.

### Basic CoT

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.run(
    model="huggingface",
    dataset="haerae_bench",
    split="test",
    subset=["standard_nomenclature"],
    model_params={
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "device": "cuda:0",
        "batch_size": 2,
        "cot": True,
        "max_new_tokens": 512
    }
)
```

### CoT Trigger (Optional)

You can add a custom trigger string (e.g., “Let’s think step by step.”) to prompt the model into structured reasoning.

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.run(
    model="huggingface",
    dataset="haerae_bench",
    split="test",
    subset=["standard_nomenclature"],
    model_params={
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "device": "cuda:0",
        "batch_size": 2,
        "cot": True,
        "cot_trigger": "Let's think step by step.",
        "max_new_tokens": 512
    }
)
print(results)
```

### CoT Parser

If you have a parsing function defined in your Python path, specify its module path as `cot_parser`, and the evaluator will import it automatically.

---

## References

* [vLLM](https://github.com/vllm-project/vllm)
* “Respond in my Language: Mitigating Language Inconsistency in Response Generation based on Large Language Models,” ACL 2024 Long Paper ([https://aclanthology.org/2024.acl-long.229.pdf](https://aclanthology.org/2024.acl-long.229.pdf))

---

## FAQ

**Q.** I see the error:

```
Make sure to have access to it at {model url} 403 Client Error. (Request ID: ~~)
```

**A.** For community-licensed models on HuggingFace (e.g., Llama, Gemma), you must log in, navigate to the model page, click **Expand to review and access** under the license agreement, agree, and submit. After about 10 minutes your access will be granted.

---

## 📩 Contact Us

* **Development Lead**: [gksdnf424@gmail.com](mailto:gksdnf424@gmail.com)
* **Research Lead**: [spthsrbwls123@yonsei.ac.kr](mailto:spthsrbwls123@yonsei.ac.kr)

---

## 📜 License

Licensed under the Apache License 2.0.
© 2025 The HAE-RAE Team. All rights reserved.
