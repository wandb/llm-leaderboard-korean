# Introduction
**HRET (HaeRae Evaluation Toolkit)** is an open-source library designed to support comprehensive validation of Korean Large Language Models (LLMs) in a standardized evaluation environment.

The **HRET framework** was developed to address the difficulty in directly comparing Korean LLMs due to inconsistent evaluation methods. Its main goals are as follows:

## Features
- HRET integrates major Korean benchmarks such as **HAE-RAE Bench, KMMLU, KUDGE, HRM8K**, and more.
- It supports multiple evaluation methods, including string match, language mismatch penalty, log-likelihood-based evaluation, and LLM-as-judge.
  Thanks to logit-based evaluation, token-level probability and model confidence can be assessed. Additionally, a penalty can be applied when a response includes non-Korean content.
- Provides **test-time scaling methods** such as Beam Search, Best-of-N, and Self-Consistency Voting to evaluate LLM performance from various angles.
- Designed to work not only with HuggingFace models in on-premise environments but also with over 100 online inference endpoints via **litellm** or **OpenAI-compatible APIs**.
- HRET aims to enhance **reproducibility and transparency** in Korean NLP research and provide a consistent large-scale experimental environment.

---

# Installation
We recommend setting up a Python virtual environment (Python >= 3.9).
Follow these steps to set up the environment:

- Create a virtual environment (via Conda or venv)
- Clone the HRET GitHub repository
- Install required packages

## Conda Virtual Environment Example
1. Install Anaconda: https://www.anaconda.com/download  
   (Click "Skip registration" at the bottom right to install without signing up)

2. Launch Anaconda Prompt

3. Create and activate the Conda environment (example: Python 3.11):
```bash
conda create -n hret python=3.11 -y && conda activate hret
```

4. Clone the repository (navigate to your preferred working directory first):
```bash
git clone https://github.com/HAE-RAE/haerae-evaluation-toolkit.git
```

5. Move to the cloned folder:
```bash
cd haerae-evaluation-toolkit
```

6. Install required packages:
```bash
pip install -r requirements.txt
```

---

# Usage

Select a model and, if needed, request access permissions:  
https://huggingface.co/models

## Using the Command-Line Interface (CLI) (Example: google/gemma-3-1b-it)
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

This command:
- Loads the `haerae_bench` dataset (subset: `csat_geo`) with test split
- Uses `google/gemma-3-1b-it` model from HuggingFace
- Evaluates with `string_match` method
- Saves results to `results.json`

---

## Evaluator API
- Backend models are loaded via registry (e.g., huggingface, vllm)
- Datasets are also loaded via registry (e.g., haerae_bench, kmmlu)
- You may provide `judge_model` and/or `reward_model` if needed. If both are `None`, the system uses a single model backend.
- Optionally, use `scaling_method` to enable test-time scaling.
- Use `evaluation_method` (e.g., string_match, logit_based, llm_judge) to measure performance.

### Python Example
```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()

results = evaluator.run(
    model="huggingface",
    model_params={"model_name_or_path":"kakaocorp/kanana-nano-2.1b-instruct", "device":"cuda:0", "batch_size": 2, "max_new_tokens": 128},

    dataset="haerae_bench",
    subset=["standard_nomenclature"],
    split="test",
    dataset_params={},

    judge_model=None,
    judge_params={},

    reward_model=None,
    reward_params={},

    scaling_method=None,
    scaling_params={},

    evaluator_params={}
)

print(results)
df = results.to_dataframe()
print(df)  # Columns: input, reference, prediction, options, chain-of-thought, logits, etc.
```

### Using `vllm` Backend
```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()

results = evaluator.run(
    model="openai",
    model_params={"api_base": "http://0.0.0.0:8000/v1/chat/completions", "model_name": "Qwen/Qwen2.5-7B-Instruct", "batch_size" : 1},

    dataset="haerae_bench",
    split="test",
    subset=["csat_geo"],

    evaluation_method='string_match',
)

print(results)
```

---

## 🔍 Evaluation Methods

### String / Partial Match Evaluation
Compares model predictions to references using exact or partial match.

#### Partial Match
```python
results = evaluator.run(
    model="huggingface",
    model_params={"model_name_or_path":"Qwen/Qwen2.5-3B-Instruct", "device":"cuda:0", "batch_size": 2, "cot":True, "max_new_tokens": 1024},

    dataset="haerae_bench",
    split="test",
    subset=["csat_geo"],

    evaluation_method='partial_match',
)
```

#### String Match
```python
results = evaluator.run(
    model="huggingface",
    model_params={"model_name_or_path":"Qwen/Qwen2.5-3B-Instruct", "device":"cuda:0", "batch_size": 2, "cot":True, "max_new_tokens": 1024},

    dataset="haerae_bench",
    split="test",
    subset=["csat_geo"],

    evaluation_method='string_match',
)
```

### Log Probability Evaluation
Uses log probabilities of model outputs to estimate answer correctness.
```python
answer_template = "{query} ### Answer:"
results = evaluator.run(
    model="huggingface",
    model_params={"model_name_or_path":"kakaocorp/kanana-nano-2.1b-instruct", "device":"cuda:0", "batch_size": 4, "max_new_tokens": 128},

    dataset="haerae_bench",
    split="test",
    evaluation_method='log_likelihood',
    subset=["csat_geo"],
    dataset_params = {"base_prompt_template" : answer_template},
)
print(results)
```

---

## Scaling Methods (Optional, may increase runtime)

### Self Consistency
```python
results = evaluator.run(
    model="huggingface",
    model_params={"model_name_or_path":"Qwen/Qwen2.5-0.5B-Instruct", "device":"cuda", "batch_size": 1},

    dataset="haerae_bench",
    split="test",

    scaling_method='self_consistency',
)
print(results)
```

---

## CoT (Chain of Thought)
CoT guides the model to solve problems step-by-step.

### Basic CoT
```python
results = evaluator.run(
    model="huggingface",
    dataset="haerae_bench",
    split="test",
    subset=["csat_geo"],
    model_params={"model_name_or_path":"Qwen/Qwen2.5-3B-Instruct", "device":"cuda:0", "batch_size": 2, "cot":True, "max_new_tokens": 512},
)
```

### CoT Trigger (Optional)
Triggers the CoT process with a specific prompt like "Let's think step by step."
```python
results = evaluator.run(
    model="huggingface",
    dataset="haerae_bench",
    split="test",
    subset=["csat_geo"],
    model_params={"model_name_or_path":"Qwen/Qwen2.5-3B-Instruct", "device":"cuda:0", "batch_size": 2, "cot":True, "cot_trigger": "Let's think step by step.", "max_new_tokens": 512},
)
```

---

## References
- [vLLM Project](https://github.com/vllm-project/vllm)
- [Respond in my Language (ACL 2024)](https://aclanthology.org/2024.acl-long.229.pdf)

---

## FAQ
**Q. I got the following error: `403 Client Error. Make sure to have access to it at {model URL}`**  
**A.** The model (e.g., LLaMA, Gemma) requires access approval. Go to the model's HuggingFace page, expand the "Community License Agreement" section, fill in the form, and click Submit. Approval typically takes about 10 minutes.

---

## 📩 Contact Us
- Development Lead: gksdnf424@gmail.com
- Research Lead: spthsrbwls123@yonsei.ac.kr

---

## 📜 License
Licensed under the Apache License 2.0.  
© 2025 The HAE-RAE Team. All rights reserved.

