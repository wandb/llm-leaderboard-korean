# Haerae-Evaluation-Toolkit

Haerae-Evaluation-Toolkit is an emerging open-source Python library designed to streamline and standardize the evaluation of Large Language Models (LLMs), with a particular focus on Korean.

> **Currently in Development!**

## ✨ Key Features

- **Multiple Evaluation Methods**
  - Logit-Based, String-Match, LLM-as-a-Judge, and more.

- **Reasoning Chain Analysis**
  - Dedicated to analyzing extended Korean chain-of-thought reasoning.

- **Extensive Korean Datasets**
  - Includes HAE-RAE Bench, KMMLU, KUDGE, QARV, K2-Eval, and HRM8K.

- **Scalable Inference-Time Techniques**
  - Best-of-N, Majority Voting, Beam Search, and other advanced methods.

- **Integration-Ready**
  - Supports OpenAI-Compatible Endpoints, vLLM, and LiteLLM.

- **Flexible and Pluggable Architecture**
  - Easily extend with new datasets, evaluation metrics, and inference backends.

---

## 🚀 Project Status

We are actively developing core features and interfaces. Current goals include:

- **Unified API**
  - Seamless loading and integration of diverse Korean benchmark datasets.

- **Configurable Inference Scaling**
  - Generate higher-quality outputs through techniques like best-of-N and beam search.

- **Pluggable Evaluation Methods**
  - Enable chain-of-thought assessments, logit-based scoring, and standard evaluation metrics.

- **Modular Architecture**
  - Easily extendable for new backends, tasks, or custom evaluation logic.

---

## 🛠️ Key Components

- **Dataset Abstraction**
  - Load and preprocess your datasets (or subsets) with minimal configuration.

- **Scalable Methods**
  - Apply decoding strategies such as sampling, beam search, and best-of-N approaches.

- **Evaluation Library**
  - Compare predictions to references, use judge models, or create custom scoring methods.

- **Registry System**
  - Add new components (datasets, models, scaling methods) via simple decorator-based registration.

---

## ⚙️ Installation

(Currently under development — installation steps may vary.)

```bash
git clone https://github.com/HAE-RAE/haerae-evaluation-toolkit.git

```


---

## 🚀 Quickstart: Using the Evaluator API

Below is a minimal example of how to use the `Evaluator` interface to load a dataset, apply a model and (optionally) a scaling method, and then evaluate the outputs.

### Python Usage

```python
from llm_eval.evaluator import Evaluator

# Initialize an Evaluator with default parameters (optional).
evaluator = Evaluator(
    default_model_backend="huggingface",     # e.g., "vllm", "openai", etc.
    default_scaling_method=None,             # e.g., "self-consistency", "best_of_n"
    default_evaluation_method="string_match",
    default_split="test"
)

# Run the evaluation pipeline
results = evaluator.run(
    model="huggingface",                     # or "vllm", "openai", etc.
    dataset="haerae_bench",                  # or "kmmlu", "qarv", ...
    subset="csat_geo",                       # optional subset (list or string)
    split="test",                            # "train"/"validation"/"test"
    dataset_params={"revision":"main"},      # example HuggingFace config
    model_params={"model_name":"gpt2"},      # example HF Transformers param
    scaling_method=None,                     # or "beam_search"
    scaling_params={},                       # e.g. {"beam_size":3, "num_iterations":5}
    evaluator_params={}                      # e.g. custom evaluation settings
)

print("Metrics:", results["metrics"])
# e.g. {"accuracy": 0.83, ...}
print("Samples:", results["samples"][0])
# e.g. {"input":"...", "reference":"...", "prediction":"..."}
```

- Dataset is loaded from the registry (e.g., `haerae_bench` is just one of many).
- Model is likewise loaded via the registry (`huggingface`, `vllm`, etc.).
- `ScalingMethod` is optional if you want to do specialized decoding.
- `EvaluationMethod` (e.g., `string_match`, `logit_based`, or `llm_judge`) measures performance.

### CLI Usage

We also provide a simple command-line interface (CLI) via `evaluator.py`:

```bash
python llm_eval/evaluator.py \
  --model huggingface \
  --dataset haerae_bench \
  --subset csat_geo \
  --split test \
  --scaling_method beam_search \
  --evaluation_method string_match \
  --model_params '{"model_name": "gpt2"}' \
  --scaling_params '{"beam_size":3, "num_iterations":5}' \
  --output_file results.json
```

This command will:

1. Load the `haerae_bench` (subset=`csat_geo`) test split.
2. Initialize a HuggingFace-based model (`model_name`: `gpt2`).
3. Apply Beam Search (`beam_size=3`).
4. Evaluate final outputs via `string_match`.
5. Save the resulting JSON file to `results.json`.

---

## 🔧 Advanced: PipelineRunner

If you need deeper customization or want to orchestrate multiple tasks in one script, you can use the `PipelineRunner` class directly:

```python
from llm_eval.runner import PipelineRunner

runner = PipelineRunner(
    dataset_name="haerae_bench",
    subset=["csat_geo", "csat_law"],
    split="test",
    model_backend_name="vllm",
    scaling_method_name="best_of_n",
    evaluation_method_name="string_match",
    dataset_params={},
    model_backend_params={"endpoint": "http://localhost:8000"},
    scaling_params={"n": 5},
    evaluator_params={},
)

results = runner.run()
print(results["metrics"])
```

---

## 🤝 Contributing & Contact

We welcome collaborators, contributors, and testers interested in advancing LLM evaluation methods, especially for Korean language tasks.

### 📩 Contact Us

- Development Lead: gksdnf424@gmail.com
- Research Lead: spthsrbwls123@yonsei.ac.kr

We look forward to hearing your ideas and contributions!

---

## 📜 License

Licensed under the Apache License 2.0.

© 2025 The HAE-RAE Team. All rights reserved.
