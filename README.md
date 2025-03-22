# Haerae-Evaluation-Toolkit

<p align="center">
  <img src="assets/imgs/logo.png" alt="logo" width="250">
</p>


Haerae-Evaluation-Toolkit is an emerging open-source Python library designed to streamline and standardize the evaluation of Large Language Models (LLMs), with a particular focus on Korean.

## ✨ Key Features

- **Multiple Evaluation Methods**
  - Logit-Based, String-Match, Partial-Match LLM-as-a-Judge, and more.

- **Reasoning Chain Analysis**
  - Dedicated to analyzing extended Korean chain-of-thought reasoning.

- **Extensive Korean Datasets**
  - Includes HAE-RAE Bench, KMMLU, KUDGE, CLiCK, K2-Eval, HRM8K, and more.

- **Scalable Inference-Time Techniques**
  - Best-of-N, Majority Voting, Beam Search, and other advanced methods.

- **Integration-Ready**
  - Supports OpenAI-Compatible Endpoints, Huggingface, and LiteLLM.

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

Below is an example, for more detailed instructions on getting it up and running, see tutorial/kor(eng)/quick_start.md.

### Python Usage

```python
from llm_eval.evaluator import Evaluator

# 1) Initialize an Evaluator with default parameters (optional).
evaluator = Evaluator()

# 2) Run the evaluation pipeline
results = evaluator.run(
    model="huggingface",                        # or "vllm", "openai", etc.
    judge_model=None,                           # specify e.g. "huggingface_judge" if needed
    reward_model=None,                          # specify e.g. "huggingface_reward" if needed
    dataset="haerae_bench",                     # or "kmmlu", "qarv", ...
    subset=["csat_geo", "csat_law"],            # optional subset(s)
    split="test",                               # "train"/"validation"/"test"
    dataset_params={"revision":"main"},         # example HF config
    model_params={"model_name_or_path":"gpt2"}, # example HF Transformers param
    judge_params={},                            # params for judge model (if judge_model is not None)
    reward_params={},                           # params for reward model (if reward_model is not None)
    scaling_method=None,                        # or "beam_search", "best_of_n"
    scaling_params={},             # e.g., {"beam_size":3, "num_iterations":5}
    evaluator_params={}                         # e.g., custom evaluation settings
)


```

- Dataset is loaded from the registry (e.g., `haerae_bench` is just one of many).
- Model is likewise loaded via the registry (`huggingface`, `vllm`, etc.).
- judge_model and reward_model can be provided if you want LLM-as-a-Judge or reward-model logic. If both are None, the system uses a single model backend.
- `ScalingMethod` is optional if you want to do specialized decoding.
- `EvaluationMethod` (e.g., `string_match`, `logit_based`, or `llm_judge`) measures performance.

### CLI Usage

We also provide a simple command-line interface (CLI) via `evaluator.py`:

```bash
python llm_eval/evaluator.py \
  --model huggingface \
  --judge_model huggingface_judge \
  --reward_model huggingface_reward \
  --dataset haerae_bench \
  --subset csat_geo \
  --split test \
  --scaling_method beam_search \
  --evaluation_method string_match \
  --model_params '{"model_name_or_path": "gpt2"}' \
  --scaling_params '{"beam_size":3, "num_iterations":5}' \
  --output_file results.json

```

This command will:

1. Load the `haerae_bench` (subset=`csat_geo`) test split.
2. Create a MultiModel internally with:
Generate model: huggingface → gpt2
Judge model: huggingface_judge (if you pass relevant judge_params)
Reward model: huggingface_reward (if you pass relevant reward_params).  
3. Apply Beam Search (`beam_size=3`).
4. Evaluate final outputs via `string_match`.
5. Save the resulting JSON file to `results.json`.


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
