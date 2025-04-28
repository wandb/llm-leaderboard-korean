# 소개 (Introduction)
HRET(HaeRae Evaluation Toolkit)는 한국어 대형 언어 모델(LLM)에 대해 표준화된 평가환경에서 포괄적인 유효성 검증 기능을 지원하기 위한 오픈소스 라이브러리입니다. 

HRET 프레임워크는 기존 한국어 LLM 평가 방식이 일관되지 않아서 직접적인 비교가 어려웠던 것을 보완하기 위해 다음과 같은 목표를 갖고 있습니다.

## 특징 (Features)
- HRET는 주요 한국어 벤치마크(HAE-RAE Bench, KMMLU, KUDGE, HRM8K 등)를 통합합니다.
- 평가 기법(문자열 일치, 언어 불일치 패널티, 로그 확률 기반 평가, LLM-as-judge)을 지원합니다. 
  로짓 기반으로 토큰 수준의 확률을 제공하기 때문에 모델 신뢰도 평가까지 가능하며, 한글으로 요청한 사항에 대해 그외 언어가 발생했을 때 검출하여 패널티를 부여할 수 있습니다.
- Test-time-scale(Beam Search, Best-of-N, Self-Consistency Voting)을 제공하여 언어 모델의 성능을 여러 각도로 평가할 수 있습니다.
- HuggingFace를 통한 on-premise 사용 뿐만 아니라, litellm, openai-compatible api를 통해 100+개의 online inference와 연동 가능하도록 설계되었습니다. 
- HRET는 한국어 NLP 연구의 재현성과 투명성을 향상시키고, 일관된 대규모 실험 환경을 제공하는 것을 목표로 합니다.

---

# 설치 (Installation)
파이썬(python >= 3.9) 가상환경 구축 후 설치를 권장합니다. 
다음과 같은 과정을 통해 실행 환경을 구축할 수 있습니다.
- 가상환경 구축 (Conda 또는 Venv)
- git clone 명령어로 HRET GitHub 프로젝트를 로컬에 복사해오기
- 요구 패키지 설치

## Conda 가상환경 (Virtual Environment) 구현 예시 
[1] 아나콘다 설치 https://www.anaconda.com/download   
   (다운로드 페이지 우측 하단의 skip registration으로 가입없이 설치 가능)

[2] Anaconda prompt 실행

[3] Conda 환경 생성 및 활성화 (예시: python 3.11)
```bash
conda create -n hret python = 3.11 -y && conda activate hret
```

[4] git clone 수행 (사용자가 선호하는 working directory로 이동 후 실행)
```bash
git clone https://github.com/HAE-RAE/haerae-evaluation-toolkit.git
```

[5] git clone 완료된 로컬 폴더로 이동
```bash
cd haerae-evaluation-toolkit
```

[6] requirements.txt로 요구되는 패키지 설치
```bash
pip install -r requirements.txt
```
---
# 활용 (Usage)

활용할 모델 선정 및 필요시 Access 권한 요청(Optional)
https://huggingface.co/models

---

## 커맨드라인 인터페이스(CLI)로 활용 (ex:google/gemma-3-1b-it)
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
위 커맨드는 다음 사항을 수행합니다.
- haerae_bench (subset=csat_geo) 테스트 분할을 로드합니다.
- 생성 모델: huggingface → google/gemma-3-1b-it
- string_match를 통해 최종 출력을 평가합니다.
- 결과 JSON 파일을 results.json에 저장합니다.

---

## Evaluator API 사용법
- 백엔드 모델: 레지스트리를 통해 로드됩니다 (huggingface, vllm 등).
- 데이터셋: 레지스트리에서 로드됩니다 (예: haerae_bench는 여러 데이터셋 중 하나입니다).
- LLM-as-a-Judge 또는 reward model 로직을 원하는 경우 judge_model 및 reward_model을 제공할 수 있습니다. 둘 다 None인 경우 시스템은 단일 모델 백엔드를 사용합니다.
- test-time-scaling을 수행하려면 ScalingMethod를 선택적으로 사용할 수 있습니다.
- EvaluationMethod(예: string_match, logit_based 또는 llm_judge)는 성능을 측정합니다.

다음은 Evaluator 인터페이스를 사용하여 데이터셋을 로드하고, 모델과 (선택적으로) 스케일링 방법을 적용한 다음, 평가하는 방법에 대한 최소한의 예시입니다.

### Python Usage

```python
from llm_eval.evaluator import Evaluator

# 1) Initialize an Evaluator.
evaluator = Evaluator()

# 2) Run the evaluation pipeline
results = evaluator.run(
    model="huggingface",                        # or "litellm", "openai", etc.
    model_params={"model_name_or_path":"kakaocorp/kanana-nano-2.1b-instruct", "device":"cuda:0", "batch_size": 2, "max_new_tokens": 128}, # example HF Transformers param

    dataset="haerae_bench",                     # or "kmmlu", "qarv", ...
    subset=["standard_nomenclature"],            # optional subset(s)
    split="test",                               # "train"/"validation"/"test"
    dataset_params={},         # example HF config

    judge_model=None,                           # specify e.g. "huggingface_judge" if needed
    judge_params={},                            # params for judge model (if judge_model is not None)

    reward_model=None,                          # specify e.g. "huggingface_reward" if needed
    reward_params={},                           # params for reward model (if reward_model is not None)

    scaling_method=None,                        # or "beam_search", "best_of_n"
    scaling_params={},             # e.g., {"beam_size":3, "num_iterations":5}

    evaluator_params={}                         # e.g., custom evaluation settings
)

print(results)
# e.g. EvaluationResult(metrics={'accuracy': 0.0, 'language_penalizer_average': 0.8733333333333333}, info={'dataset_name': 'haerae_bench', 'subset': ['csat_geo'], 'split': 'test', 'model_backend_name': 'huggingface', 'scaling_method_name': None, 'evaluation_method_name': 'string_match', 'elapsed_time_sec': 1119.5369288921356}, samples=[...]

df = results.to_dataframe()
print(df) # input, reference, prediction, options, chain-of-thought, logits, 등 확인 가능
```

### 백엔드 모델 변경 - vllm (Backend model changed)
```python
from llm_eval.evaluator import Evaluator

# 1) Initialize an Evaluator.
evaluator = Evaluator()

# 2) Run the evaluation pipeline
results = evaluator.run(
model="openai",
model_params={"api_base": "http://0.0.0.0:8000/v1/chat/completions", "model_name": "Qwen/Qwen2.5-7B-Instruct", "batch_size" : 1},

dataset="haerae_bench",
split="test",
subset=["csat_geo"],

evaluation_method='string_match',
)

print(results)
# e.g. EvaluationResult(metrics={'accuracy': 0.34, 'language_penalizer_average': 0.4533333333333333}, info={'dataset_name': 'haerae_bench', 'subset': ['csat_geo'], 'split': 'test', 'model_backend_name': 'openai', 'scaling_method_name': None, 'evaluation_method_name': 'string_match', 'elapsed_time_sec': 49.80667734146118}, samples=[...])
```

---

## 🔍 평가방법 (Evaluation)

### String / Partial Match Evaluation
모델의 예측값(prediction)과 참조값(reference)이 '완전 일치(exact match)하는지 / 부분 일치(Partial Match)하는지' 평가하는 방식을 둘 중에서 선택합니다.

#### Partial match
```python
from llm_eval.evaluator import Evaluator

# 1) Initialize an Evaluator.
evaluator = Evaluator()

# 2) Run the evaluation pipeline
results = evaluator.run(
model="huggingface",
model_params={"model_name_or_path":"Qwen/Qwen2.5-3B-Instruct", "device":"cuda:0", "batch_size": 2, "cot":True, "max_new_tokens": 1024},

dataset="haerae_bench",
split="test",
subset=["csat_geo"],

evaluation_method='partial_match',
)

print(results)
# e.g. EvaluationResult(metrics={'accuracy': 0.5866666666666667}, info={'dataset_name': 'haerae_bench', 'subset': ['csat_geo'], 'split': 'test', 'model_backend_name': 'huggingface', 'scaling_method_name': None, 'evaluation_method_name': 'partial_match', 'elapsed_time_sec': 2286.0827300548553}, samples=[...])
```

#### String match
```python
from llm_eval.evaluator import Evaluator

# 1) Initialize an Evaluator.
evaluator = Evaluator()

# 2) Run the evaluation pipeline
results = evaluator.run(
model="huggingface",
model_params={"model_name_or_path":"Qwen/Qwen2.5-3B-Instruct", "device":"cuda:0", "batch_size": 2, "cot":True, "max_new_tokens": 1024},

dataset="haerae_bench",
split="test",
subset=["csat_geo"],

evaluation_method='string_match',
)
```


### Log Probability Evaluation (로그 확률 평가)
모델이 생성한 각 선택지의 로그 확률(log probability)을 기반으로 정답을 예측하고, 이를 참조값(reference)과 비교하여 정확도를 계산하는 평가 방법입니다. 이 방법은 단순히 모델의 출력(prediction)만을 사용하는 대신, 모델의 내부 확률 정보(로그 확률)를 활용하여 더 신뢰할 수 있는 평가를 수행합니다.

```python
answer_template = "{query} ### 답:"
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
# e.g. EvaluationResult(metrics={'accuracy': 0.25333333333333335, 'language_penalizer_average': 0.0}, info={'dataset_name': 'haerae_bench', 'subset': ['csat_geo'], 'split': 'test', 'model_backend_name': 'huggingface', 'scaling_method_name': None, 'evaluation_method_name': 'log_likelihood', 'elapsed_time_sec': 84.34037137031555}, samples=[...])
```



---

## Scaling_method (선택적임, 반복으로 인해 장시간 소요될 수 있음)
- self_consistency: LLM이 같은 질문에 대해 여러 번 답변한 후, 가장 자주 등장하는 답변을 선택하는 기법


### Self_consistency
<활용 코드>
```python
from llm_eval.evaluator import Evaluator

# 1) Initialize an Evaluator.
evaluator = Evaluator()

# 2) Run the evaluation pipeline
results = evaluator.run(
model="huggingface",
model_params={"model_name_or_path":"Qwen/Qwen2.5-0.5B-Instruct", "device":"cuda", "batch_size": 1}, # example HF Transformers param

dataset="haerae_bench",
split="test",

scaling_method='self_consistency',
)
print(results)
# e.g. results['metrics'] : {'accuracy': 0.00040816326530612246}
```
---

## CoT (Chain of Thought, 생각의 연쇄)
CoT는 복잡한 문제를 단계적으로 해결하는 프로세스를 모델이 따르도록 유도하는 기법입니다.

### cot basic
<활용 코드>
```python
from llm_eval.evaluator import Evaluator

# 1) Initialize an Evaluator.
evaluator = Evaluator()

# 2) Run the evaluation pipeline
results = evaluator.run(
model="huggingface",
dataset="haerae_bench",
split="test",
subset=["csat_geo"],
model_params={"model_name_or_path":"Qwen/Qwen2.5-3B-Instruct", "device":"cuda:0", "batch_size": 2, "cot":True, "max_new_tokens": 512},
)

# e.g. EvaluationResult(metrics={'accuracy': 0.0}, info={'dataset_name': 'haerae_bench', 'subset': ['csat_geo'], 'split': 'test', 'model_backend_name': 'huggingface', 'scaling_method_name': None, 'evaluation_method_name': 'string_match', 'elapsed_time_sec': 1305.4367339611053}, samples=[...])

```

### cot_trigger (선택적)
cot_trigger는 "Chain-of-Thought (CoT)" 방식의 텍스트 생성을 지원하기 위해 사용되는 문자열 트리거입니다. cot=True로 설정된 경우, cot_trigger가 프롬프트에 추가가능합니다. cot_trigger는 모델 프롬프트(prompt)에 추가되어 모델이 체계적으로 사고 과정을 표현하도록 유도합니다.

예를 들어, cot_trigger가 "Let's think step by step."로 설정되면, 모델은 입력된 문제를 단계별로 분석하고 답을 생성하려고 시도합니다.

<활용 코드>
```python
from llm_eval.evaluator import Evaluator

# 1) Initialize an Evaluator.
evaluator = Evaluator()

# 2) Run the evaluation pipeline
results = evaluator.run(
model="huggingface",
dataset="haerae_bench",
split="test",
subset=["csat_geo"],
model_params={"model_name_or_path":"Qwen/Qwen2.5-3B-Instruct", "device":"cuda:0", "batch_size": 2, "cot":True, "cot_trigger": "Let's think step by step.", "max_new_tokens": 512},
)

print(results)
```

### cot_parser
cot_parser는 pythonpath안에 함수가 위치한 곳 이름만 적어두면, 스스로 해당 모듈을 불러와서 parser로 쓸 수 있는 기능


---


### 참조문헌 (References)
- 'vLLM', https://github.com/vllm-project/vllm
- 'Respond in my Language: Mitigating Language Inconsistency in Response Generation based on Large Language Models', https://aclanthology.org/2024.acl-long.229.pdf

## FAQ
Q. 다음 에러 메시지가 출력됩니다: 'Make sure to have access to it at {model url} 403 Client Error. (Request ID: ~~ )'
A. 해당 모델(ex: Llama, Gemma, etc)은 허깅페이스 로그인 후 모델 페이지 상단에 있는 
{Model Name} COMMUNITY LICENSE AGREEMENT의 하단에 Expand to review and access 클릭 후 정보 입력한 다음 Submit 후 허가를 받은 다음 (약 10분) 사용하실 수 있습니다.

---

### 📩 Contact Us
- Development Lead: gksdnf424@gmail.com
- Research Lead: spthsrbwls123@yonsei.ac.kr

---

## 📜 License
Licensed under the Apache License 2.0.
© 2025 The HAE-RAE Team. All rights reserved.
