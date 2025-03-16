# 소개 (Introduction)
HRET(HaeRae Evaluation Toolkit)는 한국어 대형 언어 모델(LLM)에 대해 표준화된 평가환경에서 포괄적인 유효성 검증 기능을 지원하기 위한 오픈소스 라이브러리입니다. 

HRET 프레임워크는 기존 한국어 LLM 평가 방식이 일관되지 않아서 직접적인 비교가 어려웠던 것을 보완하기 위해 다음과 같은 목표를 갖고 있습니다.

## 특징 (Features)
- HRET는 주요 한국어 벤치마크(HAE-RAE Bench, KMMLU, KUDGE, HRM8K 등)를 통합합니다.
- 평가 기법(문자열 일치, 언어 불일치 패널티, 로그 확률 기반 평가, LLM-as-judge)을 지원합니다. 
  로짓 기반으로 토큰 수준의 확률을 제공하기 때문에 모델 신뢰도 평가까지 가능하며, 한글으로 요청한 사항에 대해 그외 언어가 발생했을 때 검출하여 패널티를 부여할 수 있습니다.
- 고급 디코딩 기법(빔 서치, Best-of-N 샘플링, 다수결)을 제공하여 평가의 신뢰성을 높입니다.
- HuggingFace, OpenAI API와 연동 가능하도록 설계되었습니다. 
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

활용할 모델 선정 및 필요시 Access 권한 요청
https://huggingface.co/models

---

## 커맨드라인 인터페이스(CLI)로 활용 (ex:google/gemma-3-1b-it)
```bash
python -m llm_eval.evaluator \
  --model huggingface \
  --model_params '{"model_name_or_path": "google/gemma-3-1b-it"}' \
  --dataset haerae_bench \
  --subset csat_geo \
  --split test \
  --evaluation_method string_match \
  --output_file results.json
```
위 커맨드는 다음 사항을 수행합니다.
- haerae_bench (subset=csat_geo) 테스트 분할을 로드합니다.
- 내부적으로 다음과 같은 MultiModel을 생성합니다:
  생성 모델: huggingface → google/gemma-3-1b-it
- string_match를 통해 최종 출력을 평가합니다.
- 결과 JSON 파일을 results.json에 저장합니다.

---

## Evaluator API 사용법
다음은 Evaluator 인터페이스를 사용하여 데이터셋을 로드하고, 모델과 (선택적으로) 스케일링 방법을 적용한 다음, 평가하는 방법에 대한 최소한의 예시입니다.

### Python Usage

```python
from llm_eval.evaluator import Evaluator

# 1) Initialize an Evaluator with default parameters (optional).
evaluator = Evaluator(
    default_model_backend="huggingface",     # e.g., "vllm", "openai", ...
    default_judge_backend=None,              # e.g., "huggingface_judge"
    default_reward_backend=None,             # e.g., "huggingface_reward"
    default_scaling_method=None,             # e.g., "beam_search", "best_of_n"
    default_evaluation_method="string_match",
    default_split="test"
)

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
    scaling_params={"beam_size":3},             # e.g., {"beam_size":3, "num_iterations":5}
    evaluator_params={}                         # e.g., custom evaluation settings
)

print("Metrics:", results["metrics"])
# e.g. {"accuracy": 0.85, ...}
print("Sample #0:", results["samples"][0])
# e.g. {"input":"...", "reference":"...", "prediction":"..."}

```

- 데이터셋은 레지스트리에서 로드됩니다 (예: haerae_bench는 여러 데이터셋 중 하나입니다).
- 모델도 레지스트리를 통해 로드됩니다 (huggingface, vllm 등).
- LLM-as-a-Judge 또는 reward model 로직을 원하는 경우 judge_model 및 reward_model을 제공할 수 있습니다. 둘 다 None인 경우 시스템은 단일 모델 백엔드를 사용합니다.
- 특수 디코딩을 수행하려면 ScalingMethod를 선택적으로 사용할 수 있습니다.
- EvaluationMethod(예: string_match, logit_based 또는 llm_judge)는 성능을 측정합니다.

---

## 🔍 평가방법 (Evaluation)
### String / Partial Match Evaluation
모델의 예측값(prediction)과 참조값(reference)이 '완전 일치(exact match)하는지 / 부분 일치(Partial Match)하는지' 평가하는 방식을 선택합니다.

  1. 텍스트 정규화 (Normalization)
모델의 출력(prediction)과 정답(reference) 또는 선택지(options)를 비교하기 전에, 다음과 같은 정규화를 수행할 수 있도록 구현되었습니다:
( 대소문자 무시 (ignore_case=True): 모든 텍스트를 소문자로 변환하여 비교합니다.
구두점 제거 (ignore_punctuation=False): 옵션에 따라 구두점(.,!?)을 제거할 수 있습니다.
숫자 제거 (ignore_numbers=False): 옵션에 따라 숫자를 제거할 수 있습니다.
정규표현식 필터링 (regexes_to_ignore): 특정 패턴을 제거할 수 있습니다.
공백 정규화: 여러 개의 공백을 하나로 통일합니다. )

  2. 최종 답변 추출 (Final Answer Extraction)
모델의 출력에서 "Answer:" 이후의 텍스트만 추출하여 평가할 수도 있습니다. 이는 모델 출력에 reasoning chain(추론 과정)이 포함된 경우 유용합니다.

📖 사용 예시
<완전 일치 String Match>
예를 들어, 참조값(reference)이 "서울특별시"이고 모델의 예측값(prediction)이 "서울특별시"라면, 두 값이 정확히 동일하므로 String Match 기준에서는 정답으로 처리됩니다. 또한 MCQA 모드에서 선택지가 ["서울", "부산", "대구"]이고 모델의 예측이 "부산"이라면, "부산"이라는 선택지가 예측 결과와 정확히 일치하므로 정답으로 인정됩니다.

<부분 일치 Partial Match>
예를 들어, 참조값(reference)이 "서울특별시"이고 모델의 예측값(prediction)이 "대한민국 서울특별시 강남구"라면, 비록 정확히 일치하지는 않지만 참조값이 예측값에 포함되어 있으므로 Partial Match 기준에서는 정답으로 처리됩니다. 또한 MCQA 모드에서 선택지가 ["서울", "부산", "대구"]이고 모델의 예측이 "저는 서울에 가고 싶습니다."라면, "서울"이라는 선택지가 예측 결과에 부분적으로 포함되어 있으므로 정답으로 인정됩니다.

### Log Probability Evaluation (로그 확률 평가)
모델이 생성한 각 선택지의 로그 확률(log probability)을 기반으로 정답을 예측하고, 이를 참조값(reference)과 비교하여 정확도를 계산하는 평가 방법입니다.

이 방법은 단순히 모델의 출력(prediction)만을 사용하는 대신, 모델의 내부 확률 정보(로그 확률)를 활용하여 더 신뢰할 수 있는 평가를 수행합니다.

  1. 입력 데이터 구조
이 평가 방법은 다음과 같은 필드를 포함하는 샘플 데이터를 기대합니다:
  -options: 선택지 목록 (예: ["서울", "부산", "대구"])
  -logits: 선택지별 로그 확률 정보를 포함하는 딕셔너리
  -option_log_probs: 각 선택지에 대한 로그 확률 값의 리스트 (예: [-1.2, -0.5, -2.3])
  -reference: 정답 선택지 (예: "부산")
  (선택적) prediction: 모델이 직접 생성한 예측값 (로그 확률 기반 예측이 불가능할 때 사용)

  2. 로그 확률 기반 예측
모델이 제공한 각 선택지의 로그 확률 값(option_log_probs)을 사용하여 가장 높은 확률을 가진 선택지를 예측값으로 결정합니다:
  -최고 로그 확률 선택: 로그 확률 값 중 최댓값을 찾아 해당 선택지를 예측값으로 설정합니다.
  -예: option_log_probs = [-1.2, -0.5, -2.3]이라면, 두 번째 선택지("부산")가 가장 높은 로그 확률(-0.5)을 가지므로 이를 예측값으로 설정.
  -Fallback (대체 방식): 만약 로그 확률 값이 없거나 유효하지 않은 경우, 사전에 저장된 prediction 필드를 사용해 예측합니다.

  3. 정확도 계산
로그 확률 기반으로 결정된 예측값(predicted_option)을 참조값(reference)과 비교하여 정확도를 계산합니다: 예측값이 참조값과 동일하면 정답으로 간주하며, 전체 샘플 중 정답 비율(accuracy)을 반환.

### Math Match Evaluation (수학적 일치 평가)
모델의 출력과 참조값을 수학적으로 동등한지 비교하여 평가합니다. 단순한 문자열 비교 대신, 수학적 표현의 수학적 동등성을 확인하는 데 초점을 맞춥니다.
즉, 모델이 생성한 수학적 표현(예측값, prediction)과 참조값(reference) 간의 수학적 동등성을 평가하는 방법으로서, 다음과 같은 특징을 가집니다:
  -문자열 일치가 아닌 수학적 동등성 비교: 두 표현이 동일한 결과를 나타내는지 확인합니다.
  -LaTeX 및 일반 수학 표현 지원: LaTeX 형식이나 일반 수학 표현 모두 처리 가능합니다.
  -추론 과정 포함 가능: Chain-of-Thought(CoT) 방식의 출력에서 최종 답변만 추출하여 평가할 수 있습니다.

  1. 입력 데이터 구조
이 평가 방법은 다음과 같은 필드를 포함하는 샘플 데이터를 기대합니다:
  -prediction: 모델이 생성한 수학적 표현 (예: "정답은 \\boxed{1,2,3}")
  -reference: 정답으로 제공된 수학적 표현 (예: "${1,2} \\cup {3}")

  2. 최종 답변 추출
모델 출력에 추론 과정이 포함된 경우, 정규식 패턴을 사용하여 최종 답변만 추출합니다:
  -예: "정답은 \\boxed{1,2,3}" → "\\boxed{1,2,3}"
   (추출은 다음과 같은 패턴을 기반으로 수행됩니다):
  -정답\s*:\s*(.*?)(?:\n|$)
  -Answer\s*:\s*(.*?)(?:\n|$)

  3. 수학적 표현 파싱
math_verify 라이브러리를 사용하여 LaTeX 또는 일반 수학 표현을 파싱합니다:
  -예: \\boxed{1,2,3} → 내부적으로 수학적 객체로 변환.
  -파싱 실패 시 해당 샘플은 평가에서 제외되며 실패율(parse_failure_rate)로 기록됩니다.

  4. 수학적 동등성 검증
파싱된 두 표현(예측값과 참조값)이 동일한 결과를 나타내는지 확인합니다:

  -예: {1,2} \\cup {3}와 {3,1,2}는 동일한 집합을 나타내므로 동등성 검증에서 성공(True)으로 처리됩니다.
  -검증 실패 시 해당 샘플은 실패율(verify_failure_rate)로 기록됩니다.

  5. 정확도 계산
  -모든 샘플 중 동등성 검증에 성공한 비율을 정확도(accuracy)로 반환합니다.

📖 사용 예시
<샘플 정의>
```python
samples = [
    {
        "prediction": "정답은 \\boxed{1,2,3} 입니다.",
        "reference": "{1,2} \\cup {3}"
    },
    {
        "prediction": "Answer: x^2 + 2x + 1",
        "reference": "(x+1)^2"
    }
]
```

<활용 코드>
```python
evaluator = MathMatchEvaluator()
results = evaluator.evaluate_predictions(samples)
print(results)
```

<최종 결과>
```bash
{
    "accuracy": 1.0,
    "parse_failure_rate": 0.0,
    "verify_failure_rate": 0.0
}
```

---

## Scaling_method (선택적임, 반복으로 인해 장시간 소요될 수 있음)
- self_consistency: LLM이 같은 질문에 대해 여러 번 답변한 후, 가장 자주 등장하는 답변을 선택하는 기법
- greedy: 가장 확률이 높은 단어를 반복해서 선택하는 단순한 방식. (단점: 다양성이 부족함)
- beam_search: 여러 개의 가능성을 동시에 고려하면서 가장 좋은 문장을 찾는 방식. (좀 더 안정적인 결과 제공)
- top_k: 확률이 높은 상위 k개의 단어 중에서 랜덤하게 선택 (텍스트 다양성 증가)
- top_p: 확률의 누적 합이 p 이상이 되는 단어 집합에서 선택 (더 자연스러운 생성 가능)
- temperature_sampling:	확률 분포를 조정하여 더 창의적인 결과를 생성 (온도 높을수록 다양성 증가)



### 참조문헌 (References)
- 'vLLm', https://github.com/vllm-project/vllm
- 'Respond in my Language: Mitigating Language Inconsistency in Response Generation based on Large Language Models', https://aclanthology.org/2024.acl-long.229.pdf

---

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