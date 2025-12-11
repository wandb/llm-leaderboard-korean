# Horangi 개발 가이드

## 📁 프로젝트 구조

```
src/horangi/
├── evals/              # 벤치마크 설정 파일
│   ├── __init__.py     # 벤치마크 등록
│   ├── ko_hellaswag.py
│   ├── kmmlu.py
│   └── ...
├── core/               # 핵심 로직
│   ├── factory.py      # Task 생성 (create_benchmark)
│   ├── loaders.py      # 데이터 로딩 (Weave, JSONL)
│   └── answer_format.py # 정답 형식 변환
├── scorers/            # 커스텀 Scorer
│   ├── __init__.py     # Scorer 등록
│   ├── bfcl_scorer.py
│   ├── kobbq_scorer.py
│   └── ...
├── solvers/            # 커스텀 Solver
│   ├── __init__.py     # Solver 등록
│   └── bfcl_solver.py
└── data/               # 로컬 데이터 파일 (JSONL)
```

---

## 🎯 새 벤치마크 추가하기

### Step 1: Config 파일 생성

`evals/` 폴더에 새 파일을 만들고 `CONFIG` 딕셔너리를 정의합니다.

```python
# evals/my_benchmark.py
"""
My Benchmark - 벤치마크 설명

원본: [링크]
데이터: Weave 또는 JSONL
"""

CONFIG = {
    # 데이터 소스
    "data_type": "weave",  # "weave" 또는 "jsonl"
    "data_source": "weave:///entity/project/object/DatasetName:latest",
    
    # 필드 매핑
    "field_mapping": {
        "id": "id",           # 샘플 ID
        "input": "question",  # 입력 (질문)
        "target": "answer",   # 정답 (MCQA: A/B/C/D, 생성: 텍스트)
        "choices": "options", # 선택지 (MCQA만)
    },
    
    # 정답 형식 변환
    "answer_format": "identity",  # 아래 옵션 참고
    
    # Solver & Scorer
    "solver": "multiple_choice",  # 또는 "generate"
    "scorer": "choice",           # 또는 "match", 커스텀 scorer
    
    # 시스템 프롬프트
    "system_message": "주어진 질문에 가장 적절한 답을 선택하세요.",
}
```

### Step 2: `evals/__init__.py`에 등록

```python
# evals/__init__.py에 추가
from horangi.evals.my_benchmark import CONFIG as my_benchmark

BENCHMARKS: dict = {
    ...
    "my_benchmark": my_benchmark,
}
```

### Step 3: `eval_tasks.py`에 Task 함수 추가

```python
# eval_tasks.py에 추가
@task
def my_benchmark(shuffle: bool = False, limit: int | None = None) -> Task:
    """My Benchmark - 설명"""
    return create_benchmark(name="my_benchmark", shuffle=shuffle, limit=limit)
```

---

## 📋 Config 필드 상세 설명

### `data_type` & `data_source`

| data_type | data_source 형식 | 예시 |
|-----------|------------------|------|
| `weave` | Weave 객체 URI | `weave:///wandb-korea/evaluation-job/object/KMMLU:latest` |
| `jsonl` | 파일명 (data/ 기준) | `ko_aime2025.jsonl` |

### `field_mapping`

데이터셋 필드 → Sample 필드 매핑

| Sample 필드 | 설명 | 필수 |
|-------------|------|------|
| `id` | 샘플 고유 ID | ❌ |
| `input` | 모델 입력 (질문) | ✅ |
| `target` | 정답 | ❌ (거부 태스크 등) |
| `choices` | 선택지 리스트 | ❌ (MCQA만) |

**여러 필드 후보 지정:**
```python
"id": ["id", "sample_id", "idx"],  # 순서대로 시도
```

### `answer_format`

정답 변환 방식:

| 값 | 설명 | 예시 |
|----|------|------|
| `identity` | 변환 없음 | `"정답"` → `"정답"` |
| `index_0` | 0-indexed 숫자 → A/B/C | `0` → `"A"` |
| `to_string` | 숫자 → 문자열 | `42` → `"42"` |
| `text` | 텍스트 → 선택지 인덱스 | `"사과"` → `"A"` (choices 필요) |

### `solver`

| 값 | 설명 | 용도 |
|----|------|------|
| `multiple_choice` | 선택지 제시 + 선택 | MCQA |
| `generate` | 자유 형식 생성 | 생성 태스크 |
| `bfcl_solver` | Tool calling | BFCL |
| `bfcl_text_solver` | 프롬프트 기반 함수 호출 | BFCL (오픈소스) |

### `scorer`

| 값 | 설명 | 용도 |
|----|------|------|
| `choice` | 선택지 정확도 | MCQA |
| `match` | 정확 일치 | 단답형 |
| `match_numeric` | 숫자 일치 | 수학 |
| `model_graded_qa` | LLM 채점 | 주관식 |
| 커스텀 | `scorers/`에 정의 | 특수 평가 |

### 추가 옵션

| 필드 | 설명 | 예시 |
|------|------|------|
| `base` | inspect_evals 상속 | `"inspect_evals.hellaswag.hellaswag"` |
| `split` | 데이터 분할 | `"train"`, `"test"` |
| `sampling` | 샘플링 방식 | `"stratified"`, `"balanced"` |
| `sampling_by` | 그룹화 필드 | `"category"` |
| `default_fields` | 누락 필드 기본값 | `{"image": None}` |

---

## 🔧 커스텀 Scorer 추가하기

### Step 1: Scorer 파일 생성

```python
# scorers/my_scorer.py
"""
My Custom Scorer - 설명
"""

from inspect_ai.scorer import (
    Score, Scorer, Target, scorer, metric, Metric,
    SampleScore, accuracy, CORRECT, INCORRECT,
)
from inspect_ai.solver import TaskState


@metric
def my_custom_metric() -> Metric:
    """커스텀 메트릭"""
    def metric_fn(scores: list[SampleScore]) -> float:
        # 점수 계산 로직
        correct = sum(1 for s in scores if s.score.value == CORRECT)
        return correct / len(scores) if scores else 0.0
    return metric_fn


@scorer(metrics=[accuracy(), my_custom_metric()])
def my_scorer() -> Scorer:
    """커스텀 Scorer"""
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        expected = target.text
        
        # 평가 로직
        is_correct = response.strip() == expected.strip()
        
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=response[:100],
            explanation=f"Expected: {expected}, Got: {response[:50]}",
            metadata={"custom_field": "value"},
        )
    
    return score
```

### Step 2: `scorers/__init__.py`에 등록

```python
from horangi.scorers.my_scorer import my_scorer

__all__ = [
    ...
    "my_scorer",
]
```

### Step 3: Config에서 사용

```python
CONFIG = {
    ...
    "scorer": "my_scorer",
}
```

---

## 📝 체크리스트

새 벤치마크 추가 시 확인사항:

- [ ] `evals/` 폴더에 config 파일 생성
- [ ] `evals/__init__.py`에 import 및 BENCHMARKS 추가
- [ ] `eval_tasks.py`에 @task 함수 추가
- [ ] (커스텀 scorer 필요 시) `scorers/`에 파일 생성 및 등록
- [ ] (커스텀 solver 필요 시) `solvers/`에 파일 생성 및 등록
- [ ] 테스트 실행: `inspect eval eval_tasks.py@my_benchmark --model openai/gpt-4o -T limit=5`

---

## 🔗 참고

- [Inspect AI Docs](https://inspect.ai-safety-institute.org.uk/)
- [inspect_evals GitHub](https://github.com/UKGovernmentBEIS/inspect_evals)
- [WandB Weave](https://wandb.ai/site/weave)

