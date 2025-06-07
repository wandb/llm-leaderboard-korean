# BenchHub 데이터셋 활용 가이드

HRET(Haerae Evaluation Toolkit)은 다양한 벤치마크를 통합하고, 사용자가 원하는 기준으로 손쉽게 평가 데이터셋을 구성할 수 있는 **BenchHub**를 지원합니다. 이 가이드에서는 HRET에서 BenchHub를 로드하고 필터링 기능으로 맞춤형 평가를 수행하며, 논문 작성을 위한 인용 리포트를 생성하는 방법까지 상세히 알아봅니다.

## 목차

1. [BenchHub란?](#benchhub란)
2. [HRET에서 BenchHub 사용하기](#hret에서-benchhub-사용하기)

   * 2.1 [기본 사용법](#기본-사용법)
   * 2.2 [필터링을 통한 맞춤형 평가셋 구성](#필터링을-통한-맞춤형-평가셋-구성)
3. [평가 결과 및 정보 확인](#평가-결과-및-정보-확인)
4. [인용(Citation) 리포트 생성하기](#인용citation-리포트-생성하기)

---

## 1. BenchHub란?

* **통합 벤치마크 저장소**로, 여러 곳에 흩어진 벤치마크를 한데 모아 관리 문제를 해결합니다.
* **주요 컨셉**

  * **통합과 자동 분류**: 데이터셋을 skill, subject, target 등 기준으로 자동 분류합니다. 
  * **맞춤형 평가셋**: 사용자는 원하는 필터로 평가 목적에 맞는 데이터만 손쉽게 추출할 수 있습니다.
  * **동적 확장성**: 새로운 벤치마크가 등장하면 자동으로 포맷을 변환·분류하여 지속 확장됩니다. 

## 2. HRET에서 BenchHub 사용하기

Evaluator에 `dataset="benchhub"`를 지정하고, `dataset_params`에 필터링 조건을 넣어 데이터를 로드합니다.

### 2.1 기본 사용법

한국어 테스트셋을 필터 없이 불러오는 최소 예제입니다.

```python
from llm_eval.evaluator import Evaluator
import pandas as pd

evaluator = Evaluator()
results = evaluator.run(
    model="openai",
    model_params={
        "api_base": "http://0.0.0.0:8000/v1/chat/completions",
        "model_name": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "batch_size": 3
    },
    dataset="benchhub",
    dataset_params={"language": "ko"},  # 한국어 데이터
    evaluation_method="string_match",
    split="test"
)
print(results)
```

### 2.2 필터링을 통한 맞춤형 평가셋 구성

`simple_tutorial.ipynb` 예제를 바탕으로, 여러 조건을 조합해 데이터셋을 구성할 수 있습니다. 
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
        "language": "ko",               # 언어
        "split": "test",               # 분할
        "benchmark_names": ["KMMLU"],   # 벤치마크 이름
        "problem_types": ["MCQA"],      # 문제 유형
        "task_types": ["knowledge"],    # 과제 유형
        "target_types": ["General"],    # 대상 유형
        "subject_types": ["tech/electrical/Electrical Eng"]
    },
    evaluation_method="string_match"
)
print(results)
```

| 필터 파라미터           | 설명                                            |
| ----------------- | --------------------------------------------- |
| `language`        | 'ko' 또는 'en'                                  |
| `benchmark_names` | 원본 벤치마크 이름 리스트                                |
| `problem_types`   | 문제 형식 (e.g., MCQA, Open-ended)                |
| `task_types`      | 능력 유형 (knowledge, reasoning, value/alignment) |
| `target_types`    | 문화적 특성 (General, Cultural)                    |
| `subject_types`   | 주제 분류 (예: science/math, tech/IT)              |

## 3. 평가 결과 및 정보 확인

* `results.metrics`로 메트릭 요약 확인
* `results.to_dataframe()`로 상세 결과 획득
* `results.info()`로 적용된 필터·구성 정보 확인

```python
print(results.metrics)
df = results.to_dataframe()
print(df.head())
info = results.info()
print(info)
```

## 4. 인용(Citation) 리포트 생성하기

BenchHub로 평가한 후, LaTeX 표와 BibTeX 인용을 자동 생성합니다.

```python
# results = evaluator.run(dataset="benchhub", ...)
try:
    results.benchhub_citation_report(output_path='benchhub_citation_report.tex')
    print("Citation report saved to benchhub_citation_report.tex")
except ValueError as e:
    print(e)
```

생성된 `benchhub_citation_report.tex` 예시:

```latex
The evaluation dataset are sampled using BenchHub~\cite{kim2025benchhub}, and the evaluation are conducted using hret~\cite{lee2025hret}.

% Table of included datasets
\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Dataset} & \textbf{Number of Samples} \\ \midrule
\cite{son-etal-2025-kmmlu} & 30499 \\
\cite{son-etal-2024-hae} & 4900 \\
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

이제 이 가이드를 참고하여 BenchHub 기반 평가 파이프라인을 손쉽게 구성해 보세요~~~
