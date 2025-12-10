# 🐯 Horangi: 한국어 LLM 벤치마크 평가 프레임워크

**Horangi**(호랑이)는 [Inspect AI](https://inspect.aisi.org.uk/)와 [inspect-wandb](https://inspect-wandb.readthedocs.io/)를 활용한 한국어 LLM 평가 프레임워크입니다.

## ✨ 특징

- 🇰🇷 **한국어 특화 벤치마크**: QA, 추론, 지식, 상식 추론 등 다양한 한국어 평가 태스크
- 📊 **WandB/Weave 통합**: 평가 결과가 자동으로 WandB와 Weave에 기록됩니다
- 🔧 **확장 가능**: 커스텀 벤치마크, Solver, Scorer를 쉽게 추가할 수 있습니다
- 🚀 **간편한 실행**: Inspect CLI 또는 Python 스크립트로 바로 실행 가능

## 📦 설치

### 1. 기본 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/inspect-horangi.git
cd inspect-horangi

# 의존성 설치
pip install -e .

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

### 2. WandB 설정

```bash
# WandB 로그인
wandb login

# 프로젝트 초기화 (선택사항)
wandb init
```

또는 환경 변수로 설정:

```bash
export WANDB_API_KEY=your-api-key
export WANDB_PROJECT=korean-llm-benchmark
export WANDB_ENTITY=your-team-or-username
```

### 3. 모델 API 키 설정

사용할 모델에 따라 API 키를 설정하세요:

```bash
# OpenAI
export OPENAI_API_KEY=your-openai-key

# Anthropic
export ANTHROPIC_API_KEY=your-anthropic-key

# Google
export GOOGLE_API_KEY=your-google-key
```

## 🚀 사용법

### 방법 1: Inspect CLI 사용 (권장)

```bash
# 한국어 QA 벤치마크 실행
inspect eval eval_tasks.py@korean_qa --model openai/gpt-4o

# 한국어 추론 벤치마크 실행
inspect eval eval_tasks.py@korean_reasoning --model anthropic/claude-sonnet-4-0

# 모든 벤치마크 실행
inspect eval eval_tasks.py --model openai/gpt-4o

# 샘플 수 제한하여 테스트
inspect eval eval_tasks.py@korean_qa --model openai/gpt-4o --limit 5
```

### 방법 2: Python 스크립트 사용

```bash
# 특정 벤치마크 실행
python run_eval.py --model openai/gpt-4o --benchmark korean_qa

# 모든 벤치마크 실행
python run_eval.py --model openai/gpt-4o

# Chain-of-Thought 활성화
python run_eval.py --model openai/gpt-4o --benchmark korean_reasoning --cot

# WandB 프로젝트 지정
python run_eval.py --model openai/gpt-4o --wandb-project my-eval-project
```

### 방법 3: Python 코드에서 직접 사용

```python
from inspect_ai import eval
from horangi.benchmarks import korean_qa, korean_reasoning

# 단일 벤치마크 실행
task = korean_qa(use_cot=True)
results = eval(task, model="openai/gpt-4o")

# 여러 벤치마크 실행
tasks = [
    korean_qa(),
    korean_reasoning(use_cot=True),
]
results = eval(tasks, model="openai/gpt-4o")
```

## 📚 벤치마크 목록

### 한국어 QA (`korean_qa`)
- 한국어 읽기 이해 및 질의응답 능력 평가
- 지문을 읽고 질문에 답하는 형식

### 한국어 추론 (`korean_reasoning`)
- 논리적 추론 및 수리 추론 능력 평가
- 변형: `korean_math_reasoning`, `korean_logical_reasoning`

### 한국어 지식 (`korean_knowledge`)
- 한국 역사, 문화, 사회에 대한 지식 평가
- 객관식 문제 형식
- 변형: `korean_history`, `korean_culture`

### 한국어 상식 (`korean_commonsense`)
- 상식 추론 및 사회적 맥락 이해 능력 평가
- HellaSwag, WinoGrande 스타일
- 변형: `korean_hellaswag`, `korean_winogrande`

## 🔧 커스텀 벤치마크 추가

### 1. 데이터셋 준비

`src/horangi/benchmarks/data/` 디렉토리에 JSONL 형식으로 데이터셋을 추가합니다:

```jsonl
{"id": "001", "input": "질문 내용", "target": "정답"}
{"id": "002", "input": "객관식 질문\n\nA) 보기1\nB) 보기2", "target": "A", "choices": ["보기1", "보기2"]}
```

### 2. Task 정의

`src/horangi/benchmarks/` 디렉토리에 새 Task 파일을 생성합니다:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message

@task
def my_korean_benchmark():
    return Task(
        dataset=json_dataset("path/to/data.jsonl"),
        solver=[
            system_message("시스템 프롬프트"),
            generate(),
        ],
        scorer=match(),
        name="my_benchmark",
    )
```

### 3. 등록 및 실행

`eval_tasks.py`에 새 Task를 임포트하고 `__all__`에 추가합니다.

## 📊 WandB에서 결과 확인

평가가 완료되면 터미널에 WandB 링크가 출력됩니다:

```
wandb: ⭐️ View project at https://wandb.ai/YOUR_TEAM/YOUR_PROJECT
wandb: 🚀 View run at https://wandb.ai/YOUR_TEAM/YOUR_PROJECT/runs/RUN_ID
```

### WandB Models UI
- 평가 실행 기록 및 설정 확인
- 실행 간 비교

### WandB Weave UI
- 상세한 트레이스 시각화
- 개별 샘플 분석
- 필터링 및 검색

## 📁 프로젝트 구조

```
inspect-horangi/
├── README.md
├── pyproject.toml
├── requirements.txt
├── run_eval.py              # Python 실행 스크립트
├── eval_tasks.py            # Inspect CLI용 Task 정의
└── src/
    └── horangi/
        ├── __init__.py
        ├── benchmarks/
        │   ├── __init__.py
        │   ├── korean_qa.py
        │   ├── korean_reasoning.py
        │   ├── korean_knowledge.py
        │   ├── korean_commonsense.py
        │   └── data/
        │       ├── korean_qa.jsonl
        │       ├── korean_reasoning.jsonl
        │       ├── korean_knowledge.jsonl
        │       └── korean_commonsense.jsonl
        ├── solvers/
        │   ├── __init__.py
        │   └── korean_solver.py
        └── scorers/
            ├── __init__.py
            └── korean_scorer.py
```

## 🤝 기여하기

1. 이 저장소를 Fork합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/new-benchmark`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add new benchmark'`)
4. 브랜치에 Push합니다 (`git push origin feature/new-benchmark`)
5. Pull Request를 생성합니다

## 📄 라이선스

MIT License

## 🔗 참고 자료

- [Inspect AI 공식 문서](https://inspect.aisi.org.uk/)
- [inspect-wandb 문서](https://inspect-wandb.readthedocs.io/)
- [Weights & Biases 문서](https://docs.wandb.ai/)
- [W&B Weave 문서](https://wandb.me/weave)

