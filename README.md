## HAE-RAE Evaluation Toolkit (Korean LLM Leaderboard)

한국어 LLM을 표준화된 방식으로 평가하는 툴킷입니다. 다양한 벤치마크 데이터셋과 평가 방법을 통합해, 하나의 파이프라인에서 손쉽게 모델을 비교/분석할 수 있습니다. uv 기반으로 의존성을 관리하며, W&B(Weights & Biases)와 Weave를 통해 결과를 시각화합니다.

- 라이선스: Apache-2.0
- 최소 Python: 3.10+
- 의존성 관리자: uv

참고: 본 프로젝트는 설계 철학과 문서 구조에서 일부 아이디어를 다음 자료로부터 참고했습니다. [wandb/llm-leaderboard 문서](https://github.com/wandb/llm-leaderboard/tree/6e29db49588b920d8210eda1d415a1533e92a571)


### 주요 특징

- 다수 한국어 벤치마크를 한 번에 실행 및 비교
- 모델/데이터셋/스케일링/평가 모듈의 느슨한 결합과 레지스트리 기반 확장성
- W&B 및 Weave와의 긴밀한 통합 (싱글런 런 관리, 리더보드 테이블 생성)
- uv 기반 초고속 환경 세팅과 재현 가능한 의존성(uv.lock)


## 빠른 시작

1) uv 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 설치 확인
uv --version
```

2) 프로젝트 클론 및 의존성 설치

```bash
git clone https://github.com/wandb/llm-leaderboard-korean
cd llm-leaderboard-korean

# 의존성 설치
uv sync
```

3) 실행 예시

```bash
# 도움말
uv run python run_eval.py --help

# 기본 구성으로 전체 벤치마크 실행 (기본 모델셋)
uv run python run_eval.py

# 특정 모델 구성만 실행
uv run python run_eval.py --config gpt-4o-2024-11-20

# 특정 데이터셋만 실행 (예: kmmlu)
uv run python run_eval.py --dataset kmmlu
```

참고: conda 또는 venv 환경이 동시에 활성화되어 있다면 `conda deactivate` 또는 `deactivate` 후 `uv sync`/`uv run`을 사용하세요. (자세한 내용은 아래 트러블슈팅 참조)


## 구성 개요

```
llm-leaderboard-korean/
├─ configs/
│  ├─ base_config.yaml
│  └─ <model>.yaml
├─ llm_eval/
│  ├─ datasets/                # 데이터셋 로더/전처리
│  ├─ evaluation/              # 평가자(스코어러)와 레지스트리
│  ├─ models/                  # 모델 백엔드 어댑터(OpenAI, HF, LiteLLM, Multi, (옵션) vLLM)
│  ├─ scaling_methods/         # 스케일링/샘플링/프롬프트 보조(있을 경우)
│  ├─ utils/                   # 로깅/메트릭/프롬프트 템플릿 유틸리티
│  ├─ external/                # 외부 통합(HalluLens, SWE-bench 등 서드파티)
│  │  └─ providers/...
│  ├─ standard_weave_runner.py # Weave 표준 평가 어댑터(모델-스코어러 브릿지)
│  ├─ weave_evaluator.py       # Weave 컨트롤러(샘플/메트릭 수집 및 아티팩트화)
│  ├─ wandb_controller.py      # W&B 로깅 오케스트레이터(테이블/아티팩트/요약 생성)
│  ├─ wandb_singleton.py       # 단일 W&B 런 싱글턴(전역 공유 및 종료 관리)
│  └─ runner.py                # 파이프라인 러너(데이터셋-모델-스코어러 연결)
├─ run_eval.py                  # 멀티 모델/데이터셋 CLI 엔트리포인트
├─ experiment.py                # 편의용 일괄 실행 스크립트(선택)
├─ pyproject.toml               # 프로젝트 메타/의존성 정의
└─ uv.lock                      # 의존성 잠금(재현성 보장)
```

- `configs/`
  - `base_config.yaml`: W&B 프로젝트 정보, 각 데이터셋의 split/subset, 평가 메서드와 파라미터, 일부 모델 파라미터 오버라이드 정의
  - `<model>.yaml`: 모델별 실행 설정(예: provider, 모델명/릴리즈, 배치/토큰 한도, (옵션) vLLM 서버 파라미터)
- `llm_eval/`
  - `datasets/`: 각 데이터셋 로딩/샘플 선택/필드 정규화
  - `evaluation/`: `string_match`, `char_f1`, `ifeval`, `mt_bench_judge`, `swebench_eval`, `comet_score` 등 스코어러 구현과 레지스트리
  - `models/`: OpenAI/HuggingFace/LiteLLM/Multi/(옵션)VLLM 백엔드 어댑터 및 등록
  - `external/`: HalluLens, SWE-bench 등 외부 패키지/작업 흐름 연동 코드
  - `standard_weave_runner.py`: Weave 표준 평가 프레임워크를 통해 모델 호출과 스코어 계산을 규격화
  - `weave_evaluator.py`: EnhancedWeaveController 제공. 샘플/예측/메트릭을 Weave 아티팩트로 구조화하고, 리더보드에 필요한 부가 정보를 수집
  - `wandb_controller.py`: 예측/메트릭/샘플들을 W&B 테이블로 집계하고, 아티팩트/이미지/리치 텍스트를 첨부. 전반적인 리더보드 테이블 생성 유틸 포함
  - `wandb_singleton.py`: 멀티 데이터셋 실행 중에도 단일 W&B 런을 공유하도록 싱글턴 관리(초기화/구성 주입/마무리)
  - `runner.py`: 파이프라인 러너. 데이터셋 로딩 → 모델 추론(선택적 스케일링) → 평가 → 추가 후처리(언어 패널티 등) → 결과 수집
- `run_eval.py`: 여러 모델/데이터셋을 순차 실행. 싱글턴 W&B 런을 생성, 실행 후 리더보드 테이블 로깅
- `experiment.py`: 여러 구성 배치를 편하게 돌릴 수 있는 보조 스크립트(선택 사용)
- `pyproject.toml`, `uv.lock`: uv 기반 의존성 정의/잠금 파일


## 모델/실행 설정

- 기본 실행 모델 목록(변경 가능): `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`, `o4-mini-2025-04-16`, `gpt-4o-2024-11-20`, `gpt-4.1-2025-04-14`
- 주요 옵션
  - `--config <이름>`: 특정 모델 구성만 실행 (예: `--config gpt-4o-2024-11-20`)
  - `--dataset <키>`: 특정 데이터셋만 실행 (예: `--dataset kmmlu`)
  - `--use-standard-weave`: Weave 표준 평가 프레임워크 사용(기본 활성화)
- (옵션) vLLM: 모델 구성에서 `provider: hosted_vllm`을 지정하면 내부 서버 매니저가 자동 기동/종료합니다.

## 기본 평가 데이터셋 (run_eval.py 기준)

`run_eval.py`의 기본 선택 목록:

```
mt_bench, halluLens, ifeval_ko, komoral, korean_hate_speech,
mrcr_2_needles, haerae_bench_v1, squad_kor_v1, kobbq,
kmmlu, kmmlu_pro, kobalt_700, hle, arc_agi,
aie2025, hrm8k, bfcl, swebench, korean_parallel_corpora
```
베이스 설정은 `configs/base_config.yaml`에서 관리합니다. 여기에는 W&B 프로젝트 정보, 각 데이터셋의 split/subset/평가 방법 및 파라미터, 일부 데이터셋별 모델 파라미터 오버라이드 등이 포함됩니다.

설명(일부는 `configs/base_config.yaml`의 파라미터를 반영):
- mt_bench: 다영역 일반 능력 MT-Bench 한국어 세팅, LLM judge 사용
- halluLens: 한국어 환각(Hallucination) 평가 (정확/혼합/생성 엔티티)
- ifeval_ko: 지시문 준수(Instruction Following) 평가 (IFEval 한국어 변형)
- komoral: 한국어 도덕/규범 관련 질의 대응 평가
- korean_hate_speech: 혐오 표현 탐지
- mrcr_2_needles: Long-context Retrieval(needle-in-a-haystack) 변형 평가
- haerae_bench_v1: 한국어 일반 상식/독해 등 멀티 카테고리 MCQA 벤치마크
- squad_kor_v1: 한국어 SQuAD 스타일 독해, CharF1 평가
- kobbq: 한국어 BBQ(KoBBQ) 과제
- kmmlu / kmmlu_pro: 한국어 MMLU 스타일, 다과목 MCQA
- kobalt_700: 한국어 언어능력 세부영역 벤치마크(문법/의미 등)
- hle: 한국어 대규모 지식/학제간 문제셋 (카테고리 다수)
- arc_agi: ARC-AGI 스타일 추론 평가(Grid Match)
- aime2025 / hrm8k / math: 수학 추론 평가 (최종 답 추출)
- bfcl: 함수 호출/멀티턴/라이브 호출 등 코드 실행/에이전트 성 평가
- swebench: 코드 수정/테스트 기반 SWE-bench Verified (원격 API 실행)
- korean_parallel_corpora: 번역 품질 평가(COMET score)

특정 데이터셋만 실행하려면 `--dataset <이름>`을 사용하세요.


## 환경 변수 / API 키

`.env`에 다음 값을 설정해 두면 uv가 자동 로드합니다(uv run/sync 시 dotenv 지원).

필수(권장):

| 변수 | 용도 | 예시 |
|---|---|---|
| OPENAI_API_KEY | OpenAI(gpt-4o, gpt-4.1 등) API 사용 | sk-... |
| WANDB_API_KEY | Weights & Biases 로깅 | wandb api-key 문자열 |

선택(모델/프로바이더별 필요 시):

| 변수 | 용도 | 언제 필요한가 | 예시/비고 |
|---|---|---|---|
| ANTHROPIC_API_KEY | Anthropic(Claude) | `configs/*`에 provider: anthropic | `claude-sonnet...` 실행 시 |
| GOOGLE_API_KEY | Google Gemini | provider: google | Gemini 2.5 등 |
| COHERE_API_KEY | Cohere | BFCL/외부 태스크에서 Cohere 핸들러 사용 시 | - |
| TOGETHER_API_KEY | Together API | HalluLens 등 together 경로 사용 시 | llm_eval/external/providers/hallulens/* |
| FIREWORKS_API_KEY | Fireworks | Fireworks 핸들러 사용 시 | - |
| HUGGINGFACE_HUB_TOKEN | HuggingFace Hub | 비공개 모델/데이터 받기 | `hf_xxx...` |
| AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, (옵션)AWS_SESSION_TOKEN | AWS Bedrock | provider: bedrock 사용 시 | 또는 `AWS_PROFILE`/`AWS_SSO_PROFILE_NAME` |
| AWS_BEDROCK_REGION 또는 AWS_DEFAULT_REGION | Bedrock 리전 | Bedrock 사용 시 | 기본 us-east-1 |
| WANDB_ENTITY, WANDB_PROJECT | W&B 엔티티/프로젝트 강제 지정 | 기본은 `configs/base_config.yaml`의 값 사용 | 필요 시 override |
| WANDB_BFCL_PROJECT | BFCL 전용 로깅(ENTITY:PROJECT) | BFCL 실행 후 표/CSV 아티팩트 업로드 | `team:proj` |

메모:
- 기본 엔티티/프로젝트는 `configs/base_config.yaml`의 `wandb.params.entity/project`를 따릅니다. 키만 있으면 자동 연결됩니다.
- Weave 로깅은 내부적으로 W&B를 사용하므로 별도 `WEAVE_API_KEY`가 필요하지 않습니다(기본 설정 기준).


## vLLM 사용 (선택)

`pyproject.toml`의 `optional-dependencies`로 분리되어 있습니다. 필요한 경우에만 설치하세요.

```bash
uv sync --extra vllm
```

모델 구성(`configs/*-vllm*.yaml`)을 사용하면 vLLM 서버를 자동 기동/종료하도록 설정할 수 있습니다(예: `provider: hosted_vllm`).



## 참고 자료

- 문서 아이디어/구성 일부: [wandb/llm-leaderboard](https://github.com/wandb/llm-leaderboard/tree/6e29db49588b920d8210eda1d415a1533e92a571)


## 데이터셋 택소노미

| 데이터셋 | 태스크 | 서브셋 | 메트릭 |
|---|---|---|---|
| mt_bench | 일반 능력(다영역) 대화 평가 (LLM judge) | roleplay, humanities, writing, reasoning, coding | mt_bench_judge |
| halluLens | 환각/거부 판단 | precise_wikiqa, mixed_entities, generated_entities | 모델 심판 기반(아웃풋 판정) |
| ifeval_ko | 지시문 준수 | default | ifeval_strict |
| komoral | 도덕/규범 판단 | default | string_match |
| korean_hate_speech | 혐오 표현 탐지 | default | string_match |
| mrcr_2_needles | Long-context Retrieval | 128k | sequence_match |
| haerae_bench_v1 | MCQA(일반 상식/독해) | standard_nomenclature, loan_words, rare_words, general_knowledge, history, reading_comprehension | string_match (mcqa) |
| squad_kor_v1 | 독해 QA (SQuAD) | default | char_f1 |
| kobbq | 한국어 BBQ | default | string_match |
| kmmlu | 한국어 MMLU 스타일 MCQA | (카테고리 기반) | string_match (mcqa) |
| kmmlu_pro | 한국어 MMLU 스타일 MCQA(확장) | (카테고리 기반) | string_match (mcqa) |
| kobalt_700 | 한국어 언어능력(문법/의미 등) | Syntax, Semantics | string_match (mcqa) |
| hle | 학제간 지식·문제 풀이 | Other, Humanities/Social Science, Math, Physics, Computer Science/AI, Biology/Medicine, Chemistry, Engineering | string_match |
| arc_agi | 추론(ARC-AGI 스타일) | default | grid_match |
| aime2025 | 수학 추론 (최종 답 추출) | AIME2025-I, AIME2025-II | math_match |
| hrm8k | 수학/수리적 추론 | GSM8K, KSM, MATH, MMMLU, OMNI_MATH | math_match |
| bfcl | 에이전트/코드 실행형 평가 | simple_python, simple_java, simple_javascript, multiple, irrelevance, live_*, multi_turn_* | 시나리오 성공률 |
| swebench (Verified) | 코드 패치 생성/수리(테스트 기반) | default (Verified 80) | swebench |
| korean_parallel_corpora | 번역 품질(한↔영) | e2k, k2e | comet_score |

참고: 내부 기본 설정과 분류는 `configs/base_config.yaml`를 기준으로 하며, 상세 정의·운영 사례는 필요에 따라 변경될 수 있습니다. 추가 분류와 운영 예시는 아래 참고 페이지를 확인하세요: [W&B 리더보드 문서/가이드](https://github.com/wandb/llm-leaderboard/tree/6e29db49588b920d8210eda1d415a1533e92a571), [horangi 리더보드 링크](https://api.wandb.ai/links/horangi/cpqvrm32).

## 라이선스

Apache-2.0


