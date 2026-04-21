# 🐯 Horangi - 한국어 LLM 벤치마크 평가 프레임워크

**호랑이(Horangi)** 는 한국어 LLM의 성능을 종합적으로 평가하는 오픈소스 벤치마크 프레임워크입니다.

범용언어성능(GLP)과 가치정렬성능(ALT) 두 축으로 한국어 LLM을 평가하며, 이를 위해 표준화된 벤치마크 데이터셋과 평가 파이프라인을 제공합니다.
- 📦 20개 이상의 한국어 벤치마크가 [W&B Weave](https://wandb.ai/horangi/horangi4/weave/objects)에 등록되어 있어, 별도의 데이터 준비 없이 바로 평가를 시작할 수 있습니다.
  - 새로운 벤치마크를 추가할 수 있습니다. 자세한 내용은 [새 벤치마크 추가 가이드](./docs/README_benchmark_ko.md)를 참고하세요.
- 🔓 OpenAI, Anthropic, Google 등 API 모델은 물론, vLLM 등으로 서빙하는 오픈소스 모델까지 동일한 기준으로 평가할 수 있습니다.
- 📊 평가 결과는 Weights & Biases 플랫폼에 자동으로 기록되어 샘플별 분석, 모델 간 비교, 리더보드 생성이 가능합니다.
- 🏆 [**호랑이 리더보드**](https://horangi.ai)에서 W&B가 운영하는 공식 리더보드를 확인할 수 있습니다.
  - W&B Models로 평가 실행을 관리하고, Weave로 결과를 추적하여 **완전 자동화된 리더보드**를 제공합니다.
  - 새 모델 평가 시 리더보드가 자동으로 업데이트되어 항상 최신 결과를 반영합니다.

### 📬 문의

| | |
|---|---|
| 리더보드 등재 신청 | [신청 폼](https://docs.google.com/forms/d/e/1FAIpQLSdQERNX8jCEuqzUiodjnUdAI7JRCemy5sgmVylio-u0DRb9Xw/viewform) |
| 엔터프라이즈 도입 문의 | contact-kr@wandb.com |

---

## 📋 목차

- [특징](#-특징)
- [결과 확인](#-결과-확인)
- [지원 벤치마크](#-지원-벤치마크)
- [프로젝트 구조](#-프로젝트-구조)
- [설치](#-설치)
- [빠른 시작](#-빠른-시작)
- [설정 가이드](#️-설정-가이드)
- [SWE-bench 평가 (코드 생성)](#-swe-bench-평가-코드-생성)

---
## ✨ 특징

- 🇰🇷 **20여개 한국어 벤치마크** 지원
- 📊 **WandB/Weave 자동 로깅** - 실험 추적 및 결과 비교
- 🚀 **다양한 모델 지원** - OpenAI, Claude, Gemini, Solar, EXAONE 등
- 📈 **리더보드 자동 생성** - Weave UI에서 모델 비교

### 📈 결과 확인

평가 완료 후 출력되는 Weave URL에서 상세 결과를 확인할 수 있으며, Models workspace에서 종합 평가 결과 테이블을 확인할 수 있습니다.
자세한 내용은 [Weave 가이드](./docs/README_weave_ko.md)를 참고하세요.
- **샘플별 점수 및 응답**
- **모델 간 비교**
- **집계 메트릭**
- **자동 리더보드 생성**

![Workspace](./docs/assets/workspace.png)

---

## 📊 지원 벤치마크

### 범용언어성능 (GLP) - General Language Performance

언어 이해, 지식, 추론, 코딩, 함수호출 등 일반적인 언어 모델 능력을 평가합니다.

| 평가 영역 | 벤치마크 | 설명 | 샘플 수 | 출처 |
|----------|----------|------|--------:|------|
| **구문해석** | `ko_balt_700_syntax` | 문장 구조 분석, 문법적 타당성 평가 | 100 | [snunlp/KoBALT-700](https://huggingface.co/datasets/snunlp/KoBALT-700) |
| **의미해석** | `ko_balt_700_semantic` | 문맥 기반 추론, 의미적 일관성 평가 | 100 | [snunlp/KoBALT-700](https://huggingface.co/datasets/snunlp/KoBALT-700) |
| | `haerae_bench_v1_rc` | 독해 기반 의미 해석력 평가 | 100 | [HAERAE-HUB/HAE_RAE_BENCH_1.0](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.0) |
| **표현** | `ko_mtbench` | 글쓰기, 역할극, 인문학적 표현력 (LLM Judge) | 80 | [LGAI-EXAONE/KoMT-Bench](https://huggingface.co/datasets/LGAI-EXAONE/KoMT-Bench) |
| **정보검색** | `squad_kor_v1` | 질의응답 기반 정보검색 능력 | 100 | [KorQuAD/squad_kor_v1](https://huggingface.co/datasets/KorQuAD/squad_kor_v1) |
| **일반지식** | `kmmlu` | 상식, STEM 기초학문 이해도 | 100 | [HAERAE-HUB/KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) |
| | `haerae_bench_v1_wo_rc` | 멀티턴 질의응답 기반 지식 평가 | 100 | [HAERAE-HUB/HAE_RAE_BENCH_1.0](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.0) |
| **전문지식** | `kmmlu_pro` | 의학, 법률, 공학 등 고난도 전문지식 | 100 | [LGAI-EXAONE/KMMLU-Pro](https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro) |
| | `ko_hle` | 한국어 고난도 전문가 수준 문제 | 100 | [cais/hle](https://huggingface.co/datasets/cais/hle) + 자체 번역 |
| **상식추론** | `ko_hellaswag` | 문장 완성, 다음 문장 예측 | 100 | [davidkim205/ko_hellaswag](https://huggingface.co/datasets/davidkim205/ko_hellaswag) |
| **수학추론** | `hrm8k` | 한국어 수학 추론 (GSM8K, KSM, MATH, MMMLU, OMNI_MATH 통합) | 100 | [HAERAE-HUB/HRM8K](https://huggingface.co/datasets/HAERAE-HUB/HRM8K) |
| | `ko_aime2025` | AIME 2025 고난도 수학 | 30 | [allganize/AIME2025-ko](https://huggingface.co/datasets/allganize/AIME2025-ko) |
| **추상추론** | `ko_arc_agi` | 시각적/구조적 추론, 추상적 문제 해결 | 100 | [ARC-AGI](https://arcprize.org/) |
| **코딩** | `swebench_verified_official_80` | GitHub 이슈 해결 능력 | 80 | [SWE-bench](https://www.swebench.com/) |
| | `humaneval_100` | Python 코드 생성 (HumanEval) | 100 | [openai/human-eval](https://github.com/openai/human-eval) |
| | `bigcodebench_100` | 복합 코딩 문제 해결 | 100 | [bigcode-project/bigcodebench](https://github.com/bigcode-project/bigcodebench) |
| **함수호출** | `bfcl` | 함수 호출 정확성 (단일, 멀티턴, 무관계검출) | 258 | [BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) |

### 가치정렬성능 (ALT) - Alignment Performance

제어성, 윤리, 유해성/편향성 방지, 환각 방지 등 모델의 안전성과 정렬 수준을 평가합니다.

| 평가 영역 | 벤치마크 | 설명 | 샘플 수 | 출처 |
|----------|----------|------|--------:|------|
| **제어성** | `ifeval_ko` | 지시문 수행, 명령 준수 능력 | 100 | [allganize/IFEval-Ko](https://huggingface.co/datasets/allganize/IFEval-Ko) |
| **윤리/도덕** | `ko_moral` | 사회 규범 준수, 안전한 언어 생성 | 100 | [AI Hub 윤리 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=558) |
| **유해성방지** | `korean_hate_speech` | 혐오발언, 공격적 발화 탐지 및 억제 | 100 | [kocohub/korean-hate-speech](https://github.com/kocohub/korean-hate-speech) |
| **편향성방지** | `kobbq` | 특정 집단/속성에 대한 편향성 평가 | 100 | [naver-ai/kobbq](https://huggingface.co/datasets/naver-ai/kobbq) |
| **환각방지** | `ko_truthful_qa` | 사실성 검증, 근거 기반 답변 생성 | 100 | 자체 번역 |
| | `ko_hallulens_wikiqa` | Wikipedia QA 기반 환각 평가 | 100 | [facebookresearch/HalluLens](https://github.com/facebookresearch/HalluLens) + 자체 번역 |
| | `ko_hallulens_longwiki` | 긴 문맥 Wikipedia 환각 평가 | 100 | [facebookresearch/HalluLens](https://github.com/facebookresearch/HalluLens) + 자체 번역 |
| | `ko_hallulens_nonexistent` | 가상 엔티티 거부 능력 평가 | 100 | [facebookresearch/HalluLens](https://github.com/facebookresearch/HalluLens) + 자체 번역 |


<details>
<summary>📦 데이터셋 참조 (Weave)</summary>

데이터셋은 `horangi/horangi4` 프로젝트에 업로드되어 있습니다:

| 데이터셋 | Weave Ref |
|----------|-----------|
| KoHellaSwag_mini | `weave:///horangi/horangi4/object/KoHellaSwag_mini:latest` |
| KoAIME2025_mini | `weave:///horangi/horangi4/object/KoAIME2025_mini:latest` |
| IFEval_Ko_mini | `weave:///horangi/horangi4/object/IFEval_Ko_mini:latest` |
| HAERAE_Bench_v1_mini | `weave:///horangi/horangi4/object/HAERAE_Bench_v1_mini:latest` |
| KoBALT_700_mini | `weave:///horangi/horangi4/object/KoBALT_700_mini:latest` |
| KMMLU_mini | `weave:///horangi/horangi4/object/KMMLU_mini:latest` |
| KMMLU_Pro_mini | `weave:///horangi/horangi4/object/KMMLU_Pro_mini:latest` |
| SQuAD_Kor_v1_mini | `weave:///horangi/horangi4/object/SQuAD_Kor_v1_mini:latest` |
| KoTruthfulQA_mini | `weave:///horangi/horangi4/object/KoTruthfulQA_mini:latest` |
| KoMoral_mini | `weave:///horangi/horangi4/object/KoMoral_mini:latest` |
| KoARC_AGI_mini | `weave:///horangi/horangi4/object/KoARC_AGI_mini:latest` |
| HRM8K_mini | `weave:///horangi/horangi4/object/HRM8K_mini:latest` |
| KoreanHateSpeech_mini | `weave:///horangi/horangi4/object/KoreanHateSpeech_mini:latest` |
| KoBBQ_mini | `weave:///horangi/horangi4/object/KoBBQ_mini:latest` |
| KoHLE_mini | `weave:///horangi/horangi4/object/KoHLE_mini:latest` |
| KoHalluLens_WikiQA_mini | `weave:///horangi/horangi4/object/KoHalluLens_WikiQA_mini:latest` |
| KoHalluLens_LongWiki_mini | `weave:///horangi/horangi4/object/KoHalluLens_LongWiki_mini:latest` |
| KoHalluLens_NonExistent_mini | `weave:///horangi/horangi4/object/KoHalluLens_NonExistent_mini:latest` |
| BFCL_mini | `weave:///horangi/horangi4/object/BFCL_mini:latest` |
| KoMTBench_mini | `weave:///horangi/horangi4/object/KoMTBench_mini:latest` |
| SWEBench_Verified_80_mini | `weave:///horangi/horangi4/object/SWEBench_Verified_80_mini:latest` |

</details>

---


## 📁 프로젝트 구조

```
horangi/
├── run_eval.py             # 평가 실행 스크립트
├── configs/
│   ├── base_config.yaml    # 전역 기본 설정
│   └── models/             # 모델 설정 파일
├── src/
│   ├── benchmarks/
│   │   └── horangi.py      # @task 함수 정의 (벤치마크 진입점)
│   ├── core/               # 핵심 로직
│   ├── scorers/            # 커스텀 Scorer
│   └── solvers/            # 커스텀 Solver
└── logs/                   # 평가 로그
```

> 📖 확장 가이드:
> - 새 모델 추가 → [docs/README_models_ko.md](docs/README_models_ko.md)
> - 새 벤치마크 추가 → [docs/README_benchmark_ko.md](docs/README_benchmark_ko.md)

---


## 📦 설치

### 요구 사항

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (권장) 또는 pip

### 설치 방법

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 저장소 클론
git clone https://github.com/wandb/llm-leaderboard-korean.git
cd llm-leaderboard-korean

# 의존성 설치
uv sync
```

---

## 🚀 빠른 시작

**0 → 첫 평가 결과까지 5분**. 아래 4단계를 순서대로 따라오면 됩니다.

### 1. `.env` 파일 생성

```bash
cp .env.sample .env
```

필수 3종(W&B)은 반드시 채워야 합니다. Horangi는 결과를 W&B Models + Weave로만 기록하므로, 셋 중 하나라도 없으면 실행이 중단됩니다.

```bash
# 필수
WANDB_API_KEY=...        # https://wandb.ai/authorize
WANDB_ENTITY=your-entity
WANDB_PROJECT=your-project

# 평가할 모델에 해당하는 API 키만 채우면 됩니다
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
# ...
```

> `WANDB_MODE=offline|disabled|dryrun` 은 지원하지 않습니다.

### 2. 평가할 모델 선택

`configs/models/` 디렉토리에 있는 YAML 파일 이름(확장자 제외)이 `--config` 값입니다.

```bash
ls configs/models/
# claude-opus-4-5-20251101_high-effort.yaml
# gpt-4o.yaml
# ...
```

리포지토리에 없는 모델을 추가하려면 → [새 모델 추가 가이드](./docs/README_models_ko.md)

### 3. Smoke test (소량 실행)

처음엔 한 벤치마크를 5 샘플만 돌려서 설정이 맞는지 확인합니다.

```bash
uv run python run_eval.py --config gpt-4o --only kmmlu --limit 5
```

실행 직후 W&B run URL과 Weave URL이 출력되면 성공입니다. 링크를 눌러 기록이 실제로 쌓였는지 확인하세요.

### 4. 전체 평가

```bash
uv run python run_eval.py --config gpt-4o
```

전체 벤치마크를 순차 실행하고, 완료 시 W&B Models에 요약 테이블이, Weave에 샘플별 트레이스와 리더보드가 업로드됩니다.

---

### 자주 쓰는 옵션

| 옵션 | 설명 | 예시 |
|---|---|---|
| `--config` | 모델 설정 파일명 (필수) | `--config gpt-4o` |
| `--only` | 특정 벤치마크만 실행 (쉼표 구분) | `--only kmmlu,kobbq` |
| `--limit` | 벤치마크당 샘플 수 제한 | `--limit 10` |
| `--resume` | 중단된 W&B run 이어가기 | `--resume abc123xy` |
| `--tag` | W&B 태그 추가 (여러 번 가능) | `--tag exp1 --tag test` |
| `--log-dir` | inspect_ai 로그 디렉토리 | `--log-dir /tmp/my_logs` |

### 동작 요약

- vLLM 모델은 평가 시작 시 서버가 **자동 기동**되고 끝나면 종료됩니다.
- 각 벤치마크 결과는 **실시간**으로 W&B에 기록됩니다.
- 평가 완료 시 **Weave Leaderboard** 가 자동으로 업데이트됩니다.

---

## ⚙️ 설정 가이드

작업 유형별 상세 가이드는 아래 문서에 있습니다.

| 하려는 것 | 문서 |
|---|---|
| 새 모델을 추가하고 평가하기 | [새 모델 추가 가이드](./docs/README_models_ko.md) |
| 새 벤치마크를 추가하기 | [새 벤치마크 추가 가이드](./docs/README_benchmark_ko.md) |
| SWE-bench 평가 환경 구축 | [SWE-bench 가이드](./docs/README_swebench_ko.md) |
| Weave에서 결과 보고 분석하기 | [Weave 가이드](./docs/README_weave_ko.md) |

프로젝트 레이아웃:

```
configs/
├── base_config.yaml          # 전역 기본 설정 (벤치마크 공통)
└── models/
    ├── _template_api.yaml    # API 모델 템플릿
    ├── _template_vllm.yaml   # vLLM 모델 템플릿
    └── <model-name>.yaml     # --config <model-name> 으로 지정
```

---

## 🔧 SWE-bench 평가 (코드 생성)

SWE-bench는 실제 오픈소스 프로젝트의 버그 수정 능력을 평가하는 벤치마크입니다.

📖 **자세한 설정 가이드**: [docs/README_swebench_ko.md](docs/README_swebench_ko.md)

### 빠른 시작

```bash
# 1. 서버 실행 (Docker가 있는 Linux 환경)
uv run python src/server/swebench_server.py --host 0.0.0.0 --port 8000

# 2. 클라이언트 설정 (macOS 등)
export SWE_SERVER_URL=http://YOUR_SERVER:8000

# 3. 평가 실행
uv run python run_eval.py --config gpt-4o --only swebench_verified_official_80 --limit 5
```

---

## 📚 참고 자료
- [Horangi 리더보드](https://horangi.ai)
- [W&B Models](https://wandb.ai/site/models)
- [W&B Weave](https://wandb.ai/site/weave)


