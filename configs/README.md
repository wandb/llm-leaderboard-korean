# 실험 설정 파일 구조

## 개요

이 디렉토리에는 LLM 평가 실험을 위한 설정 파일들이 있습니다. 설정은 계층적 구조로 되어 있어 공통 설정과 모델별 특화 설정을 분리할 수 있습니다.

## 파일 구조

### `base_config.yaml`
- 모든 실험에서 공통으로 사용되는 기본 설정
- 데이터셋 활성화/비활성화 기본값
- 평가 방법, 언어 설정 등 공통 설정

### 모델별 설정 파일들
각 모델별 설정 파일은 `base_config.yaml`을 기반으로 하며, 필요한 부분만 오버라이드합니다.

- `gpt4_experiment.yaml` - GPT-4 모델 실험용
- `claude_experiment.yaml` - Claude 모델 실험용  
- `huggingface_experiment.yaml` - HuggingFace 모델 실험용
- `advanced_experiment.yaml` - Judge 모델, Scaling 등 고급 기능 활용
- `quick_test.yaml` - 빠른 테스트용 (작은 모델, 단일 데이터셋)

## 설정 병합 규칙

1. `base_config.yaml`이 먼저 로드됩니다
2. 모델별 config 파일이 로드되어 base config를 오버라이드합니다
3. 딕셔너리는 재귀적으로 병합되며, 다른 타입은 완전히 교체됩니다

예시:
```yaml
# base_config.yaml
datasets:
  haerae_bench: true
  kmmlu: true
  click: false

# gpt4_experiment.yaml
datasets:
  click: true  # false -> true로 오버라이드
  k2_eval: true  # 새로 추가

# 최종 병합 결과
datasets:
  haerae_bench: true  # base에서 유지
  kmmlu: true         # base에서 유지  
  click: true         # 오버라이드됨
  k2_eval: true       # 새로 추가
```

## 데이터셋 설정

### 기본 활성화 데이터셋 (base_config.yaml)
- `haerae_bench: true`
- `kmmlu: true`
- 나머지: `false`

### 데이터셋 목록과 자동 선택되는 평가 방법
- `haerae_bench` - HAE-RAE Bench (한국 문화, 지식 평가) → **string_match**
- `kmmlu` - Korean Massive Multitask Language Understanding → **string_match**
- `click` - CLiCK 데이터셋 → **string_match**
- `hrm8k` - HRM8K 수학 문제 → **math_eval**
- `k2_eval` - K2-Eval 데이터셋 → **llm_judge**
- `KUDGE` - KUDGE 데이터셋 → **llm_judge**
- `benchhub` - BenchHub 데이터셋 → **string_match**
- `hrc` - HRC 데이터셋 → **string_match**
- `kbl` - KBL 데이터셋 → **string_match**
- `kormedmcqa` - 한국어 의료 MCQ 데이터셋 → **string_match**
- `aime2025` - AIME 2025 수학 문제 → **math_eval**
- `generic_file` - 커스텀 파일 데이터셋 → **string_match**

### 평가 방법 자동 선택
시스템은 각 데이터셋의 특성에 맞는 최적의 평가 방법을 자동으로 선택합니다:

- **string_match**: 객관식 문제, 정확한 답변이 필요한 경우
- **math_eval**: 수학 문제, 수식의 동등성 검증이 필요한 경우  
- **llm_judge**: 생성 태스크, 주관적 판단이 필요한 경우

### 평가 방법 오버라이드

**1. 전역 오버라이드 (모든 데이터셋에 동일한 방법 적용):**
```yaml
evaluation:
  method: "string_match"  # 모든 데이터셋에 강제 적용
```

**2. 데이터셋별 개별 오버라이드:**
```yaml
dataset_specific:
  hrm8k:
    evaluation_method: "string_match"  # 기본 math_eval 대신 string_match 사용
  k2_eval:
    evaluation_method: "string_match"  # 기본 llm_judge 대신 string_match 사용
```

## 사용법

```bash
# 빠른 테스트
python run.py --config configs/quick_test.yaml

# GPT-4 실험
python run.py --config configs/gpt4_experiment.yaml

# Claude 실험  
python run.py --config configs/claude_experiment.yaml

# 고급 실험 (Judge 모델, Scaling 등)
python run.py --config configs/advanced_experiment.yaml
```

## 새 설정 파일 만들기

1. 기존 설정 파일을 복사하여 시작
2. 모델 설정 부분 수정
3. 필요한 경우 데이터셋 활성화 설정 조정
4. 데이터셋별 특별 설정 추가 (선택사항)

예시:
```yaml
# my_experiment.yaml
datasets:
  my_dataset: true  # 새 데이터셋 활성화

model:
  name: "my_model"
  params:
    model_name_or_path: "my/model/path"
```
