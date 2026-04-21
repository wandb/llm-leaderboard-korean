# 새 모델 추가 가이드

> 새 모델을 Horangi로 평가하는 방법을 설명합니다.
> 설치·환경변수 설정은 [루트 README](../README.md#-설치)를 먼저 참고하세요.

---

## 📋 Step 1: 모델 정보 수집

평가할 모델의 [HuggingFace 모델 카드](https://huggingface.co/models) 또는 공식 API 문서에서 다음을 확인합니다.


| 항목                        | 예시                                                | 메모                                   |
| ------------------------- | ------------------------------------------------- | ------------------------------------ |
| 모델 ID                     | `LGAI-EXAONE/EXAONE-4.0-32B`, `gpt-4o-2024-11-20` | API 모델은 provider가 부여한 이름             |
| 파라미터 수 / MoE 여부           | 32B / MoE면 active params 별도                       | size_category 산정에 사용                 |
| Context window            | 32k, 128k 등                                       | `max_model_len`/`context_window`에 반영 |
| Reasoning(thinking) 모델 여부 | `enable_thinking`, `reasoning_effort`             | 해당 시 타임아웃·토큰 수 크게 설정                 |
| Tool calling 지원           | `hermes`, `llama3_json` 등 parser                  | BFCL 평가에 필요                          |
| 출시일                       | `2025-07-15`                                      | 리더보드 표시용                             |
| 커스텀 fork 요구               | 예: `transformers` / `vllm` 특정 브랜치                 | README_swebench_ko의 트러블슈팅 참고         |


---

## 🎯 Step 2: 템플릿 선택


| 상황                                                  | 템플릿      | 파일                                                                            |
| --------------------------------------------------- | -------- | ----------------------------------------------------------------------------- |
| OpenAI/Anthropic/Google/xAI/OpenRouter 등 **API 모델** | API 템플릿  | `[configs/models/_template_api.yaml](../configs/models/_template_api.yaml)`   |
| 로컬 GPU에서 **vLLM으로 서빙**하는 오픈소스 모델                    | vLLM 템플릿 | `[configs/models/_template_vllm.yaml](../configs/models/_template_vllm.yaml)` |


> vLLM 템플릿을 쓰면 `run_eval.py` 실행 시 vLLM 서버가 **자동으로 시작·종료**됩니다. 별도로 서버를 띄우지 않아도 됩니다.

```bash
# API 모델
cp configs/models/_template_api.yaml configs/models/my-model.yaml

# 로컬 vLLM 모델
cp configs/models/_template_vllm.yaml configs/models/my-model.yaml
```

config 파일명이 `--config` 인자로 그대로 쓰입니다. 예: `configs/models/my-model.yaml` → `--config my-model`.

---

## 🔧 Step 3: Config 편집

### 공통 블록

```yaml
wandb:
  run_name: "My-Model-v1"          # W&B run과 리더보드에 표시될 이름

metadata:
  release_date: "2026-04-01"
  size_category: "Large (30B<)"    # "Small (<10B)" | "Medium (10B-30B)" | "Large (30B<)"
  model_size: 32000000000          # 총 파라미터 수
  active_params: 32000000000       # MoE면 active, dense면 model_size와 동일
  context_window: 32768
```

### API 모델 전용

```yaml
model:
  name: claude-opus-4-5-20251101   # provider가 부여한 정확한 이름
  client: litellm                  # openai | litellm
  provider: anthropic              # anthropic | openai | google | xai | openrouter | ...
  api_key_env: ANTHROPIC_API_KEY   # .env 에 존재해야 하는 키 이름

  params:
    max_tokens: 4096
    temperature: 0.6
    top_p: 0.95
    timeout: 7200                  # Reasoning 모델은 크게
    max_retries: 2
    max_connections: 30            # 동시 요청 수

    # 리즈닝 모델은 아래 중 해당하는 한 가지를 추가:
    # reasoning_effort: high       # OpenAI reasoning (low/medium/high/xhigh)
    # effort: high                 # Anthropic
    # extra_body:                  # OpenRouter thinking
    #   reasoning:
    #     enabled: true
```


| Provider      | `client`  | `api_key_env` 예시     |
| ------------- | --------- | -------------------- |
| OpenAI (직접)   | `openai`  | `OPENAI_API_KEY`     |
| Anthropic     | `litellm` | `ANTHROPIC_API_KEY`  |
| Google Gemini | `litellm` | `GEMINI_API_KEY`     |
| xAI Grok      | `litellm` | `XAI_API_KEY`        |
| OpenRouter    | `litellm` | `OPENROUTER_API_KEY` |


### vLLM 모델 전용

```yaml
vllm:
  model_path: "LGAI-EXAONE/EXAONE-4.0-32B"   # HuggingFace ID 또는 로컬 경로
  tensor_parallel_size: 2                    # 사용할 GPU 수
  port: 8000
  host: "0.0.0.0"
  max_model_len: 32768
  trust_remote_code: true
  served_model_name: "exaone-4.0-32b"        # model.name 과 일치시킬 것

  # Tool calling을 지원하는 모델이면:
  enable_auto_tool_choice: true
  tool_call_parser: "hermes"                 # hermes | llama3_json | ...

  # Thinking 모델이면:
  reasoning_parser: "deepseek_r1"

model:
  name: exaone-4.0-32b
  client: litellm
  provider: hosted_vllm
  api_key_env: HOSTED_VLLM_API_KEY

  params:
    max_tokens: 16384
    temperature: 0.6
    top_p: 0.95
    timeout: 7200
    max_retries: 2
    max_connections: 30

    # thinking 계열:
    # extra_body:
    #   chat_template_kwargs:
    #     enable_thinking: true
```

### 벤치마크별 override (공통)

특정 벤치마크에서만 파라미터를 바꾸고 싶을 때 사용합니다.

```yaml
benchmarks:
  bfcl:
    use_native_tools: false              # 오픈소스 모델은 text-based 권장, API 모델은 true 가능
  swebench_verified_official_80:
    max_tokens: 64000                    # 코드 생성은 출력이 길어야 함
  ko_arc_agi:
    extra_body:
      repetition_penalty: 1.05           # 반복 억제
```

---

## ▶️ Step 4: 실행

```bash
# 먼저 한 벤치마크만 소량으로 smoke test
uv run python run_eval.py --config my-model --only kmmlu --limit 5

# OK면 전체 실행
uv run python run_eval.py --config my-model

# 일부만 다시 돌리고 싶을 때
uv run python run_eval.py --config my-model --only bfcl,kobbq

# 중단된 run 이어가기
uv run python run_eval.py --config my-model --resume <wandb_run_id>
```

SWE-bench 를 평가하려면 별도 서버가 필요합니다. [SWE-bench 가이드](./README_swebench_ko.md)를 참고하세요.

---

## ✅ 체크리스트

리더보드 등재 신청 전에 다음을 확인하세요.

- Smoke test (`--limit 5 --only kmmlu`) 가 성공하고 Weave URL이 출력되었는가
- `metadata.release_date`, `size_category`, `model_size` 가 정확한가
- API 모델이면 `api_key_env` 가 `.env`에 실제로 존재하는가
- vLLM 모델이면 `served_model_name` 과 `model.name` 이 일치하는가
- Reasoning 모델이면 `timeout` ≥ 7200, `max_retries` ≥ 5 로 설정했는가
- BFCL 에서 `use_native_tools` 가 모델 특성에 맞게 설정됐는가

---

## 📚 관련 문서

- [새 벤치마크 추가](./README_benchmark_ko.md)
- [SWE-bench 서버 설정](./README_swebench_ko.md)
- [Weave 결과 분석](./README_weave_ko.md)

