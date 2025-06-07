# Advanced Backend Usages

이 튜토리얼에서는 LiteLLM과 OpenAI-compatible API를 사용하여 Haerae Evaluation Toolkit을 활용하는 고급 방법을 살펴보겠습니다.

## 1. LiteLLM 통합

### 1.1 LiteLLM이란?
- **LiteLLM**은 다양한 LLM 공급자(OpenAI, Azure, Anthropic, Claude, Cohere 등)의 API를 통합 인터페이스로 제공해주는 라이브러리입니다.
- 단일 인터페이스로 30개 이상의 LLM 제공업체에 접근할 수 있어 모델 간 전환이 용이합니다.
- Haerae는 LiteLLM을 통해 다양한 모델에 쉽게 접근할 수 있도록 지원합니다.

### 1.2 LiteLLM 설정 및 사용방법

```python
from llm_eval.evaluator import Evaluator

# 1) Evaluator 인스턴스 생성
evaluator = Evaluator()

# 2) LiteLLM을 통한 모델 평가
results = evaluator.run(
    # LiteLLM 백엔드 지정
    model="litellm",
    
    # LiteLLM 파라미터 설정
    model_params={
        "model_name": "gpt-4", # OpenAI 모델명
        # 혹은 "anthropic/claude-3-opus-20240229"와 같이 다른 제공업체 모델
        "api_key": "your-api-key-here",
        "max_tokens": 512,
        "temperature": 0.7
    },
    
    dataset="haerae_bench",
    subset=["csat_math"],
    split="test",
    
    evaluation_method="string_match"
)

print(results)
df = results.to_dataframe()
print(df)
```

## 2. OpenAI-Compatible API 활용

### 2.1 OpenAI-Compatible API란?
- 다양한 LLM 서비스들이 OpenAI의 API 인터페이스와 호환되는 엔드포인트를 제공하고 있습니다.
- 이를 통해 OpenAI 클라이언트를 사용하는 코드를 수정 없이 다른 호환 서비스로 전환할 수 있습니다.
- Haerae는 이러한 OpenAI 호환 엔드포인트에 접근할 수 있는 방법을 제공합니다.

### 2.2 OpenAI-Compatible API 사용하기

```python
from llm_eval.evaluator import Evaluator

# 1) Evaluator 인스턴스 생성
evaluator = Evaluator()

# 2) OpenAI-Compatible API를 통한 평가
results = evaluator.run(
    # OpenAI 백엔드 지정
    model="openai",
    
    # OpenAI-Compatible API 파라미터
    model_params={
        "model_name": "text-davinci-003",  # 모델명
        "api_key": "your-api-key-here",
        "api_base": "https://your-custom-endpoint.com/v1",  # 커스텀 엔드포인트
        "max_tokens": 256,
        "temperature": 0.2
    },
    
    dataset="haerae_bench",
    subset=["csat_eng"],
    split="test",
    
    evaluation_method="string_match"
)

print(results)
df = results.to_dataframe()
print(df)
```

### 2.3 자체 호스팅된 OpenAI-Compatible 서버 활용

자체 호스팅된 LLM 서버(예: vLLM, FastChat, text-generation-inference 등)를 사용하는 경우:

```python
# 자체 호스팅된 OpenAI 호환 서버 연결
results = evaluator.run(
    model="openai",
    model_params={
        "model_name": "local-llama-2-13b",  # 로컬 모델명
        "api_base": "http://localhost:8000/v1",  # 로컬 서버 주소
        "api_key": "not-needed",  # 자체 호스팅 서버는 API 키가 필요 없을 수 있음
        "max_tokens": 512
    },
    
    dataset="haerae_bench",
    subset=["csat_kor"],
    split="test",
    
    evaluation_method="string_match"
)
```
