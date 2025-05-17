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

### 1.3 여러 모델 비교하기 (On Develop!)

LiteLLM의 장점은 동일한 인터페이스로 여러 모델을 쉽게 비교할 수 있다는 것입니다.

```python
# 여러 모델 설정
models = [
    {
        "model": "litellm",
        "model_params": {
            "model_name": "gpt-4",
            "api_key": "your-openai-api-key"
        },
        "name": "GPT-4"  # 결과 비교를 위한 별칭
    },
    {
        "model": "litellm",
        "model_params": {
            "model_name": "anthropic/claude-3-opus-20240229",
            "api_key": "your-anthropic-api-key"
        },
        "name": "Claude-3"  # 결과 비교를 위한 별칭
    }
]

# 모델 비교 평가 실행
comparison_results = evaluator.compare(
    models=models,
    dataset="haerae_bench",
    subset=["csat_math", "csat_geo"],
    split="test",
    evaluation_method="string_match"
)

print(comparison_results)
print(comparison_results.to_dataframe())
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

## 3. HRET Agents 활용하기 (On Develop!)

### 3.1 HRET Agents란?
- **HRET Agents**는 Haerae Evaluation Toolkit을 위한 데이터셋 준비를 자동화하는 도구입니다.
- Hugging Face의 데이터셋을 한국어로 번역하고, HRET에서 사용 가능한 데이터셋 모듈을 자동 생성합니다.
- `smolagents` 프레임워크를 기반으로 각 기능이 모듈화되어 있어 확장성이 뛰어납니다.

### 3.2 HRET Agents 설치 및 환경 설정

```bash
# HRET Agents 저장소 클론
git clone https://github.com/HAE-RAE/hret-agents.git
cd hret-agents

# 필요한 패키지 설치
pip install -r requirements.txt
```

`config/config.py` 파일에서 API 키와 설정을 구성합니다:

```python
# config/config.py 예시
OPENAI_API_KEY = "your-openai-api-key-here"
HF_TOKEN = "your-huggingface-token-here"
BATCH_SIZE = 10  # 번역 배치 크기
MAX_RETRIES = 3  # 오류 시 재시도 횟수
```

### 3.3 HRET Agents 기본 사용법

```bash
# 기본 사용법: Hugging Face 데이터셋을 변환하여 로컬에 저장
python src/main.py --dataset "HAERAE-HUB/QARV" --subset "your_subset" --split train

# Hugging Face Hub에 자동으로 업로드하는 경우
python src/main.py --dataset "HAERAE-HUB/QARV" --subset "your_subset" --split train --push
```

Python 스크립트에서 직접 사용하는 방법:

```python
from hret_agents.agent import DatasetTranslationAgent
from hret_agents.tools import (
    DatasetDownloader, 
    DatasetTranslator, 
    ModuleGenerator, 
    DatasetPusher
)

# 에이전트 설정
agent = DatasetTranslationAgent(
    openai_api_key="your-openai-api-key",
    hf_token="your-huggingface-token"
)

# 필요한 도구 추가
agent.add_tool(DatasetDownloader())
agent.add_tool(DatasetTranslator(batch_size=10, max_retries=3))
agent.add_tool(ModuleGenerator())
agent.add_tool(DatasetPusher())

# 데이터셋 처리 실행
result = agent.process_dataset(
    dataset_name="HAERAE-HUB/QARV",
    subset="your_subset",
    split="train",
    push_to_hub=True
)

print(f"생성된 데이터셋 모듈: {result['module_path']}")

```

### 3.4 데이터셋 변환 과정 상세 설명

HRET Agents는 다음과 같은 단계로 데이터셋을 처리합니다:

1. **데이터셋 다운로드**: Hugging Face Hub에서 지정된 데이터셋을 다운로드합니다.
2. **데이터셋 분석**: 데이터셋 구조를 분석하고 상위 5개 행을 마크다운 테이블로 변환합니다.
3. **한국어 번역**: 데이터셋의 열 이름과 내용을 한국어로 번역합니다(배치 처리 및 재시도 로직 포함).
4. **모듈 생성**: OpenAI API를 사용하여 HRET 호환 데이터셋 모듈 코드를 생성합니다.
5. **저장 및 업로드**: 생성된 모듈을 로컬에 저장하고 선택적으로 Hugging Face Hub에 업로드합니다.

### 3.5 커스텀 데이터셋 가이드 프롬프트 활용

데이터셋 모듈 생성에 사용되는 가이드 프롬프트를 커스터마이징할 수 있습니다:

```python
# 커스텀 가이드 프롬프트 정의
custom_guide_prompt = """
당신은 Haerae Evaluation Toolkit을 위한 데이터셋 모듈을 생성하는 전문가입니다.
다음 데이터셋 스키마와 예시 데이터를 바탕으로 완전한 Python 모듈을 작성해주세요:

데이터셋 이름: {dataset_name}
데이터셋 구조:
{dataset_schema}

예시 데이터:
{sample_data}

다음 BaseDataset 클래스를 상속받아 구현해주세요:
{base_class_definition}

추가 요구사항:
1. 모든 필드는 한국어로 번역되어야 합니다.
2. get_prompt 메소드는 문제를 효과적으로 포맷팅해야 합니다.
3. evaluate_prediction 메소드는 정답을 정확히 평가해야 합니다.
"""

# 커스텀 가이드 프롬프트를 사용한 모듈 생성
from hret_agents.tools import ModuleGenerator

module_generator = ModuleGenerator(guide_prompt=custom_guide_prompt)
module_code = module_generator.generate_module(
    dataset_info={
        "name": "HAERAE-HUB/QARV",
        "schema": "...",  # 데이터셋 스키마 정보
        "sample_data": "..."  # 샘플 데이터
    }
)

print(module_code)
```


## 4. HRET 데이터셋 모듈 구조 이해하기

### 4.1 HRET 데이터셋 모듈이란?
- HRET 데이터셋 모듈은 Haerae Evaluation Toolkit에서 평가에 사용할 수 있는 표준화된 데이터셋 클래스입니다.
- `BaseDataset` 클래스를 상속받아 구현되며, 데이터 로딩, 프롬프트 생성, 답변 평가 등의 기능을 제공합니다.
- HRET Agents는 이러한 모듈을 자동으로 생성해주므로 수동 작업이 크게 줄어듭니다.

### 4.2 데이터셋 모듈의 기본 구조

```python
# 자동 생성된 데이터셋 모듈 예시
from llm_eval.dataset.base import BaseDataset
from datasets import load_dataset
import re

class QARVDataset(BaseDataset):
    def __init__(self, subset="your_subset", split="train"):
        """
        QARV 데이터셋 초기화
        
        Args:
            subset (str): 사용할 서브셋
            split (str): 데이터 분할(train, validation, test)
        """
        self.subset = subset
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """데이터셋 로드 및 전처리"""
        dataset = load_dataset("HAERAE-HUB/QARV", self.subset)
        self.data = dataset[self.split]
    
    def get_prompt(self, index):
        """
        특정 인덱스의 프롬프트 생성
        
        Args:
            index (int): 데이터 인덱스
            
        Returns:
            str: 포맷팅된 프롬프트
        """
        example = self.data[index]
        prompt = f"질문: {example['질문']}\n\n"
        prompt += f"다음 중 올바른 답을 선택하세요:\n"
        
        for i, option in enumerate(example['선택지']):
            prompt += f"{chr(65+i)}. {option}\n"
        
        return prompt
    
    def evaluate_prediction(self, index, prediction):
        """
        예측 결과 평가
        
        Args:
            index (int): 데이터 인덱스
            prediction (str): 모델의 예측 결과
            
        Returns:
            tuple: (정답 여부, 점수)
        """
        example = self.data[index]
        correct_answer = example['정답']
        
        # 예측에서 A, B, C, D 등의 답안 추출
        pattern = r'([A-D])'
        matches = re.findall(pattern, prediction)
        
        if not matches:
            return False, 0
        
        predicted_answer = matches[0]
        is_correct = (predicted_answer == correct_answer)
        
        return is_correct, 1 if is_correct else 0
```


## 5. 데이터셋 생성 및 사용 워크플로우

HRET Agents와 LiteLLM/OpenAI-compatible API를 함께 활용한 완전한 워크플로우를 알아보겠습니다.

```python
import os
from llm_eval.evaluator import Evaluator
from hret_agents.agent import DatasetTranslationAgent
from hret_agents.tools import DatasetDownloader, DatasetTranslator, ModuleGenerator

# 1. 환경 설정
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["HF_TOKEN"] = "your-huggingface-token"

# 2. 데이터셋 준비 (HRET Agent 사용)
agent = DatasetTranslationAgent(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    hf_token=os.environ["HF_TOKEN"]
)

# 필요한 도구 추가
agent.add_tool(DatasetDownloader())
agent.add_tool(DatasetTranslator(batch_size=10, max_retries=3))
agent.add_tool(ModuleGenerator())

# 데이터셋 처리 실행
result = agent.process_dataset(
    dataset_name="HAERAE-HUB/QARV",
    subset="your_subset",
    split="train"
)

# 3. LiteLLM을 사용한 평가
evaluator = Evaluator()

results = evaluator.run(
    model="litellm",
    model_params={
        "model_name": "gpt-4",
        "api_key": os.environ["OPENAI_API_KEY"],
        "max_tokens": 512
    },
    
    # 생성된 데이터셋 모듈 사용
    dataset="HAERAE-HUB/QARV",
    subset=["your_subset"],
    split="train",
    
    evaluation_method="string_match"
)

print(results)
print(results.to_dataframe())
```

## 6. 주의사항 및 팁

1. **API 키 보안**
   - API 키는 안전하게 관리해야 합니다. 환경 변수나 보안 설정을 통해 키를 관리하는 것이 좋습니다.
   - 예: `os.environ.get("OPENAI_API_KEY")` 또는 `.env` 파일 사용

2. **번역 비용 및 시간 관리**
   - 대량의 데이터셋을 번역할 경우 OpenAI API 비용이 상당히 발생할 수 있습니다.
   - 배치 크기와 재시도 횟수를 적절히 조정하여 비용과 시간을 최적화하세요.
   - 테스트 용도로는 작은 데이터셋 서브셋으로 먼저 시도해 보는 것이 좋습니다.

3. **데이터셋 모듈 검증**
   - 자동 생성된 데이터셋 모듈은 항상 수동으로 검토하여 품질을 확인하는 것이 좋습니다.
   - 특히 `evaluate_prediction` 메소드의 평가 로직이 정확한지 확인하세요.

4. **백엔드 간 전환**
   - 다양한 백엔드 간 전환 시, 모델별 특성(토큰 한계, 프롬프트 포맷 등)이 다를 수 있으니 주의하세요.
   - 동일한 테스트를 여러 백엔드에서 실행하여 결과를 비교해보는 것이 좋습니다.

5. **에러 처리 및 로깅**
   - HRET Agents는 기본적으로 번역 실패 시 재시도 로직을 포함하고 있지만, 추가적인 에러 처리가 필요할 수 있습니다.
   - `verbose=True` 옵션을 통해 상세 로그를 활성화하여 문제 해결에 활용하세요.

6. **데이터셋 확장**
   - 새로운 평가 기준이 필요한 경우, 생성된 데이터셋 모듈을 확장하여 추가 메소드를 구현할 수 있습니다.
   - 예를 들어, 다양한 평가 메트릭(BLEU, ROUGE 등)을 추가하여 성능을 다각도로 평가할 수 있습니다.
