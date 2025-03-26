# 1. Best-of-N 기법

## 1.1 Best-of-N이란?
- **Best-of-N**은 동일한 입력(프롬프트)에 대해 여러 번(`N`번) 샘플링을 수행한 뒤, **가장 높은 스코어**(로그 확률 혹은 리워드 등)를 갖는 답변을 최종 결과로 결정하는 방식입니다.
- 같은 프롬프트라도 모델이 내부적으로 샘플링을 달리할 경우, 서로 다른 출력을 생성하기 때문에 활용 가치가 큽니다.

## 1.2 Best-of-N 사용 방법
```python
from llm_eval.evaluator import Evaluator

# 1) Evaluator 인스턴스 생성
evaluator = Evaluator()

# 2) Best-of-N 평가 실행
results = evaluator.run(
    model="huggingface",  # 또는 "openai", "vllm" 등
    model_params={
        "model_name_or_path": "kakaocorp/kanana-nano-2.1b-instruct",
        "device": "cuda:0",
        "batch_size": 1,
        "max_new_tokens": 128
    },

    dataset="haerae_bench",
    subset=["csat_geo"],
    split="test",

    # Best-of-N 설정
    scaling_method="best_of_n",
    scaling_params={
        "n": 5,  # 동일 프롬프트에 대해 5회 샘플링
        "batch_size": 1
    },

    evaluation_method="string_match"  # 채점 방식
)

print(results)
df = results.to_dataframe()
print(df)
```

# 2. Beam Search 기법

## 2.1 Beam Search란?
- **Beam Search**는 각 생성 단계에서 **여러 후보(beam)**를 추적하며 탐색을 확장해나가는 알고리즘입니다.
- Greedy Search(단일 후보)보다 넓은 경로를 탐색하여 더 높은 확률의 문장을 도출할 가능성이 큽니다.
- `beam_size`를 크게 잡으면 탐색 범위가 넓어지지만, 연산량 및 시간도 증가하므로 적절한 균형이 필요합니다.

## 2.2 Beam Search 사용 방법
```python
from llm_eval.evaluator import Evaluator

# 1) Evaluator 생성
evaluator = Evaluator()

# 2) Beam Search 적용
results = evaluator.run(
    model="huggingface",
    model_params={
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "device": "cuda:0",
        "batch_size": 1
    },

    dataset="haerae_bench",
    subset=["csat_geo"],
    split="test",

    scaling_method="beam_search",
    scaling_params={
        "beam_size": 4,      # Beam 크기
        "max_tokens": 50,    # 최대 생성 토큰 수
        "agg_strategy": "sum",  # 점수 합산 방식(sum, mean, max 등)
        "batch_size": 1
    },

    evaluation_method="string_match"
)

print(results)
df = results.to_dataframe()
print(df)
```

# 3. 주의 사항 및 팁

1. **시간 및 리소스 사용**  
   - Beam Search와 Best-of-N 모두 추론을 여러 회 반복하거나 여러 후보를 동시에 탐색하므로 **추론 시간이 늘어나고** GPU 메모리 사용량이 증가합니다.  
   - 빠르게 결과가 필요한 상황이라면, `beam_size`나 `n` 값을 작게 조절해보십시오.

2. **스케일링 기법 간 호환성**  
   - HRET에서는 한 번의 평가(`run()`)에서 **단일 `scaling_method`**만 적용 가능합니다.  
   - 예: `best_of_n`와 `beam_search`를 동시에 사용할 수 없습니다.

3. **LLM-as-Judge와 연동**  
   - Best-of-N이나 Beam Search 시, **후보 간 우위를 결정하는** 기준으로 로그 확률 대신 **LLM-as-Judge** 점수를 활용할 수도 있습니다.  
   - 이 경우 `judge_model` 또는 `reward_model`을 지정하여, 후보 답안을 모델이 직접 평가(채점)하게 만들 수 있습니다.  
   - 자연스러운 평가 스키마나 특정 기준이 필요할 때 유용합니다.


