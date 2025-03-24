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


