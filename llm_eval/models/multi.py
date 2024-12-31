"""
multi.py
현재 llm-as-a-judge, scaling_method와 같은 방법론에서는 두개의 모델을 사용하여야 하는데
단순히 base 구조로는 한개의 모델밖에 다룰 수가 없음.
이 모듈은 여러 개의 LLM 백엔드를 동시에 로드하고, 
하나의 모델 객체처럼 동작하는 'MultiModel' 클래스를 정의.
(예: vLLM, HuggingFace, OpenAI-Compatible 서버 등 다양한 백엔드 혼합)

사용 예시:

    multi_config = [
        {"name": "vllm", "endpoint": "http://localhost:8000"},
        {"name": "huggingface", "model_name": "gpt2"},
        {"name": "openai", "api_base": "https://api.openai.com", "api_key": "...", "model_name": "text-davinci-003"}
    ]

    multi_model = load_model("multi", models_config=multi_config, aggregate_strategy="first")
    outputs = multi_model.generate_batch(data, return_logits=False)
    # data -> [{"input":"...", "reference":"..."}, ...]

    # 결과: 
    # 각 item에 "multi_outputs" 필드: 
    #   [
    #     {"model_name": "vllm", "prediction":... , "logits":...},
    #     {"model_name": "huggingface", "prediction":... },
    #     {"model_name": "openai", "prediction":... },
    #   ]
    # 그리고 최종 "prediction"은 aggregate_strategy="first"에 따라 첫 모델 결과만 사용
"""
from typing import List, Dict, Any, Optional

from .base import BaseModel
from . import load_model, register_model



@register_model("multi")
class MultiModel(BaseModel):
    """
    여러 모델(서브 모델) 인스턴스를 내부에 보관하고,
    'generate_batch' 호출 시 각 서브 모델을 순회하여 
    결과를 한 번에 모으는 모델 클래스.

    Attributes:
        models_config (List[Dict[str, Any]]): 
            - 서브 모델을 로드하기 위한 설정 목록.
            - 각 항목은 
              {
                "name": <str: MODEL_REGISTRY 키>,
                ... (모델 별 필요한 인자)
              }
            - 예: [
                {"name": "vllm", "endpoint": "http://127.0.0.1:8000"},
                {"name": "huggingface", "model_name": "gpt2"},
                ...
              ]

        aggregate_strategy (str): 
            - 여러 서브 모델의 결과를 최종적으로 어떻게 합칠지 결정.
            - "first"    : 첫 번째 모델 결과만 최종 prediction으로 사용
            - "vote"     : 간단한 majority voting (동일 문자열 가장 많은 것)
            - "combine"  : 모든 모델 결과를 리스트로 보관 (prediction은 None)
            - (필요하다면 추가 전략 구현 가능)

        models (List[BaseModel]):
            - 실제 서브 모델 인스턴스들의 리스트
    """

    def __init__(
        self,
        models_config: Optional[List[Dict[str, Any]]] = None,
        aggregate_strategy: str = "first",
        **kwargs
    ):
        """
        MultiModel 초기화 메서드.

        Args:
            models_config (List[Dict[str, Any]]): 
                여러 서브 모델을 로드하기 위한 설정 목록.
            aggregate_strategy (str, optional):
                여러 모델 출력 중 최종 prediction을 결정하는 방식.
                Defaults to "first".
            **kwargs:
                BaseModel에서 물려받을 수 있는 추가 파라미터 (필요시 사용).
        """
        super().__init__(**kwargs)
        if models_config is None:
            models_config = []

        self.models_config: List[Dict[str, Any]] = models_config
        self.aggregate_strategy = aggregate_strategy

        # 실제 서브 모델 인스턴스 보관
        self.models: List[BaseModel] = []

        # 모델 불러오기
        for idx, mc in enumerate(self.models_config):
            model_name = mc.pop("name", None)
            if not model_name:
                raise ValueError(f"Model config {idx} is missing 'name' field.")
            sub_model = load_model(model_name, **mc)
            self.models.append(sub_model)

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False
    ) -> List[Dict[str, Any]]:
        """
        여러 서브 모델에 대해 순차적으로 generate_batch를 호출한 뒤,
        각 샘플에 "multi_outputs" 필드로 모든 모델의 결과를 기록한다.
        이후 aggregate_strategy에 따라 최종 "prediction" 필드를 결정.

        Args:
            inputs (List[Dict[str, Any]]): 
                [{ "input": str, "reference": str, ... }, ...]
            return_logits (bool, optional):
                모델 로그 확률 정보 등 반환 여부. Defaults to False.

        Returns:
            List[Dict[str, Any]]:
                각 아이템은 다음 필드를 가질 수 있음:
                {
                  "input": ...,
                  "reference": ...,
                  "multi_outputs": [
                    {
                      "model_name": <str>,
                      "prediction": <str>,
                      "logits": (optional, if return_logits=True)
                      ...
                    },
                    ...
                  ],
                  "prediction": <str or None> (aggregate_strategy에 따라)
                }
        """
        if not self.models:
            return inputs

        # 먼저, "multi_outputs"를 초기화
        for item in inputs:
            item["multi_outputs"] = []

        # 1) 각 모델에 대해 순차적으로 generate_batch 호출
        #    (병렬로 처리하고 싶다면 concurrent.futures 등을 사용할 수도 있음)
        for idx, sub_model in enumerate(self.models):
            # 서브 모델이 inputs를 in-place로 수정 후 반환할 수도 있으니,
            # 복사본 없이 바로 부르는 방식을 사용 (주의)
            sub_results = sub_model.generate_batch(inputs, return_logits=return_logits)

            # 2) sub_results = inputs (동일 객체 참조) 이므로, 
            #    각 아이템에 해당 모델의 결과를 담아둔다.
            for item, sub_item in zip(inputs, sub_results):
                # sub_item["prediction"]가 모델의 생성 결과
                # sub_item.get("logits") 등등이 있을 수 있음
                item["multi_outputs"].append({
                    "model_name": type(sub_model).__name__,  # or registry key
                    "prediction": sub_item.get("prediction", None),
                    "logits": sub_item.get("logits", None) if return_logits else None
                })

        # 3) aggregate_strategy에 따라 최종 "prediction" 결정
        self._apply_aggregate_strategy(inputs)

        return inputs

    def _apply_aggregate_strategy(self, batch: List[Dict[str, Any]]) -> None:
        """
        내부 헬퍼 메서드: batch 내 각 아이템에 대해 aggregate_strategy를 적용해
        최종 prediction을 결정하거나, 보관만 해둘 수 있다.
        """
        if self.aggregate_strategy == "first":
            self._aggregate_first(batch)
        elif self.aggregate_strategy == "vote":
            self._aggregate_vote(batch)
        elif self.aggregate_strategy == "combine":
            self._aggregate_combine(batch)
        else:
            raise

    def _aggregate_first(self, batch: List[Dict[str, Any]]) -> None:
        """
        첫 번째 모델 결과를 그대로 최종 prediction으로 사용.
        """
        for item in batch:
            outputs = item["multi_outputs"]
            if not outputs:
                item["prediction"] = None
            else:
                item["prediction"] = outputs[0]["prediction"]

    def _aggregate_vote(self, batch: List[Dict[str, Any]]) -> None:
        """
        간단한 Majority Voting 로직:
        동일한 문자열 결과가 가장 빈번히 등장하는 것을 최종 prediction으로 설정.
        """
        for item in batch:
            outputs = item["multi_outputs"]
            if not outputs:
                item["prediction"] = None
                continue

            count_map = {}
            for out in outputs:
                pred = out["prediction"]
                count_map[pred] = count_map.get(pred, 0) + 1

            # 빈도수가 가장 높은 문자열
            best_pred = max(count_map, key=count_map.get)
            item["prediction"] = best_pred

    def _aggregate_combine(self, batch: List[Dict[str, Any]]) -> None:
        """
        여러 모델의 결과를 전부 리스트로 보관하고, 
        최종 prediction은 None으로 두는 모드.
        - 스케일링 메서드나 상위 로직에서 따로 처리하게끔 할 수 있음.
        """
        for item in batch:
            item["prediction"] = None  # 실제 prediction은 multi_outputs에만 저장
