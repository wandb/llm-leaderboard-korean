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
from typing import List, Dict, Any, Optional, Union

from .base import BaseModel, BaseJudge, BaseRewardModel
from . import load_model, register_model


@register_model("multi")
class MultiModel:
    """
    여러 모델(또는 Judge, Reward 모델)을 내부에 보관하고,
    호출 방식에 따라 'generate_batch' / 'judge_batch' / 'score_batch' 등을
    각각 순차 실행 가능하게 설계한 클래스.

    Attributes:
        items_config: 
            - 서브 모델(혹은 Judge/Reward)의 설정 목록.
            - 예: [
                {"name": "vllm", "endpoint": "...", "type": "model"},
                {"name": "judge_llm", "some_judge_config": "...", "type": "judge"},
                {"name": "rm_model", "some_reward_config": "...", "type": "reward"}
              ]
        sub_items:
            - 실제 인스턴스들(BaseModel, BaseJudge, BaseRewardModel).
    """

    def __init__(
        self,
        items_config: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Args:
            items_config (List[Dict[str, Any]]): 
                여러 서브 엔티티를 로드하기 위한 설정.
                각 item은 최소 'name' 키(레지스트리 키)와 'type' 키(model/judge/reward 등)를 가져야 함.
        """
        super().__init__()
        if items_config is None:
            items_config = []

        self.items_config: List[Dict[str, Any]] = items_config
        self.sub_items: List[Union[BaseModel, BaseJudge, BaseRewardModel]] = []
        
        # 로딩
        for idx, cfg in enumerate(self.items_config):
            model_name = cfg.pop("name", None)
            item_type = cfg.pop("type", None)  # "model", "judge", "reward" 등
            if not model_name or not item_type:
                raise ValueError(f"Config {idx} must have 'name' and 'type' fields.")

            sub_obj = load_model(model_name, **cfg)
            # sub_obj가 BaseModel, BaseJudge, BaseRewardModel 중 하나
            self.sub_items.append(sub_obj)

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False
    ) -> List[Dict[str, Any]]:
        """
        모든 'BaseModel' 타입의 서브 아이템에 대해서만 generate_batch를 호출.
        Judge/Reward 타입인 경우 무시(혹은 에러).
        """
        for sub_item in self.sub_items:
            if isinstance(sub_item, BaseModel):
                sub_item.generate_batch(inputs, return_logits=return_logits)
            else:
                # Judge나 Reward는 실제 텍스트 생성을 하지 않으므로 패스
                pass
        return inputs

    def judge_batch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        모든 'BaseJudge' 타입의 서브 아이템에 대해 judge_batch 호출.
        """
        for sub_item in self.sub_items:
            if isinstance(sub_item, BaseJudge):
                judge_outputs = sub_item.judge_batch(inputs)
                # judge_outputs가 inputs와 동일 객체(혹은 새 객체)인지 여부는 구현마다 다름
                # 여기서는 sub_item이 in-place로 inputs에 "judge_score" 등 붙인다고 가정
            else:
                # model/reward는 pass
                pass
        return inputs

    def score_batch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        모든 'BaseRewardModel' 타입의 서브 아이템에 대해 score_batch 호출.
        """
        for sub_item in self.sub_items:
            if isinstance(sub_item, BaseRewardModel):
                reward_outputs = sub_item.score_batch(inputs)
                # 역시 reward_outputs가 inputs와 동일 객체인지 체크
            else:
                pass
        return inputs