from typing import List, Dict, Any, Optional, Union
import logging

from .base import BaseModel, BaseJudge, BaseRewardModel
from . import load_model, register_model

logger = logging.getLogger(__name__)

@register_model("multi")
class MultiModel:
    """
    "하나의 MultiModel 객체에 세 가지 역할을 위한 모델을 각각 최대 한 개씩 보관"하고,
    필요 시점에 따라 메서드를 호출할 수 있는 구조.

    - self.generate_model: BaseModel 상속 (텍스트 생성)
    - self.judge_model: BaseJudge 상속 (LLM-as-a-Judge)
    - self.reward_model: BaseRewardModel 상속 (보상 점수 계산)

    사용 예시:
        config = {
          "generate_model": { "name": "huggingface", "params": { "model_name_or_path": "gpt2" } },
          "judge_model": { "name": "my_judge_llm", "params": {...} },
          "reward_model": None
        }
        multi_model = load_model("multi", **config)

        # 텍스트 생성
        generated = multi_model.generate_batch(data, return_logits=False)

        # Judge 평가
        judged = multi_model.judge_batch(generated)

        # Reward 점수
        scored = multi_model.score_batch(judged)
    """

    def __init__(
        self,
        generate_model: Optional[Dict[str, Any]] = None,
        judge_model: Optional[Dict[str, Any]] = None,
        reward_model: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Args:
            generate_model: 
                - {"name":"huggingface", "params":{...}} 형태, BaseModel 구현체 로드에 필요한 정보
            judge_model:
                - {"name":"some_judge_backend", "params":{...}}, BaseJudge 구현체
            reward_model:
                - {"name":"some_reward_backend", "params":{...}}, BaseRewardModel 구현체
            kwargs:
                - 나머지 인자들은 무시 or 확장 용도로
        """
        # BaseModel, BaseJudge, BaseRewardModel 인스턴스 or None
        self.generate_model: Optional[BaseModel] = None
        self.judge_model: Optional[BaseJudge] = None
        self.reward_model: Optional[BaseRewardModel] = None

        # generate_model 로딩
        if generate_model is not None:
            g_name = generate_model.get("name")
            g_params = generate_model.get("params", {})
            logger.info(f"[MultiModel] Loading generate model: {g_name} with {g_params}")
            loaded = load_model(g_name, **g_params)
            if not isinstance(loaded, BaseModel):
                raise ValueError(f"Loaded generate_model is not a BaseModel: {type(loaded)}")
            self.generate_model = loaded

        # judge_model 로딩
        if judge_model is not None:
            j_name = judge_model.get("name")
            j_params = judge_model.get("params", {})
            logger.info(f"[MultiModel] Loading judge model: {j_name} with {j_params}")
            loaded = load_model(j_name, **j_params)
            if not isinstance(loaded, BaseJudge):
                raise ValueError(f"Loaded judge_model is not a BaseJudge: {type(loaded)}")
            self.judge_model = loaded

        # reward_model 로딩
        if reward_model is not None:
            r_name = reward_model.get("name")
            r_params = reward_model.get("params", {})
            logger.info(f"[MultiModel] Loading reward model: {r_name} with {r_params}")
            loaded = load_model(r_name, **r_params)
            if not isinstance(loaded, BaseRewardModel):
                raise ValueError(f"Loaded reward_model is not a BaseRewardModel: {type(loaded)}")
            self.reward_model = loaded

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        1) generate_model이 있으면, 각 샘플에 대해 text generation 수행
        2) 생성된 "prediction" 필드를 샘플에 추가
        3) return_logits=True면, "logits" 필드도 추가
        4) N개의 입력 -> N개의 출력 (same length)

        예:
          inputs = [{"input":"Hello", "reference":"World"}, ...]
          return -> [{"input":"Hello", "reference":"World", "prediction":"..."}, ...]
        """
        if self.generate_model is None:
            # generate_model이 없으면 그냥 inputs 그대로 반환
            return inputs

        return self.generate_model.generate_batch(
            inputs,
            return_logits=return_logits,
            **kwargs
        )

    def judge_batch(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        1) judge_model이 있으면, 각 샘플에 대해 judge_batch 로직을 수행
        2) judge_model은 BaseJudge를 상속, 예: "judge_batch(inputs)" -> 
           샘플별로 "judge_score", "judge_explanation" 등을 추가
        3) N->N 매핑
        """
        if self.judge_model is None:
            return inputs
        
        # judge_model이 BaseJudge면 judge_model.judge_batch(inputs)로 호출
        return self.judge_model.judge_batch(inputs, **kwargs)

    def score_batch(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        1) reward_model이 있으면, 각 샘플에 대해 score_batch 로직
        2) reward_model은 BaseRewardModel 상속, 샘플별 "reward" 필드 추가
        3) N->N 매핑
        """
        if self.reward_model is None:
            return inputs

        return self.reward_model.score_batch(inputs, **kwargs)
