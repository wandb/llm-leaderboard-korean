from typing import List, Dict, Any
from llm_eval.models.base import BaseModel
from .base import BaseScalingMethod
from . import register_scaling_method

@register_scaling_method("best_of_n")
class BestOfN(BaseScalingMethod):
    """
    N번 샘플링하여 가장 우수한 답변(또는 첫 답변)을 선택하는 방식.
    * "우수" 판단은 score_fn으로 할 수도 있고, default로 첫 번째만 택할 수도 있음.
    """

    def __init__(
        self, 
        model: BaseModel = None, 
        n: int = 5, 
        score_fn: Optional[Callable[[str, Dict[str, Any]], float]] = None, 
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.n = n
        self.score_fn = score_fn

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        data 내 각 샘플에 대해:
          - N번 반복하여 후보 생성
          - score_fn이 있으면 최고 점수 후보 선택, 없으면 첫 후보
        """
        if self.model is None:
            raise ValueError("BestOfN requires a 'model' instance.")

        for sample in data:
            prompt = sample["input"]
            candidates = []
            logit_values = []

            for _ in range(self.n):
                outputs = self.model.generate_batch([sample], return_logits=True)  
                candidate_text = outputs[0]["prediction"]
                logit_value = outputs[0].get("logit", None) 
                candidates.append(candidate_text)
                logit_values.append(logit_value)

            if self.score_fn is not None:
                # 사용자가 제공한 score_fn 기반으로 최고 점수 후보 선택
                scores = [self.score_fn(candidate, sample) for candidate in candidates]
                best_idx = max(range(len(scores)), key=lambda i: scores[i])

            else:
                
                best_idx = self._select_highest_confidence(logit_values)

            sample["prediction"] = candidates[best_idx]

        return data

    def _select_highest_confidence(self, logit_values: List[Optional[float]]):
       

        return max(range(len(logit_values)), key=lambda i: logit_values[i])
