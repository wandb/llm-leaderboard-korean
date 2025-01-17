from typing import List, Dict, Any, Optional, Callable
from collections import Counter
import re
from llm_eval.models.base import BaseModel
from .base import BaseScalingMethod
from . import register_scaling_method

@register_scaling_method("self_consistency")
class SelfConsistencyScalingMethod(BaseScalingMethod):
    """
    Self-Consistency Chain of Thought (CoT) 스케일링 기법 (정규식 기반 파싱 포함).

    Args:
        model (BaseModel): LLM 모델 (generate_batch() 제공).
        n_paths (int): 몇 번의 reasoning path(=샘플링)을 생성할지
        aggregator_fn (Callable): 여러 후보 중 최종 답안을 결정하는 함수.
            디폴트: majority_voting
        prompt_cot (str): CoT 유도를 위해 prompt에 추가할 문구
        parse_answer_pattern (str): 정규식 패턴. 매칭되면 group(1) 등을 최종 답안으로 사용.
            예: r"(?i)final\s*answer:\s*(.*)"
            None이면 별도 파싱 없이 raw_text를 그대로 사용.
    """

    def __init__(
        self,
        model: BaseModel = None,
        n_paths: int = 5,
        aggregator_fn: Optional[Callable[[List[str]], str]] = None,
        prompt_cot: Optional[str] = None,
        parse_answer_pattern: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.n_paths = n_paths
        self.prompt_cot = prompt_cot or "\nLet's think step by step.\n"
        self.aggregator_fn = aggregator_fn if aggregator_fn else self._majority_voting
        self.parse_answer_pattern = parse_answer_pattern  # 정규식 패턴 (None이면 identity 그대로로)

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.model:
            raise ValueError("SelfConsistencyScalingMethod: model is not set.")

        for sample in data:
            original_prompt = sample["input"]
            prompt = original_prompt + self.prompt_cot

            candidates = []
            for _ in range(self.n_paths):
                outputs = self.model.generate_batch(
                    [{"input": prompt}],
                    return_logits=False
                )
                raw_text = outputs[0].get("prediction", "")

                # 정규식 패턴이 있으면 regex 파싱, 없으면 identity
                if self.parse_answer_pattern:
                    parsed_answer = self._regex_parse_answer(raw_text, self.parse_answer_pattern)
                else:
                    parsed_answer = raw_text.strip()

                candidates.append(parsed_answer)

            # aggregator로 최종 답 결정
            final_answer = self.aggregator_fn(candidates)
            sample["prediction"] = final_answer

            # 필요 시, 후보 목록을 저장할 수도 있으나? 일단은 그대로 두기기
            # sample["cot_candidates"] = candidates

        return data

    def _regex_parse_answer(self, text: str, pattern: str) -> str:
        """
        주어진 정규식 패턴으로 text를 매칭해, group(1) 등을 반환.
        매칭 실패 시 text 전체를 trim하여 반환.
        예:
          pattern = r"(?i)final\s*answer:\s*(.*)"
          "Here is my reasoning. Final Answer: 42" -> "42"
        """
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # group(1) 기준으로 추출 (필요하면 group(2) etc. 수정)
            return match.group(1).strip()
        return text.strip()

    def _majority_voting(self, candidates: List[str]) -> str:
        """
        가장 많이 등장한 답을 고르는 단순 다수결.
        """
        if not candidates:
            return ""
        counter = Counter(candidates)
        # most_common(1)[0] -> (답변, 빈도)
        return counter.most_common(1)[0][0]
