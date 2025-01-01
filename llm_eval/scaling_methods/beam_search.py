import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

from .base import BaseScalingMethod
from . import register_scaling_method
from llm_eval.models.base import BaseModel


@dataclass
class Beam:
    """
    Beam(빔 서치) 하나를 나타내는 보조 Class.
    """
    prompt: str               # 원본 프롬프트(질문)
    index: int                # 빔 식별자(0, 1, 2, ...)
    current_text: str = ""    # 현재까지 생성된 텍스트
    completed: bool = False   # EOS 등으로 생성이 끝났는지 여부
    pruned: bool = False      # 점수가 낮아 제거(prune)되었는지
    score_history: List[float] = field(default_factory=list)  # 매 스텝에서의 점수(혹은 logprob) 기록
    completion_tokens: int = 0  # 생성에 사용된 토큰 수 (옵션)

    def aggregate_score(self, agg_fn) -> float:
        """
        매 스텝 기록한 점수(혹은 로그확률)를 agg_fn으로 합산해 최종 점수화.
        예: sum, mean, max 등.
        """
        if not self.score_history:
            return 0.0
        return agg_fn(self.score_history)


@register_scaling_method("beam_search")
class BeamSearch(BaseScalingMethod):
    """
    간단한 Beam Search (25.1.1 동작 확인 아직 안함함)
    """

    def __init__(
        self,
        model: BaseModel = None,
        beam_size: int = 4,
        num_iterations: int = 3,
        n: int = 1,
        agg_strategy: str = "mean",
        max_tokens: int = 50,
        filter_duplicates: bool = True,
        **kwargs
    ):
        """
        Args:
            model (BaseModel): 토큰 생성 메서드(generate_batch)를 제공하는 모델.
            beam_size (int): 각 iteration 당 유지할 빔의 개수.
            num_iterations (int): 몇 번의 iteration을 돌면서 빔을 확장할지.
            n (int): 초기 빔 생성 시, prompt별로 몇 개나 복제할지(또는 탐색 후보).
            agg_strategy (str): 점수를 합산/평균 등 어떻게 aggregate할지 결정.
            max_tokens (int): 모델 호출 시 생성할 최대 토큰 길이.
            filter_duplicates (bool): 동일한 current_text를 갖는 빔을 중복 제거할지 여부.
            kwargs: 그 외 BaseScalingMethod에서 공통적으로 사용하는 파라미터들.
        """
        super().__init__(model=model, **kwargs)
        self.beam_size = beam_size
        self.num_iterations = num_iterations
        self.n = n
        self.agg_strategy = agg_strategy
        self.max_tokens = max_tokens
        self.filter_duplicates = filter_duplicates

    def _aggregate_scores(self, scores: List[float]) -> float:
        """
        점수 리스트를 self.agg_strategy에 따라 합산/평균 등으로 스칼라화.
        """
        if not scores:
            return 0.0
        if self.agg_strategy == "sum":
            return sum(scores)
        elif self.agg_strategy == "max":
            return max(scores)
        else:
            # default: mean
            return sum(scores) / len(scores)

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        data: [{"input":"...", "reference":"..."}, ...] 형태.
        최종적으로 각 샘플에 "prediction" 필드를 채워서 반환.
        """
        if self.model is None:
            raise ValueError("BeamSearch requires a 'model' instance.")

        # 최종 결과 저장용
        results = []

        for sample in data:
            prompt = sample["input"]
            
            # 1) 초기 빔 생성
            beams = []
            for i in range(self.beam_size):
                beam = Beam(
                    prompt=prompt,
                    index=i,
                    current_text="", 
                )
                beams.append(beam)

            completed_beams = []

            # 2) iteration 반복
            for it in range(self.num_iterations):
                # 아직 pruned/complete 아닌 active beam만 추출
                active_beams = [b for b in beams if not b.pruned and not b.completed]

                # 만약 active_beams가 0개면 종료
                if len(active_beams) == 0:
                    break

                # 만약 active_beams가 beam_size 미만이면, beam_size까지 복제 (예시 로직)
                # pseudo-code에서 n과 beam_size 개념이 혼재할 수 있음. 
                # 필요에 따라 조정 가능.
                if len(active_beams) < self.beam_size:
                    repeats = (self.beam_size // len(active_beams)) + 1
                    extended = (active_beams * repeats)[: self.beam_size]
                    active_beams = [copy.deepcopy(b) for b in extended]

                # 모델 호출을 위해 batch로 준비
                # 여기서는 "한 번에 한 토큰만" 추가한다고 가정(실제론 여러 토큰이 생성될 수 있음)
                batch_inputs = []
                for b in active_beams:
                    # prompt + 현재 텍스트를 합쳐 모델에 전달
                    # (실제론 system_prompt나 special tokens 등을 추가할 수 있음)
                    combined_input = b.prompt + b.current_text
                    batch_inputs.append({"input": combined_input})

                # 3) 모델 호출
                # generate_batch: 
                #   [{"input":"...", "prediction":"생성된 토큰(또는 문자열)", ...}, ...]
                #   여기서는 pseudo-code로 "한 토큰 또는 짧은 문구"만 반환한다고 가정
                generation_outputs = self.model.generate_batch(
                    batch_inputs,
                    return_logits=False  # logprob 계산할거면 True
                )

                # 4) beam 업데이트
                for b, gen_out in zip(active_beams, generation_outputs):
                    # 여기선 단일 토큰(혹은 짧은 스텝)을 b.current_text에 이어붙임
                    token_or_text = gen_out.get("prediction", "")
                    b.current_text += token_or_text  
                    
                    # 점수(예: logprob) 등 추가 처리 가능
                    # b.score_history.append(...)
                    
                    # 만약 EOS 조건(빈 문자열, 특정 stop token 등)이면 completed = True
                    if token_or_text == "" or len(b.current_text) >= self.max_tokens:
                        b.completed = True
                        completed_beams.append(b)

                # 중복 제거 (filter_duplicates=True 시)
                if self.filter_duplicates:
                    unique_dict = {}
                    for b in active_beams:
                        if b.current_text not in unique_dict:
                            unique_dict[b.current_text] = b
                        else:
                            b.pruned = True  # 이미 같은 text를 갖는 빔이 존재하면 prune
                    active_beams = [b for b in active_beams if not b.pruned]

                # beam score 계산 & 상위 beam_size만 남기기 (가장 간단한 예: 길이가 긴 것 선호)
                # 실제로는 logprob 등으로 정렬하는 게 일반적
                sorted_beams = sorted(
                    active_beams,
                    key=lambda x: self._aggregate_scores(x.score_history),
                    reverse=True
                )
                beams = sorted_beams[: self.beam_size]

            # 3) 모든 iteration 종료 후, completed_beams + 남은 beams 중 top beam_size 추려서 최종 완성
            # 실제로는 logprob, aggregator 사용
            final_candidates = [b for b in beams if not b.pruned]
            final_candidates += completed_beams
            final_candidates = list(set(final_candidates))  # set()은 같은 객체인지 판별 (데이터클래스 해싱 주의)
            
            # 정렬
            final_candidates = sorted(
                final_candidates,
                key=lambda x: self._aggregate_scores(x.score_history),
                reverse=True
            )[: self.beam_size]

            # 최상위 빔을 최종 prediction으로
            best_beam = final_candidates[0] if final_candidates else None
            sample["prediction"] = best_beam.current_text if best_beam else ""
            results.append(sample)

        return results
