"""SequenceMatcher 기반 평가 모듈.

이 모듈은 `difflib.SequenceMatcher`를 활용하여 모델의 예측과 참조
출력이 얼마나 유사한지를 계량화합니다. 기본적으로 문자열을 그대로
비교하지만, `prefix_key`를 지정하면 특정 prefix가 존재할 때만 평가를
진행하도록 설정할 수 있습니다. 이는 MRCR(Multi-round Co-reference
Resolution)와 같은 prefix 기반 검증이 필요한 벤치마크에 유용합니다.

주요 특징:
    - 랜덤 prefix를 강제할 수 있는 옵션(`prefix_key`, `require_prefix`).
    - prefix 제거 여부(`strip_prefix`).
    - 평균 ratio와 prefix 누락 비율 등 핵심 메트릭 제공.

기본 동작은 MRCR 벤치마크에서 사용하는 규칙을 따르지만, 다른 목적에도
재사용할 수 있도록 매개변수화되어 있습니다.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from .base import BaseEvaluator
from . import register_evaluator


@register_evaluator("sequence_match")
class SequenceMatchEvaluator(BaseEvaluator):
    """일반화된 SequenceMatcher 기반 평가기."""

    name = "sequence_match"

    def __init__(
        self,
        prefix_key: Optional[str] = "random_string_to_prepend",
        require_prefix: bool = True,
        strip_prefix: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """평가기 초기화.

        Args:
            prefix_key: 메타데이터에서 prefix를 찾을 때 사용할 키. None이면 prefix 검사 생략.
            require_prefix: True일 때 prefix가 없다면 점수를 0으로 처리.
            strip_prefix: True일 때 prefix를 제거한 본문끼리 비교.
        """

        super().__init__(*args, **kwargs)
        self.prefix_key = prefix_key
        self.require_prefix = require_prefix
        self.strip_prefix = strip_prefix

    def parse_prediction(self, raw_output: Any) -> str:
        if raw_output is None:
            return ""
        return str(raw_output)

    def _get_prefix(self, sample: Dict[str, Any]) -> str:
        if not self.prefix_key:
            return ""
        metadata = sample.get("metadata", {})
        return metadata.get(self.prefix_key, "")

    def _strip(self, text: str, prefix: str) -> str:
        if prefix and text.startswith(prefix):
            return text[len(prefix):]
        return text

    def evaluate_predictions(self, subsets: Optional[List[str]], samples: List[Dict[str, Any]]) -> Dict[str, float]:
        if not samples:
            return {"sequence_match_score": 0.0}

        total_ratio = 0.0
        prefix_miss = 0
        # 요청된 subsets 기준 초기화 + all 집계
        subset_stats: Dict[str, Dict[str, float]] = {}
        if subsets:
            for s in subsets:
                subset_stats[s] = {"sum_ratio": 0.0, "count": 0.0, "prefix_miss": 0.0}
        subset_stats["all"] = {"sum_ratio": 0.0, "count": 0.0, "prefix_miss": 0.0}

        for sample in samples:
            subset_name = sample.get("_subset_name")
            if subset_name and subset_name not in subset_stats:
                subset_stats[subset_name] = {"sum_ratio": 0.0, "count": 0.0, "prefix_miss": 0.0}
            prediction = self.parse_prediction(sample.get("prediction"))
            reference = self.parse_prediction(sample.get("reference"))
            prefix = self._get_prefix(sample)

            has_prefix = True
            if self.prefix_key:
                has_prefix = bool(prefix) and prediction.startswith(prefix)
                if self.require_prefix and not has_prefix:
                    ratio = 0.0
                    prefix_miss += 1
                    if subset_name:
                        subset_stats[subset_name]["prefix_miss"] += 1
                    self._update_sample(sample, has_prefix, ratio, prefix)
                    total_ratio += ratio
                    if subset_name:
                        subset_stats[subset_name]["sum_ratio"] += ratio
                        subset_stats[subset_name]["count"] += 1
                    # all 집계
                    subset_stats["all"]["sum_ratio"] += ratio
                    subset_stats["all"]["count"] += 1
                    continue

            if self.strip_prefix and prefix:
                prediction = self._strip(prediction, prefix)
                reference = self._strip(reference, prefix)

            ratio = SequenceMatcher(None, prediction, reference).ratio()
            self._update_sample(sample, has_prefix, ratio, prefix)
            total_ratio += ratio
            if subset_name:
                subset_stats[subset_name]["sum_ratio"] += ratio
                subset_stats[subset_name]["count"] += 1
            # all 집계
            subset_stats["all"]["sum_ratio"] += ratio
            subset_stats["all"]["count"] += 1

        # 전체 메트릭 유지 (sequence_match_score, prefix_miss_rate, AVG)
        avg_ratio = total_ratio / len(samples)
        metrics: Dict[str, float] = {"sequence_match_score": avg_ratio, "AVG": avg_ratio}
        if self.prefix_key:
            metrics["sequence_match_prefix_miss_rate"] = prefix_miss / len(samples)
        # subsets 전달 여부에 따라 분기
        if subsets:
            for sname, st in subset_stats.items():
                cnt = st["count"]
                s_avg = (st["sum_ratio"] / cnt) if cnt > 0 else 0.0
                metrics[f"{sname}/AVG"] = s_avg
        return metrics

    @staticmethod
    def _update_sample(
        sample: Dict[str, Any],
        has_prefix: bool,
        ratio: float,
        prefix: str,
    ) -> None:
        sample.setdefault("evaluation", {})
        sample["evaluation"].update(
            {
                "sequence_match_prefix_present": has_prefix,
                "sequence_match_ratio": ratio,
                "sequence_match_prefix": prefix,
            }
        )


