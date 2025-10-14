from typing import List, Dict, Any, Optional
import re

from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.prompt_template import extract_final_answer
from llm_eval.utils.logging import get_logger
import logging
from tqdm import tqdm
from fuzzywuzzy import fuzz

logger = get_logger(name="char_f1", level=logging.INFO)


@register_evaluator("char_f1")
class CharF1Evaluator(BaseEvaluator):
    """
    Character-level F1 evaluator.

    - Normalizes predictions/references (optional lowercasing, whitespace removal, punctuation removal)
    - Computes F1 based on multiset character overlap (bag-of-chars)
    - Returns average F1 across samples as {"char_f1": value}
    """

    name = "char_f1"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_whitespace: bool = True,
        ignore_punctuation: bool = False,
        extract_final_answer: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignore_case = ignore_case
        self.ignore_whitespace = ignore_whitespace
        self.ignore_punctuation = ignore_punctuation
        self.extract_final_answer = extract_final_answer

    def remove_markdown_formatting(self, text: str) -> str:
        return re.sub(r"(\*\*|__|~~|`)", "", text)

    def _char_f1(self, pred: str, ref: str) -> float:
        # Use fuzzywuzzy token_sort_ratio as requested
        return fuzz.token_sort_ratio(pred, ref) / 100.0

    def evaluate_predictions(self, subsets: Optional[List[str]], samples: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(samples)
        # subset 별 집계를 위한 통계 구조
        stats = {}
        if subsets:
            for subset in subsets:
                stats[subset] = {"sum_f1": 0.0, "count": 0.0}
        stats["all"] = {"sum_f1": 0.0, "count": 0.0}
        for sample in tqdm(samples, desc="Char-F1 Evaluation"):
            subset_name = sample.get("_subset_name")

            pred = sample.get("prediction", "")
            ref = sample.get("reference", "")
            f1 = self._char_f1(pred, ref)

            # subset 통계 업데이트
            if subset_name:
                stats[subset_name]["sum_f1"] += f1
                stats[subset_name]["count"] += 1
            stats["all"]["sum_f1"] += f1
            stats["all"]["count"] += 1

            sample["evaluation"] = {
                "normalized_pred": pred,
                "normalized_ref": ref,
                "char_f1": f1,
            }

        # 전체 AVG는 항상 포함
        overall = stats["all"]["sum_f1"] / stats["all"]["count"] if stats["all"]["count"] > 0 else 0.0
        metrics: Dict[str, float] = {"AVG": overall}
        if subsets:
            for sname, st in stats.items():
                if sname == "all":
                    continue
                metrics[f"{sname}/AVG"] = st["sum_f1"] / st["count"] if st["count"] > 0 else 0.0

        return metrics


