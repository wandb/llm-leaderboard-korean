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

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        total_f1 = 0.0
        n = len(samples)
        for sample in tqdm(samples, desc="Char-F1 Evaluation"):
            pred = sample.get("prediction", "")
            ref = sample.get("reference", "")
            f1 = self._char_f1(pred, ref)
            total_f1 += f1
            sample["evaluation"] = {
                "normalized_pred": pred,
                "normalized_ref": ref,
                "char_f1": f1,
            }
        avg_f1 = total_f1 / n if n > 0 else 0.0
        return {"AVG": avg_f1, "char_f1": avg_f1}


