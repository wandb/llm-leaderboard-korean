import re
import string
from typing import List, Dict, Any, Optional, Union

from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.prompt_template import extract_final_answer
from llm_eval.utils.logging import get_logger
import logging
from tqdm import tqdm

logger = get_logger(name="string_match", level=logging.INFO)

@register_evaluator("string_match")
class StringMatchEvaluator(BaseEvaluator):
    """
    Evaluation based on string matching, calculating accuracy by checking
    whether the predicted value (prediction) exactly matches the reference value (reference).

    * If you want to ignore Chain-of-Thought (CoT) and extract only the text after 'Answer:',
      pass 'extract_final_answer=True' as an argument.
    * Additionally, you can configure ignore_case / ignore_punctuation / ignore_numbers
      to normalize the text (remove noise) before comparison.
    * When mcqa=True, if "options" are provided in the sample, the evaluator checks if the
      normalized prediction is one of the normalized options.
    """

    name = "string_match"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        regexes_to_ignore: Optional[List[str]] = None,
        extract_final_answer: bool = True,
        mcqa: bool = True,  # 추가: mcqa 플래그
        *args,
        **kwargs
    ):
        """
        Args:
            ignore_case (bool): If True, ignores case differences. (Default: True)
            ignore_punctuation (bool): If True, removes all punctuation. (Default: False)
            ignore_numbers (bool): If True, removes all numbers. (Default: False)
            regexes_to_ignore (List[str] | None): List of regex patterns to remove. (Default: None)
            extract_final_answer (bool): If True, extracts only the text after 'Answer:' from model outputs.
            mcqa (bool): If True, and if sample contains "options", the evaluator compares the prediction with options.
        """
        super().__init__(*args, **kwargs)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_numbers = ignore_numbers
        self.regexes_to_ignore = regexes_to_ignore or []
        self.extract_final_answer = extract_final_answer
        self.mcqa = mcqa

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing specified patterns, adjusting case, punctuation, numbers,
        and extra spaces.
        """
        for pattern in self.regexes_to_ignore:
            text = re.sub(pattern, "", text)
        if self.ignore_case:
            text = text.lower()
        if self.ignore_punctuation:
            repl_table = str.maketrans("", "", string.punctuation)
            text = text.translate(repl_table)
        if self.ignore_numbers:
            repl_table = str.maketrans("", "", string.digits)
            text = text.translate(repl_table)
        text = " ".join(text.strip().split())
        return text

    def parse_prediction(self, raw_output: Any) -> str:
        """
        Extract final answer if enabled and normalize the text.
        """
        if raw_output is None:
            raw_output = ""
        elif not isinstance(raw_output, str):
            raw_output = str(raw_output)
        if self.extract_final_answer:
            raw_output = extract_final_answer(raw_output)
        return self._normalize_text(raw_output)

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes accuracy based on whether the normalized prediction matches the reference.
        If mcqa is True and the sample contains an "options" field, the prediction is compared
        against the candidate options instead.
        Also, records evaluation info in each sample.
        """
        total = len(samples)
        correct = 0
        logger.info("Evaluating outputs using string match with mcqa={}".format(self.mcqa))

        for sample in tqdm(samples, desc="String-Match Evaluation"):
            pred_norm = self.parse_prediction(sample["prediction"])
            ref_norm = self.parse_prediction(sample["reference"])
            # If mcqa enabled and options are provided, compare prediction with options.
            if self.mcqa and "options" in sample and isinstance(sample["options"], list) and sample["options"]:
                normalized_options = [self._normalize_text(opt) for opt in sample["options"]]
                is_correct = pred_norm in normalized_options
            else:
                is_correct = (pred_norm == ref_norm)

            if is_correct:
                correct += 1

            sample["evaluation"] = {
                "normalized_pred": pred_norm,
                "normalized_ref": ref_norm,
                "is_correct": is_correct,
                "options": sample.get("options", None)
            }

        accuracy = correct / total if total else 0.0
        return {"accuracy": accuracy}
