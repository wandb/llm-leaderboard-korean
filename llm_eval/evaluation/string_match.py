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

    * If extract_final_answer is enabled, the evaluator extracts only the text after 'Answer:'.
    * Additionally, it can normalize text by ignoring case, punctuation, and numbers.
    * When mcqa is True, if an "options" field is provided in the sample, the evaluator
      compares the prediction against the normalized options.
    * If the generated text is long (e.g., contains multiple lines), only the first non-empty
      line is used for evaluation.
    * Additionally, it removes markdown formatting tokens from the final answer.
    """

    name = "string_match"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        regexes_to_ignore: Optional[List[str]] = None,
        extract_final_answer: bool = True,
        mcqa: bool = True,  # Use options-based matching for MCQA tasks if True.
        *args,
        **kwargs
    ):
        """
        Args:
            ignore_case (bool): If True, ignores case differences. (Default: True)
            ignore_punctuation (bool): If True, removes punctuation. (Default: False)
            ignore_numbers (bool): If True, removes digits. (Default: False)
            regexes_to_ignore (List[str] | None): List of regex patterns to remove. (Default: None)
            extract_final_answer (bool): If True, extracts text after 'Answer:' from CoT outputs.
            mcqa (bool): If True, and if "options" are provided in the sample, compares prediction with options.
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
        Normalizes text by:
          1) Removing regex-specified patterns.
          2) Converting to lowercase (if enabled).
          3) Removing punctuation (if enabled).
          4) Removing numbers (if enabled).
          5) Normalizing whitespace.
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

    def remove_markdown_formatting(self, text: str) -> str:
        """
        Removes markdown formatting tokens (such as **, __, ~~ and `) from the text.
        """
        return re.sub(r"(\*\*|__|~~|`)", "", text)

    def parse_prediction(self, raw_output: Any) -> str:
        """
        Processes the model's raw output:
          1) If extract_final_answer is enabled, extract the text after 'Answer:'.
          2) Remove markdown formatting tokens.
          3) If the resulting text is long (contains multiple lines), only the first non-empty line is used.
          4) Normalize the final text.
        """
        if raw_output is None:
            raw_output = ""
        elif not isinstance(raw_output, str):
            raw_output = str(raw_output)

        if self.extract_final_answer:
            raw_output = extract_final_answer(raw_output)

        # Remove markdown formatting tokens from the extracted text.
        raw_output = self.remove_markdown_formatting(raw_output)

        # Split into lines and take the first non-empty line.
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if lines:
            raw_output = lines[0]
        else:
            raw_output = ""

        return self._normalize_text(raw_output)

    def evaluate_predictions(self, subsets: Optional[List[str]], samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes accuracy by comparing the normalized prediction against the normalized reference.
        
        For MCQA (Multiple-Choice Question Answering) tasks:
        - If 'mcqa' is enabled and an 'options' list is provided in the sample,
            the evaluation considers a prediction correct only if:
            1. The normalized prediction exactly matches the normalized reference.
            2. The normalized reference is one of the normalized options.
        
        For non-MCQA tasks, the prediction is considered correct if the normalized prediction 
        exactly matches the normalized reference.
        
        Each sample is augmented with detailed evaluation information in the 'evaluation' key.
        
        Returns:
            A dictionary with the computed accuracy metric, e.g. {"accuracy": 0.85}.
        """
        total = len(samples)
        correct = 0
        # subset 별 집계를 위한 통계 구조 (요청된 subsets 기준으로 초기화) + all 집계
        stats: Dict[str, Dict[str, int]] = {}
        if subsets:
            for subset in subsets:
                stats[subset] = {"total": 0, "correct": 0}
        stats["all"] = {"total": 0, "correct": 0}

        # Log information when evaluating MCQA tasks with provided options.
        if self.mcqa and "options" in samples[0] and isinstance(samples[0]["options"], list) and samples[0]["options"]:
            logger.info(f"Evaluating outputs using string match with mcqa={self.mcqa}")

        for sample in tqdm(samples, desc="String-Match Evaluation"):
            subset_name = sample.get("_subset_name")
            # Normalize the prediction and reference texts.
            pred_norm = self.parse_prediction(sample["prediction"])
            ref_norm = self.parse_prediction(sample["reference"])

            if self.mcqa and "options" in sample and isinstance(sample["options"], list) and sample["options"]:
                # Normalize all provided options.
                normalized_options = [self._normalize_text(opt) for opt in sample["options"]]
                # For MCQA tasks, consider the prediction correct only if
                # the normalized prediction exactly matches the normalized reference
                # and the normalized reference is among the provided options.
                is_correct = (pred_norm == ref_norm) and (ref_norm in normalized_options)
            else:
                # For non-MCQA tasks, a simple exact match suffices.
                is_correct = (pred_norm == ref_norm)

            if is_correct:
                correct += 1
                if subset_name:
                    stats[subset_name]["correct"] += 1
            if subset_name:
                stats[subset_name]["total"] += 1
            # always update 'all'
            stats["all"]["total"] += 1
            if is_correct:
                stats["all"]["correct"] += 1

            # Record detailed evaluation info within the sample for debugging and analysis.
            sample["evaluation"] = {
                "normalized_pred": pred_norm,
                "normalized_ref": ref_norm,
                "is_correct": is_correct,
                "options": sample.get("options", None)
            }

        # 전체 AVG는 항상 포함
        metrics: Dict[str, float] = {"AVG": (correct / total if total > 0 else 0.0)}
        if subsets:
            for sname, st in stats.items():
                s_total = st["total"]
                s_correct = st["correct"]
                s_acc = (s_correct / s_total) if s_total > 0 else 0.0
                metrics[f"{sname}/AVG"] = s_acc
        return metrics
