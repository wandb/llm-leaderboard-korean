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
    """

    name = "string_match"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        regexes_to_ignore: Optional[List[str]] = None,
        extract_final_answer: bool = True,   # Whether to extract only the text after "Answer:" in CoT outputs
        *args,
        **kwargs
    ):
        """
        Args:
            ignore_case (bool):
                If True, ignores case differences. (Default: True)
            ignore_punctuation (bool):
                If True, removes all punctuation. (Default: False)
            ignore_numbers (bool):
                If True, removes all numbers. (Default: False)
            regexes_to_ignore (List[str] | None):
                List of string patterns (regular expressions) to remove. (Default: None)
            extract_final_answer (bool):
                If True, extracts only the text after 'Answer:' from model outputs
                that include Chain-of-Thought reasoning.
        """
        super().__init__(*args, **kwargs)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_numbers = ignore_numbers
        self.regexes_to_ignore = regexes_to_ignore or []
        self.extract_final_answer = extract_final_answer

    def _normalize_text(self, text: str) -> str:
        """
        Text normalization pipeline (applied in order):
          1) Remove patterns specified in regexes_to_ignore
          2) Convert to lowercase if ignore_case is enabled
          3) Remove punctuation if ignore_punctuation is enabled
          4) Remove numbers if ignore_numbers is enabled
          5) Normalize spaces (convert multiple spaces into a single space)
        """
        # 1) Remove specified regex patterns
        for pattern in self.regexes_to_ignore:
            text = re.sub(pattern, "", text)

        # 2) Convert to lowercase
        if self.ignore_case:
            text = text.lower()

        # 3) Remove punctuation
        if self.ignore_punctuation:
            repl_table = str.maketrans("", "", string.punctuation)
            text = text.translate(repl_table)

        # 4) Remove numbers
        if self.ignore_numbers:
            repl_table = str.maketrans("", "", string.digits)
            text = text.translate(repl_table)

        # 5) Normalize spaces
        text = " ".join(text.strip().split())
        return text

    def parse_prediction(self, raw_output: Any) -> str:
        """
        Since model outputs or reference values may include Chain-of-Thought (CoT),
        this method extracts only the text after 'Answer:' if extract_final_answer is enabled.

        After that, _normalize_text() is applied for preprocessing.
        """
        if raw_output is None:
            raw_output = ""
        elif not isinstance(raw_output, str):
            raw_output = str(raw_output)

        # 1) Extract final answer if enabled
        if self.extract_final_answer:
            raw_output = extract_final_answer(raw_output)

        # 2) Normalize text
        return self._normalize_text(raw_output)

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes accuracy based on whether the normalized prediction
        and reference values are exactly the same.

        Additionally, each sample will have an "evaluation" field with
        normalized_pred, normalized_ref, and is_correct.
        """
        total = len(samples)
        correct = 0
        logger.info("Evaluating outputs using string match...")

        for sample in tqdm(samples, desc="String-Match Evaluation"):
            pred_norm = self.parse_prediction(sample["prediction"])
            ref_norm = self.parse_prediction(sample["reference"])

            # Check if they match
            is_correct = (pred_norm == ref_norm)
            if is_correct:
                correct += 1

            ### CHANGED / ADDED SECTION ###
            # Record the evaluation info
            sample["evaluation"] = {
                "normalized_pred": pred_norm,
                "normalized_ref": ref_norm,
                "is_correct": is_correct
            }
            ### CHANGED / ADDED SECTION ###

        accuracy = correct / total if total else 0.0
        return {"accuracy": accuracy}
