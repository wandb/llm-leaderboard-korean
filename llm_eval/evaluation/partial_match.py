import re
import string
from typing import List, Dict, Any, Optional, Union

from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.prompt_template import extract_final_answer
from llm_eval.utils.logging import get_logger
from tqdm import tqdm
import logging

logger = get_logger(name="partial_match", level=logging.INFO)

@register_evaluator("partial_match")
class PartialMatchEvaluator(BaseEvaluator):
    """
    Evaluator that checks if the normalized prediction is contained within the normalized reference.
    
    Instead of using an exact equality comparison, this evaluator considers a prediction correct if 
    the normalized prediction string is found as a substring within the normalized reference.
    
    It supports options to ignore case, punctuation, numbers, and apply regex-based filtering, similar 
    to the StringMatchEvaluator.

    When mcqa is True, and if an "options" field is provided in the sample, the evaluator compares 
    the normalized prediction against the normalized candidate options using a partial match criterion.
    """

    name = "partial_match"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        regexes_to_ignore: Optional[List[str]] = None,
        extract_final_answer: bool = True,
        mcqa: bool = True,  # Added mcqa flag
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
                If True, extracts only the text after 'Answer:' from model outputs.
            mcqa (bool):
                If True, and if sample contains an "options" field, the evaluator compares the prediction
                with the candidate options using partial matching.
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
        Normalize text by:
          1) Removing patterns specified in regexes_to_ignore.
          2) Converting to lowercase if ignore_case is enabled.
          3) Removing punctuation if ignore_punctuation is enabled.
          4) Removing numbers if ignore_numbers is enabled.
          5) Normalizing spaces.
        """
        # 1) Remove regex patterns
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
        # 5) Normalize whitespace
        text = " ".join(text.strip().split())
        return text

    def parse_prediction(self, raw_output: Any) -> str:
        """
        Process the raw output:
          - If extract_final_answer is True, extract only the text after 'Answer:'.
          - Then normalize the text.
          - If the text is long, only consider the first line.
        """
        if raw_output is None:
            raw_output = ""
        elif not isinstance(raw_output, str):
            raw_output = str(raw_output)
        if self.extract_final_answer:
            raw_output = extract_final_answer(raw_output)
        # If the extracted answer is long, take only the first line
        lines = raw_output.strip().splitlines()
        if lines:
            raw_output = lines[0]
        return self._normalize_text(raw_output)

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes accuracy based on whether the normalized prediction is contained within the normalized reference.
        
        When mcqa is enabled and an "options" field is provided, the evaluation instead checks if the normalized
        prediction is contained within any of the candidate options.
        
        Additionally, each sample is augmented with an "evaluation" dict containing normalized texts and correctness.
        """
        total = len(samples)
        correct = 0
        logger.info("Evaluating outputs using partial (containment) match with mcqa={}".format(self.mcqa))

        for sample in tqdm(samples, desc="Partial-Match Evaluation"):
            pred_norm = self.parse_prediction(sample.get("prediction", ""))
            ref_norm = self.parse_prediction(sample.get("reference", ""))

            # If mcqa enabled and options are provided, compare prediction with options using partial match.
            if self.mcqa and "options" in sample and isinstance(sample["options"], list) and sample["options"]:
                normalized_options = [self._normalize_text(opt) for opt in sample["options"]]
                # Correct if the prediction is contained within any of the candidate options.
                is_correct = any(pred_norm in opt for opt in normalized_options)
            else:
                # Otherwise, check if the normalized prediction is a substring of the normalized reference.
                is_correct = (pred_norm in ref_norm)

            if is_correct:
                correct += 1

            sample["evaluation"] = {
                "normalized_pred": pred_norm,
                "normalized_ref": ref_norm,
                "is_correct": is_correct,
                "options": sample.get("options", None)
            }

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}
