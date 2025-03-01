import re
import string
from typing import List, Dict, Any, Optional
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
    Evaluator that checks if the normalized reference is contained within the normalized prediction.
    
    Instead of using an exact equality comparison, this evaluator considers a prediction correct if 
    the normalized reference string is found as a substring within the normalized prediction.
    
    It supports options to ignore case, punctuation, numbers, and apply regex-based filtering, similar 
    to the StringMatchEvaluator.
    """

    name = "partial_match"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        regexes_to_ignore: Optional[List[str]] = None,
        extract_final_answer: bool = True,   # Whether to extract only the text after 'Answer:' in CoT outputs
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_numbers = ignore_numbers
        self.regexes_to_ignore = regexes_to_ignore or []
        self.extract_final_answer = extract_final_answer

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
          - If extract_final_answer is True, extract only the text after "Answer:".
          - Then normalize the text.
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
        Compute accuracy by checking if the normalized reference is a substring of the normalized prediction.
        
        Also, each sample is augmented with an "evaluation" dict containing normalized texts and correctness.
        """
        total = len(samples)
        correct = 0
        logger.info("Evaluating outputs using partial (containment) match...")

        for sample in tqdm(samples, desc="Partial-Match Evaluation"):
            pred_norm = self.parse_prediction(sample.get("prediction", ""))
            ref_norm = self.parse_prediction(sample.get("reference", ""))

            # Instead of exact equality, check if the reference is contained within the prediction.
            is_correct = (ref_norm in pred_norm)
            if is_correct:
                correct += 1

            sample["evaluation"] = {
                "normalized_pred": pred_norm,
                "normalized_ref": ref_norm,
                "is_correct": is_correct
            }

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}
