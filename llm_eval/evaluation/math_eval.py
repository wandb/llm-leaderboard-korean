import re
from typing import List, Dict, Any, Optional, Union
from .base import BaseEvaluator
from . import register_evaluator
from math_verify import parse, verify
from math_verify import LatexExtractionConfig, ExprExtractionConfig
from llm_eval.utils.logging import get_logger
import logging
from tqdm import tqdm

logger = get_logger(name="math_match", level=logging.INFO)

@register_evaluator("math_match")
class MathMatchEvaluator(BaseEvaluator):
    """
    Mathematical expression evaluator that uses math_verify to compare predictions 
    against reference answers. Handles LaTeX and mathematical expressions, comparing
    them for mathematical equivalence rather than string equality.

    Features:
    - Supports both LaTeX and plain mathematical expressions
    - Handles set operations, equations, and other mathematical notations
    - Can extract final answers from Chain-of-Thought responses
    
    Example Usage:
        from llm_eval.evaluation.math_eval import MathMatchEvaluator
        
        evaluator = MathMatchEvaluator(
            latex_only=True,  # If you're only dealing with LaTeX
            extract_final_answer=False  # If you need to extract from CoT set to True
        )
        
        # Example evaluation
        results = evaluator.evaluate_predictions([
            {
                "prediction": "정답은 \\boxed{1,2,3,4\\text{inch}} 입니다.",
                "reference": "${1,3} \\cup {2,4}$"
            }
        ])
        
        # results will contain:
        # {
        #     "accuracy": 1.0,  # if expressions are mathematically equivalent
        #     "parse_failure_rate": 0.0,
        #     "verify_failure_rate": 0.0
        # }
    """
    name = "math_match"

    def __init__(
        self,
        latex_only: bool = True,  # If True, only use LaTeX extraction
        expr_only: bool = False,  # If True, only use expression extraction
        extract_final_answer: bool = True,  # Extract after "정답:" or "Answer:"
        answer_patterns: List[str] = None,  # Custom patterns for answer extraction
        *args,
        **kwargs
    ):
        """
        Args:
            latex_only (bool):
                If True, only attempts LaTeX extraction. (Default: True)
            expr_only (bool):
                If True, only attempts expression extraction. (Default: False)
            extract_final_answer (bool):
                If True, extracts answer after keywords like "정답:" or "Answer:"
            answer_patterns (List[str] | None):
                Custom regex patterns for answer extraction
        """
        super().__init__(*args, **kwargs)
        self.latex_only = latex_only
        self.expr_only = expr_only
        self.extract_final_answer = extract_final_answer
        self.answer_patterns = answer_patterns or [
            r"정답\s*:\s*(.*?)(?:\n|$)",
            r"Answer\s*:\s*(.*?)(?:\n|$)"
        ]

        # Configure extraction based on settings
        self.extraction_config = []
        if latex_only:
            self.extraction_config.append(LatexExtractionConfig())
        elif expr_only:
            self.extraction_config.append(ExprExtractionConfig())
        else:
            self.extraction_config.extend([
                LatexExtractionConfig(),
                ExprExtractionConfig()
            ])

    def extract_answer(self, text: str) -> str:
        """
        Extracts the final answer from text containing Chain-of-Thought reasoning.
        """
        if not text or not self.extract_final_answer:
            return text

        for pattern in self.answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        return text

    def parse_math(self, text: str) -> Any:
        """
        Parses mathematical expressions using math_verify.
        Returns the parsed mathematical object or None if parsing fails.
        """
        try:
            return parse(text, extraction_config=self.extraction_config)
        except Exception as e:
            self.logger.warning(f"Failed to parse mathematical expression: {text}")
            self.logger.warning(f"Error: {str(e)}")
            return None

    def verify_equivalent(self, pred: Any, ref: Any) -> bool:
        """
        Verifies if two mathematical expressions are equivalent.
        Returns False if either expression is None or verification fails.
        """
        if pred is None or ref is None:
            return False
            
        try:
            return verify(ref, pred)
        except Exception as e:
            self.logger.warning(f"Verification failed between: {pred} and {ref}")
            self.logger.warning(f"Error: {str(e)}")
            return False

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluates mathematical expressions for equivalence and calculates accuracy.
        
        Returns:
            Dict containing accuracy score and optionally detailed metrics.
        """
        total = len(samples)
        correct = 0
        parse_failures = 0
        verify_failures = 0

        for sample in samples:
            # Extract final answers if needed
            pred_text = self.extract_answer(str(sample.get("prediction", "")))
            ref_text = self.extract_answer(str(sample.get("reference", "")))

            # Parse mathematical expressions
            pred = self.parse_math(pred_text)
            ref = self.parse_math(ref_text)

            if pred is None or ref is None:
                parse_failures += 1
                continue

            # Verify mathematical equivalence
            if self.verify_equivalent(pred, ref):
                correct += 1
            else:
                verify_failures += 1

        metrics = {
            "accuracy": correct / total if total else 0.0,
            "parse_failure_rate": parse_failures / total if total else 0.0,
            "verify_failure_rate": verify_failures / total if total else 0.0
        }

        return metrics
