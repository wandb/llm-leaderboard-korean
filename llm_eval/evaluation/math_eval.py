import re
from typing import List, Dict, Any, Optional, Union
from .base import BaseEvaluator
from . import register_evaluator
from math_verify import parse, verify
from math_verify import LatexExtractionConfig, ExprExtractionConfig
from llm_eval.utils.logging import get_logger
from llm_eval.utils.prompt_template import extract_final_answer
import logging
from tqdm import tqdm
from fractions import Fraction

logger = get_logger(name="math_match", level=logging.INFO)

@register_evaluator("math_match")
class MathMatchEvaluator(BaseEvaluator):
    """
    Mathematical expression evaluator that uses math_verify to compare predictions 
    against reference answers. It handles LaTeX and mathematical expressions, 
    comparing them for mathematical equivalence rather than simple string equality.
    
    Features:
    - Supports both LaTeX and plain mathematical expressions.
    - Handles set operations, equations, and other mathematical notations.
    - Can extract final answers from Chain-of-Thought (CoT) responses using extract_final_answer.
    - Stores extracted and parsed answers for debugging or analysis.
    
    Example Usage:
        from llm_eval.evaluation.math_eval import MathMatchEvaluator
        
        evaluator = MathMatchEvaluator(
            latex_only=True,           # If you're only dealing with LaTeX
            extract_final_answer=True  # To extract from Chain-of-Thought responses
        )
        
        # Example evaluation
        results = evaluator.evaluate_predictions([
            {
                "prediction": "Answer: \\boxed{1,2,3,4\\text{inch}}",
                "reference": "${1,3} \\cup {2,4}$"
            }
        ])
        
        # results will contain:
        # {
        #     "accuracy": 1.0,           # if expressions are mathematically equivalent
        #     "parse_failure_rate": 0.0,
        #     "verify_failure_rate": 0.0
        # }
    """
    name = "math_match"

    def __init__(
        self,
        latex_only: bool = True,  # If True, only use LaTeX extraction
        expr_only: bool = False,  # If True, only use expression extraction
        extract_final_answer: bool = True,  # Extract final answer using extract_final_answer
        answer_patterns: List[str] | None = None,  # Custom regex patterns for answer extraction (not used in this implementation)
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
                If True, extracts the final answer using keywords like "Answer:" or "정답:".
            answer_patterns (List[str] | None):
                Custom regex patterns for answer extraction (not used in this implementation).
        """
        super().__init__(*args, **kwargs)
        self.latex_only = latex_only
        self.expr_only = expr_only
        self.extract_final_answer = extract_final_answer
        # Although answer_patterns are defined, they are not used directly here.
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
        Extracts the final answer from a Chain-of-Thought (CoT) text.
        If extract_final_answer is enabled, it uses the extract_final_answer function
        to retrieve the answer following keywords like "정답:" or "Answer:".
        It prioritizes extracting content from \\boxed{...} if present, otherwise
        it takes the last non-empty line from the extracted text.
        """
        if not text or not self.extract_final_answer:
            return text

        # extract_final_answer 함수를 사용해 정답 부분 추출
        answer_text = extract_final_answer(text)

        # 1. \\boxed{...} 내부를 균형 괄호로 정확히 추출 (중첩 지원)
        boxed_content = self._extract_boxed_balanced(answer_text)
        if boxed_content:
            return boxed_content

        # 2. \\boxed{...} 패턴이 없다면, 비어있지 않은 마지막 줄을 반환
        lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
        return lines[-1] if lines else answer_text

    def parse_math(self, text: str) -> Any:
        """
        Parses mathematical expressions using math_verify.
        Returns the parsed mathematical object or None if parsing fails.
        """
        def _is_invalid_parse(obj: Any) -> bool:
            return self._is_invalid_parse(obj)

        # 1차 시도: 현재 설정된 extraction_config로 파싱
        try:
            primary = parse(text, extraction_config=self.extraction_config)
        except Exception as e:
            logger.warning(f"Failed to parse mathematical expression: {text}")
            logger.warning(f"Error: {str(e)}")
            primary = None

        if not _is_invalid_parse(primary):
            return primary

        # 2차 시도: 설정에 따른 반대 추출기로 폴백
        try:
            secondary = self._parse_with_fallback_extractors(text)
            if not _is_invalid_parse(secondary):
                return secondary
        except Exception as e:
            logger.warning(f"Fallback parse failed for: {text}")
            logger.warning(f"Error: {str(e)}")

        return None

    def verify_equivalent(self, pred: Any, ref: Any) -> bool:
        """
        Verifies if two mathematical expressions are equivalent.
        Returns False if verification fails.
        """
        if pred is None or ref is None:
            return False

        try:
            return verify(ref, pred)
        except Exception as e:
            logger.warning(f"Verification failed between: {pred} and {ref}")
            logger.warning(f"Error: {str(e)}")
            return False

    def evaluate_predictions(self, subsets: Optional[List[str]], samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluates mathematical expressions for equivalence and calculates accuracy.
        Additionally, stores extracted and parsed answers for debugging or analysis.
        
        For each sample, an "evaluation" field is added with:
            - "extracted_pred": The final extracted prediction text.
            - "extracted_ref": The final extracted reference text.
            - "parsed_pred": The parsed prediction object.
            - "parsed_ref": The parsed reference object.
            - "is_correct": Boolean indicating whether the prediction is mathematically equivalent to the reference.
            - "fallback_used": Boolean indicating if fallback string comparison was used.
        
        Returns:
            A dictionary containing metrics:
                - "accuracy": Correct predictions / total samples.
                - "parse_failure_rate": Fraction of samples where parsing failed.
                - "verify_failure_rate": Fraction of samples where verification failed.
        """
        total = len(samples)
        correct = 0
        parse_failures = 0
        verify_failures = 0
        # 요청된 subsets 기준 초기화
        subset_stats: Dict[str, Dict[str, int]] = {}
        if subsets:
            for s in subsets:
                subset_stats[s] = {"total": 0, "correct": 0}

        for sample in tqdm(samples, desc="Math-Match Evaluation"):
            subset_name = sample.get("_subset_name")
            if subset_name and subset_name not in subset_stats:
                subset_stats[subset_name] = {"total": 0, "correct": 0}
            # Extract final answers
            extracted_pred = self.extract_answer(str(sample.get("prediction", "")))
            extracted_ref = self.extract_answer(str(sample.get("reference", "")))

            # Parse mathematical expressions
            parsed_pred = self.parse_math(extracted_pred)
            parsed_ref = self.parse_math(extracted_ref)

            fallback_used = False
            # If parsing fails (None/빈 구조) for either expression, try numeric equivalence then fall back to string equality
            def _is_invalid_parse(obj: Any) -> bool:
                return self._is_invalid_parse(obj)

            if _is_invalid_parse(parsed_pred) or _is_invalid_parse(parsed_ref):
                # 1) 숫자 동등성 비교 시도 (예: 2.0 == 2, 1/2 == 0.5, \frac{1}{2} == 0.5)
                def _numeric_equivalent(a_text: str, b_text: str, abs_tol: float = 1e-9, rel_tol: float = 1e-6) -> bool:
                    a_val = self._parse_numeric_value(a_text)
                    b_val = self._parse_numeric_value(b_text)
                    if a_val is None or b_val is None:
                        return False
                    diff = abs(a_val - b_val)
                    scale = max(1.0, abs(b_val))
                    return diff <= max(abs_tol, rel_tol * scale)

                if _numeric_equivalent(extracted_pred, extracted_ref):
                    is_correct = True
                    fallback_used = True
                    parse_failures += 1
                else:
                    logger.warning(f"Parse failure for sample:\npred: {extracted_pred}\nref: {extracted_ref}\nFalling back to string match.")
                    is_correct = extracted_pred.strip() == extracted_ref.strip()
                    fallback_used = True
                    parse_failures += 1
            else:
                try:
                    is_correct = self.verify_equivalent(parsed_pred, parsed_ref)
                except Exception as e:
                    logger.warning(f"Verification exception for sample:\npred: {parsed_pred}\nref: {parsed_ref}\nFalling back to string match.")
                    is_correct = extracted_pred.strip() == extracted_ref.strip()
                    fallback_used = True
                    verify_failures += 1

            if is_correct:
                correct += 1
                if subset_name:
                    subset_stats[subset_name]["correct"] += 1
            if subset_name:
                subset_stats[subset_name]["total"] += 1

            sample["evaluation"] = {
                "extracted_pred": extracted_pred,
                "extracted_ref": extracted_ref,
                "parsed_pred": parsed_pred,
                "parsed_ref": parsed_ref,
                "is_correct": is_correct,
                "fallback_used": fallback_used
            }

        # 전체 메트릭 유지 (accuracy) + 실패율
        overall_acc = correct / total if total else 0.0
        metrics: Dict[str, float] = {
            "AVG": overall_acc,
            "accuracy": overall_acc,
            "parse_failure_rate": parse_failures / total if total else 0.0,
            "verify_failure_rate": verify_failures / total if total else 0.0,
        }
        if subsets:
            for sname, st in subset_stats.items():
                s_total = st["total"]
                s_correct = st["correct"]
                s_acc = (s_correct / s_total) if s_total > 0 else 0.0
                metrics[f"{sname}/accuracy"] = s_acc
        return metrics

    # -------------------- Private helper methods --------------------
    def _extract_boxed_balanced(self, s: str) -> Optional[str]:
        marker = r"\boxed{"
        start_idx = s.find(marker)
        if start_idx == -1:
            return None
        i = start_idx + len(marker)
        depth = 1
        content_chars: List[str] = []
        while i < len(s):
            ch = s[i]
            if ch == '{':
                depth += 1
                content_chars.append(ch)
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return ''.join(content_chars).strip()
                content_chars.append(ch)
            else:
                content_chars.append(ch)
            i += 1
        return ''.join(content_chars).strip() if content_chars else None

    def _is_invalid_parse(self, obj: Any) -> bool:
        if obj is None:
            return True
        if isinstance(obj, str):
            return obj.strip() == ""
        if isinstance(obj, (list, tuple, set, dict)):
            return len(obj) == 0
        return False

    def _parse_with_fallback_extractors(self, text: str) -> Any:
        if self.latex_only:
            return parse(text, extraction_config=[ExprExtractionConfig()])
        if self.expr_only:
            return parse(text, extraction_config=[LatexExtractionConfig()])
        # 혼합 모드에서는 순서를 바꿔 재시도
        return parse(text, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])

    def _parse_numeric_value(self, text_value: Optional[str]) -> Optional[float]:
        if text_value is None:
            return None
        s = text_value.strip()
        if s == "":
            return None
        # 천 단위 구분자 제거
        s = s.replace(",", "")
        # LaTeX 분수 \frac{a}{b}
        m = re.fullmatch(r"\\frac\{\s*([+-]?\d+(?:\.\d+)?)\s*\}\{\s*([+-]?\d+(?:\.\d+)?)\s*\}", s)
        if m:
            try:
                num = float(m.group(1))
                den = float(m.group(2))
                if den == 0:
                    return None
                return num / den
            except Exception:
                return None
        # 단순 분수 a/b
        m2 = re.fullmatch(r"\(?\s*([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)\s*\)?", s)
        if m2:
            try:
                num = float(m2.group(1))
                den = float(m2.group(2))
                if den == 0:
                    return None
                return num / den
            except Exception:
                return None
        # 일반 실수/정수/지수표기
        try:
            return float(s)
        except Exception:
            return None