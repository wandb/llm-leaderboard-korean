import re
import string
from typing import List, Dict, Any, Optional, Union

from .base import BaseEvaluator
from . import register_evaluator

@register_evaluator("string_match")
class StringMatchEvaluator(BaseEvaluator):
    """
    문자열 정확도를 평가하는 Evaluator.
    다양한 옵션(ignore_case, ignore_punctuation, ignore_numbers, regexes_to_ignore)을
    통해 텍스트 정규화 과정을 커스터마이징할 수 있다.

    Metrics:
        - accuracy: 전체 샘플 중 정규화된 예측값과 정답값이 일치하는 샘플의 비율
    """
    name = "string_match"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        regexes_to_ignore: Optional[List[str]] = None,
        *args,
        **kwargs
    ):
        """
        Args:
            ignore_case (bool): 대소문자 구분을 무시할지 여부 (default: True)
            ignore_punctuation (bool): 문장부호를 제거할지 여부 (default: False)
            ignore_numbers (bool): 숫자를 제거할지 여부 (default: False)
            regexes_to_ignore (List[str]): 정규표현식 리스트. 매칭되면 제거할 문자열 패턴 (default: None)
        """
        super().__init__(*args, **kwargs)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_numbers = ignore_numbers
        self.regexes_to_ignore = regexes_to_ignore or []

    def _normalize_text(self, text: str) -> str:
        """
        문자열 정규화 함수.
        1) regexes_to_ignore 패턴 제거
        2) 대소문자 무시 (옵션)
        3) 문장부호 제거 (옵션)
        4) 숫자 제거 (옵션)
        5) 공백 정규화
        """
        # 1) regex 패턴 제거
        for pattern in self.regexes_to_ignore:
            text = re.sub(pattern, "", text)

        # 2) 대소문자 무시
        if self.ignore_case:
            text = text.lower()

        # 3) 문장부호 제거
        if self.ignore_punctuation:
            # str.maketrans를 활용한 제거
            repl_table = str.maketrans("", "", string.punctuation)
            text = text.translate(repl_table)

        # 4) 숫자 제거
        if self.ignore_numbers:
            repl_table = str.maketrans("", "", string.digits)
            text = text.translate(repl_table)

        # 5) 공백 정규화 (연속된 공백 -> 단일 공백)
        text = " ".join(text.strip().split())
        return text

    def parse_prediction(self, raw_output: Any) -> str:
        """
        모델/참조의 원본 텍스트를 받아 _normalize_text 과정을 거친 뒤 반환.
        None 등 예외값은 빈 문자열로 처리.
        """
        if raw_output is None:
            raw_output = ""
        elif not isinstance(raw_output, str):
            raw_output = str(raw_output)

        return self._normalize_text(raw_output)

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        예측값(prediction)과 참조값(reference)을 정규화하여 정확도(accuracy)를 계산.
        """
        total = len(samples)
        correct = 0

        for sample in samples:
            pred = self.parse_prediction(sample["prediction"])
            ref = self.parse_prediction(sample["reference"])

            if pred == ref:
                correct += 1

        accuracy = correct / total if total else 0.0
        return {"accuracy": accuracy}
