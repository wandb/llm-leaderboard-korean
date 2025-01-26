import re
import string
from typing import List, Dict, Any, Optional, Union

from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.prompt_template import extract_final_answer

@register_evaluator("string_match")
class StringMatchEvaluator(BaseEvaluator):
    """
    문자열 기반 평가(Evaluation)로, 예측값(prediction)과 참조값(reference)의
    정확 일치 여부(accuracy)를 계산한다.

    * Chain-of-thought(CoT)을 무시하고 '정답:' 뒤에 오는 텍스트만 추출하고 싶다면
      'extract_final_answer=True'를 인자로 넘긴다.

    * 추가적으로, ignore_case / ignore_punctuation / ignore_numbers 등을 설정해
      텍스트를 정규화(노이즈 제거) 후 비교할 수도 있다.
    """

    name = "string_match"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        regexes_to_ignore: Optional[List[str]] = None,
        extract_final_answer: bool = True,   # CoT 중 "정답:" 뒤만 추출할지 여부
        *args,
        **kwargs
    ):
        """
        Args:
            ignore_case (bool):
                True면 대소문자 차이를 무시한다. (기본값: True)
            ignore_punctuation (bool):
                True면 문장부호를 모두 제거한다. (기본값: False)
            ignore_numbers (bool):
                True면 숫자를 제거한다. (기본값: False)
            regexes_to_ignore (List[str] | None):
                제거할 문자열 패턴(정규 표현식) 목록 (기본값: None)
            extract_final_answer (bool):
                True면, Chain-of-thought가 섞여 있는 모델 출력에서
                '정답:' / 'Answer:' 등 키워드 뒤의 텍스트만 추출.
        """
        super().__init__(*args, **kwargs)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_numbers = ignore_numbers
        self.regexes_to_ignore = regexes_to_ignore or []
        self.extract_final_answer = extract_final_answer

    def _normalize_text(self, text: str) -> str:
        """
        문자열 정규화 파이프라인 (순서대로 적용):
          1) regexes_to_ignore: 추가 패턴 제거
          2) 대소문자 무시 -> lower()
          3) 문장부호 제거
          4) 숫자 제거
          5) 공백 정규화 (여러 공백 -> 단일 공백)
        """
        # 1) regex 패턴 제거
        for pattern in self.regexes_to_ignore:
            text = re.sub(pattern, "", text)

        # 2) ignore_case
        if self.ignore_case:
            text = text.lower()

        # 3) 문장부호 제거
        if self.ignore_punctuation:
            repl_table = str.maketrans("", "", string.punctuation)
            text = text.translate(repl_table)

        # 4) 숫자 제거
        if self.ignore_numbers:
            repl_table = str.maketrans("", "", string.digits)
            text = text.translate(repl_table)

        # 5) 공백 정규화
        text = " ".join(text.strip().split())
        return text

    def parse_prediction(self, raw_output: Any) -> str:
        """
        모델 또는 참조값이 체인 오브 쏘트(CoT) 형태로 주어졌을 수 있으므로,
        extract_final_answer 옵션에 따라 '정답:' 뒤의 부분만 추출할 수 있다.

        그 뒤, _normalize_text()로 전처리.
        """
        if raw_output is None:
            raw_output = ""
        elif not isinstance(raw_output, str):
            raw_output = str(raw_output)

        # 1) CoT를 무시하고 최종 정답만 파싱
        if self.extract_final_answer:
            raw_output = extract_final_answer(raw_output)

        # 2) 정규화
        return self._normalize_text(raw_output)

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        최종적으로 normalize된 prediction과 reference가 같은지 여부로 정확도(accuracy) 계산.
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
