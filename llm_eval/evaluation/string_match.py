from typing import List, Dict, Any
from .base import BaseEvaluator

class StringMatchEvaluator(BaseEvaluator):
    """
    문자열 정확도를 평가하는 Evaluator.
    prediction과 reference 간의 정확한 일치(exact match) 여부를 검사하며,
    공백과 대소문자를 무시한 비교를 수행합니다.
    
    Metrics:
        - accuracy: 전체 샘플 중 정확히 일치하는 샘플의 비율
    """
    name = "string_match"

    def parse_prediction(self, raw_output: str) -> str:
        """
        예측 값을 파싱하여 비교하기 쉽게 정제합니다.
        
        Args:
            raw_output (str): 모델의 원본 출력
            
        Returns:
            str: 정제된 문자열 (공백/대소문자 정규화)
        """
        if raw_output is None:
            return "none"
        if not isinstance(raw_output, str):
            return str(raw_output).lower()  # 문자열이 아닌 경우 변환 후 소문자로
        return " ".join(raw_output.strip().lower().split())  # 연속된 공백도 정규화

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        예측값과 참조값을 비교하여 정확도를 계산합니다.
        
        Args:
            samples: 평가할 샘플들의 리스트
                - prediction: 모델의 예측값
                - reference: 정답값
                
        Returns:
            Dict[str, float]: 평가 메트릭
                - accuracy: 정확도 (0.0 ~ 1.0)
        """
        total = len(samples)
        correct = 0
        
        for sample in samples:
            pred = self.parse_prediction(sample["prediction"])
            ref = self.parse_prediction(sample["reference"])
            if pred == ref:
                correct += 1
                
        return {"accuracy": correct / total if total > 0 else 0.0}