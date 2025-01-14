from typing import List, Dict, Any
from llm_eval.models.base import BaseModel

class BaseScalingMethod:
    """
    Scaling Method Base 클래스.

    * 핵심 아이디어:
      1) data(샘플 리스트)와 모델(BaseModel)을 입력받아,
      2) 여러 후보 생성/검색 전략을 사용하여 "prediction" 필드를 업데이트한 뒤
      3) 동일한 데이터 리스트(또는 복사본)를 반환한다.

    * data 형식 예시:
      [
        {"input": str, "reference": str, ...},
        {"input": str, "reference": str, ...},
        ...
      ]

    * 반환 형식:
      [
        {"input":..., "reference":..., "prediction": "...", ...},
        ...
      ]
    """

    def __init__(self, model: BaseModel = None, use_cot: bool = False, **kwargs):
        self.model = model
        self.use_cot = use_cot
        self.kwargs = kwargs  # n, beam_width, etc.

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        스케일링 로직의 메인 엔트리 포인트.
        Args:
            data: [{"input":..., "reference":...}, ...]

        Returns:
            data와 동일한 구조이되, "prediction" 필드가 업데이트된 리스트.
        """
        raise NotImplementedError("Subclasses must implement apply().")
    
    def set_params(self, **kwargs):
        """파라미터 업데이트 (n, beam_width 등)."""
        self.kwargs.update(kwargs)
