from typing import List, Dict, Any

class BaseModel:
    """
    (개발 용이를 위해 개발 단계에서는 한국어로 작성, 이후 영어로 대체)
    모든 모델 백엔드가 상속해야 할 추상 클래스.

    필수 구현 메서드:
      - generate_batch(self, inputs, return_logits=False) -> List[Dict[str, Any]]
        * inputs: [{"input": str, "reference": str, ...}, ...]
        * return_logits: True일 경우 logits이나 log_probs 정보를 추가로 반환할 수도 있음
        * 반환: [{"input":..., "reference":..., 
                  "prediction":...,        # 모델 최종 문자열 출력
                  "logits": (optional)..., # if return_logits=True
                  ...}, ...]
    """

    def __init__(self, **kwargs):
        # config, endpoint, 인증토큰 등 필요한 거 저장
        pass

    def generate_batch(
        self, 
        inputs: List[Dict[str, Any]], 
        return_logits: bool = False
    ) -> List[Dict[str, Any]]:
        """
        실제 모델 호출(로컬 or API)을 통해 'prediction' (문자열) 
        + 필요한 경우 'logits' 정보를 반환.
        """
        raise NotImplementedError("Subclasses must implement generate_batch().")
