from typing import Dict, Type
from .base import BaseEvaluator

# 1) evaluator들을 등록할 전역 레지스트리 (dict)
EVALUATION_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}

# 2) 레지스트리에 등록할 헬퍼 함수
def register_evaluator(name: str):
    """
    Evaluator 클래스를 레지스트리에 등록하기 위한 데코레이터.
    사용:
    @register_evaluator("logit_based")
    class LogitBasedEvaluator(BaseEvaluator):
        ...
    """
    def decorator(cls: Type[BaseEvaluator]):
        EVALUATION_REGISTRY[name] = cls
        return cls
    return decorator

# 3) 레지스트리에서 Evaluator 인스턴스를 생성하는 함수
def get_evaluator(name: str) -> BaseEvaluator:
    """
    문자열로 주어진 evaluator 이름(name)을 통해 해당 클래스를 찾아 인스턴스화 후 반환.
    만약 파라미터가 필요하면 여기서 추가 인자를 받을 수도 있음.
    """
    if name not in EVALUATION_REGISTRY:
        raise ValueError(f"Evaluator '{name}' not found in registry. Check available keys: {list(EVALUATION_REGISTRY.keys())}")
    evaluator_cls = EVALUATION_REGISTRY[name]
    return evaluator_cls()  # init 파라미터가 없다면 이렇게

from .string_match import StringMatchEvaluator
 
# 4) 실제 evaluator 파일들 import & 등록
# logit_based.py, exact_match.py, 등등 안에 구현된 클래스들을 import
#  -> import * 해서 해당 파일 안에서 @register_evaluator 붙인 클래스를 등록
#  -> 혹은 여기서 직접 import 후 등록해도 됨