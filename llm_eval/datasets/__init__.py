from typing import Dict, Type
from .base import BaseDataset

# 1) dataset들을 등록할 전역 레지스트리 (dict)
DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}

# 2) 레지스트리에 등록할 헬퍼 함수
def register_dataset(name: str):
    """
    Dataset 클래스를 레지스트리에 등록하기 위한 데코레이터.
    사용 예:
        @register_dataset("hae_rae")
        class HaeRaeDataset(BaseDataset):
            ...
    """
    def decorator(cls: Type[BaseDataset]):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' already registered.")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

# 3) 실제로 Dataset 클래스를 인스턴스화해주는 헬퍼 함수
def load_datasets(name: str, split: str = "test", **kwargs) -> BaseDataset:
    """
    문자열 이름(name)을 받아 해당 데이터셋 클래스를 찾아 인스턴스를 생성/반환.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Please register it in DATASET_REGISTRY.")
    
    # registry에 저장된 것은 '클래스'이므로 여기서 instantiate
    dataset_class = DATASET_REGISTRY[name]
    return dataset_class(split=split, **kwargs)

# 4) 실제로 .py 모듈들을 import 하여 데코레이터가 실행되도록 함
# 예:
from .haerae import HaeraeDataset
from .kmmlu import KMMLUDataset
from .click import ClickDataset
from .hrm8k import HRM8KDataset
from .k2_eval import K2_EvalDataset
from .kudge import KUDGEDataset
from dataset_loader import GenericFileDataset
# ...
