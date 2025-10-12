from typing import Dict, Type
from .base import BaseDataset

# 1) Global registry for datasets (dict)
DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}

# 2) Helper function to register a dataset class in the registry
def register_dataset(name: str):
    """
    Decorator to register a Dataset class in the registry.
    Usage example:
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

# 3) Helper function to instantiate a Dataset class
def load_datasets(name: str, split: str = "test", **kwargs) -> BaseDataset:
    """
    Given a string name, find the corresponding Dataset class and create/return an instance.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Please register it in DATASET_REGISTRY.")
    
    # The registry stores classes, so we instantiate here
    dataset_class = DATASET_REGISTRY[name]
    return dataset_class(split=split, **kwargs)

# 4) Import modules so that the decorators get executed
# Example:
from .haerae import HaeraeDataset
from .kmmlu import KMMLUDataset
from .click import ClickDataset
from .hrm8k import HRM8KDataset
from .k2_eval import K2_EvalDataset
from .kudge import KUDGEDataset
from .dataset_loader import GenericFileDataset
from .benchhub import BenchHubDataset
from .hrc import HRCDataset
from .kbl import KBLDataset
from .kormedqa import KorMedMCQADataset
from .aime2025 import AIME2025Dataset
from .kmmlu_pro import KMMLUProDataset
from .kobalt_700 import KoBALT700Dataset
from .kmmlu_hard import KMMLUHardDataset
from .bfcl import BFCLDataset
from .korean_sat import KoreanSATDataset
from .mrcr import MRCRDataset
