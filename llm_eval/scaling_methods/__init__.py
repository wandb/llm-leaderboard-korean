from typing import Dict, Type
from .base import BaseScalingMethod

# 1) Global registry (dict) for registering scaling methods
SCALING_REGISTRY: Dict[str, Type[BaseScalingMethod]] = {}

# 2) Helper function to register a scaling method in the registry
def register_scaling_method(name: str):
    """
    Decorator for registering a ScalingMethod class in the registry.
    Usage:
        @register_scaling_method("best_of_n")
        class BestOfN(BaseScalingMethod):
            ...
    """
    def decorator(cls: Type[BaseScalingMethod]):
        if name in SCALING_REGISTRY:
            raise ValueError(f"Scaling method '{name}' already registered.")
        SCALING_REGISTRY[name] = cls
        return cls
    return decorator

# 3) Helper function to instantiate a ScalingMethod from the registry using its name
def load_scaling_method(name: str, **kwargs) -> BaseScalingMethod:
    """
    Retrieve a ScalingMethod class by its string name from the registry, instantiate it, and return the instance.
    """
    if name not in SCALING_REGISTRY:
        raise ValueError(f"Unknown scaling method: {name}. Available: {list(SCALING_REGISTRY.keys())}")
    scaling_cls = SCALING_REGISTRY[name]
    return scaling_cls(**kwargs)

# 4) Import modules to trigger registration via decorators
from .self_consistency import SelfConsistencyScalingMethod
from .best_of_n import BestOfN
from .beam_search import BeamSearch
