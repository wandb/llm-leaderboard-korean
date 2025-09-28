from typing import Dict, Type
from .base import BaseEvaluator

# 1) Global registry (dict) for registering evaluator classes
EVALUATION_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}

# 2) Helper function to register evaluator classes in the registry
def register_evaluator(name: str):
    """
    A decorator to register an Evaluator class in the registry.
    
    Usage:
    @register_evaluator("logit_based")
    class LogitBasedEvaluator(BaseEvaluator):
        ...
    """
    def decorator(cls: Type[BaseEvaluator]):
        EVALUATION_REGISTRY[name] = cls
        return cls
    return decorator

# 3) Function to create an instance of an Evaluator from the registry
def get_evaluator(name: str, **kwargs) -> BaseEvaluator:
    """
    Retrieves the evaluator class by its name (string), instantiates it with any additional parameters, and returns the instance.
    """
    if name not in EVALUATION_REGISTRY:
        raise ValueError(f"Evaluator '{name}' not found in registry. Available keys: {list(EVALUATION_REGISTRY.keys())}")
    evaluator_cls = EVALUATION_REGISTRY[name]
    return evaluator_cls(**kwargs)

# 4) Import and register actual evaluator files
# Import evaluator implementations from files such as logit_based.py, exact_match.py, etc.
# This can be done via wildcard imports where the classes decorated with @register_evaluator will register themselves,
# or by directly importing them here.

from .string_match import StringMatchEvaluator
from .llm_judge import LLMJudgeEvaluator
from .math_eval import MathMatchEvaluator
from .partial_match import PartialMatchEvaluator
from .log_prob import LogProbEvaluator
from .ifeval import IFEvalStrictEvaluator, IFEvalLooseEvaluator
