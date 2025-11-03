from typing import Dict, Type
from .base import BaseModel

# 1) Global registry (dict) to register models
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


# 2) Helper function to register a model in the registry
def register_model(name: str):
    """
    Decorator to register a Model / Judge / Reward class in the registry.
    Example usage:
        @register_model("vllm")
        class VLLMModel(BaseModel):
            ...
    """

    def decorator(cls: Type[BaseModel]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered.")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


# 3) Function to create a model instance from the registry
def load_model(name: str, **kwargs) -> BaseModel:
    """
    Takes a string 'name', finds the corresponding model class, instantiates it, and returns the instance.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Please register it in MODEL_REGISTRY."
        )
    model_cls = MODEL_REGISTRY[name]
    return model_cls(**kwargs)


# 5) Import actual backends -> they are registered via decorators
from .openai_backend import OpenAIModel
from .openai_responses_backend import OpenAIResponsesModel
from .openai_judge import OpenAIJudge
from .multi import MultiModel
from .huggingface_backend import HuggingFaceModel
from .huggingface_judge import HuggingFaceJudge
from .huggingface_reward import HuggingFaceReward

# Optional imports for backends that may have compatibility issues
try:
    from .litellm_backend import LiteLLMBackend
    from .litellm_judge import LiteLLMJudge
except ImportError as e:
    import warnings
    warnings.warn(f"LiteLLM backend not available: {e}", ImportWarning)

try:
    from .vllm_backend import VLLMModel
except ImportError as e:
    import warnings
    warnings.warn(f"VLLM backend not available: {e}", ImportWarning)