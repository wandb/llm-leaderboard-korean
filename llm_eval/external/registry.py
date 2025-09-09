from typing import Dict, Type

from .provider_base import ExternalProvider


PROVIDER_REGISTRY: Dict[str, Type[ExternalProvider]] = {}


def register_provider(name: str):
    def decorator(cls: Type[ExternalProvider]):
        if name in PROVIDER_REGISTRY:
            raise ValueError(f"External provider '{name}' already registered.")
        PROVIDER_REGISTRY[name] = cls
        return cls

    return decorator


def get_provider(name: str) -> ExternalProvider:
    if name not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown external provider: {name}")
    return PROVIDER_REGISTRY[name]()
