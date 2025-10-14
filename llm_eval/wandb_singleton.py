from typing import Optional, Any
from types import SimpleNamespace

try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:
    OmegaConf = None  # type: ignore


class WandbConfigSingleton:
    _instance: Optional[SimpleNamespace] = None

    @classmethod
    def get_instance(cls) -> SimpleNamespace:
        if cls._instance is None:
            raise Exception("WandbConfigSingleton has not been initialized")
        return cls._instance

    @classmethod
    def initialize(cls, run, llm: Optional[Any] = None):
        if cls._instance is not None:
            raise Exception("WandbConfigSingleton has already been initialized")
        config_dict = dict(getattr(run, "config", {}) or {})
        if OmegaConf is not None:
            config = OmegaConf.create(config_dict)
        else:
            config = config_dict
        cls._instance = SimpleNamespace(run=run, config=config, blend_config=None, llm=llm)


