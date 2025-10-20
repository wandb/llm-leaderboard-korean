from typing import Optional, Any
from types import SimpleNamespace
import weave
import wandb

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
    def initialize(cls, run, llm: Optional[Any] = None, wandb_params: Optional[Any] = None):
        if cls._instance is not None:
            raise Exception("WandbConfigSingleton has already been initialized")
        config_dict = dict(getattr(run, "config", {}) or {})
        if OmegaConf is not None:
            config = OmegaConf.create(config_dict)
        else:
            config = config_dict
        cls._instance = SimpleNamespace(run=run, config=config, blend_config=None, llm=llm, wandb_params=wandb_params)

    @classmethod
    def download_artifact(cls, dataset_name: str):
        api = wandb.Api()
        if "mt_bench" in dataset_name:
            artifact = api.artifact(f"wandb-korea/korean-llm-leaderboard/{dataset_name}:latest")
            artifact_path = artifact.download()
            return artifact_path
        artifact = api.artifact(f"{cls._instance.wandb_params.get('entity')}/{cls._instance.wandb_params.get('project_dataset')}/{dataset_name}:latest")
        artifact_path = artifact.download()
        return artifact_path