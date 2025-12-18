"""
Configuration Loader

Loads and integrates base_config.yaml + models/<model_name>.yaml configurations.

Usage:
    from core.config_loader import ConfigLoader
    
    config = ConfigLoader()
    
    # Load full configuration
    full_config = config.load("gpt-4o")
    
    # Individual access
    model_config = config.get_model("gpt-4o")
    defaults = config.defaults
    
    # W&B settings use environment variables:
    #   WANDB_ENTITY, WANDB_PROJECT
"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries
    
    override takes precedence, nested dicts are recursively merged
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


class ConfigLoader:
    """Configuration loader class"""
    
    def __init__(self, config_dir: Optional[str | Path] = None):
        """
        Args:
            config_dir: Path to configs directory. Auto-detected if None
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Find configs directory at project root
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "configs").exists():
                    self.config_dir = current / "configs"
                    break
                current = current.parent
            else:
                raise FileNotFoundError("configs directory not found")
        
        self._base_config: Optional[dict] = None
        self._model_configs: dict[str, dict] = {}
    
    @property
    def base_config_path(self) -> Path:
        return self.config_dir / "base_config.yaml"
    
    @property
    def models_dir(self) -> Path:
        return self.config_dir / "models"
    
    def _load_yaml(self, path: Path) -> dict:
        """Load YAML file"""
        if not path.exists():
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    @property
    def base(self) -> dict:
        """Load base_config.yaml (cached)"""
        if self._base_config is None:
            self._base_config = self._load_yaml(self.base_config_path)
        return self._base_config
    
    @property
    def wandb(self) -> dict:
        """WandB settings"""
        return self.base.get("wandb", {})
    
    @property
    def defaults(self) -> dict:
        """Default settings"""
        return self.base.get("defaults", {})
    
    @property
    def benchmarks(self) -> dict:
        """Benchmark common settings"""
        return self.base.get("benchmarks", {})
    
    @property
    def testmode(self) -> bool:
        """Test mode flag"""
        return self.base.get("testmode", False)
    
    def get_model(self, model_name: str) -> dict:
        """
        Load model configuration
        
        Args:
            model_name: Model name (e.g., "gpt-4o", "claude-3-5-sonnet")
                       Specify without file extension
        
        Returns:
            Model configuration dictionary
        """
        if model_name in self._model_configs:
            return self._model_configs[model_name]
        
        # Normalize filename: openai/gpt-4o -> gpt-4o
        # If slash exists, use only the last part
        if "/" in model_name:
            file_name = model_name.split("/")[-1]
        else:
            file_name = model_name
        
        model_path = self.models_dir / f"{file_name}.yaml"
        model_config = self._load_yaml(model_path)
        
        # Cache
        self._model_configs[model_name] = model_config
        
        return model_config
    
    def list_models(self) -> list[str]:
        """List available model configurations"""
        if not self.models_dir.exists():
            return []
        
        return [
            p.stem
            for p in self.models_dir.glob("*.yaml")
            if p.is_file()
        ]
    
    def load(self, model_name: Optional[str] = None) -> dict:
        """
        Load full configuration (base + model merged)
        
        Args:
            model_name: Model name. Returns base_config only if None
        
        Returns:
            Merged configuration dictionary
            {
                "wandb": {...},
                "defaults": {...},
                "benchmarks": {...},
                "testmode": bool,
                "model": {...}  # If model_name is specified
            }
        """
        config = self.base.copy()
        
        if model_name:
            model_config = self.get_model(model_name)
            config["model"] = model_config
            
            # Merge model config defaults with base defaults
            if "defaults" in model_config:
                config["defaults"] = _deep_merge(
                    config.get("defaults", {}),
                    model_config["defaults"]
                )
        
        return config
    
    def get_model_api_base(self, model_name: str) -> Optional[str]:
        """Return model's API base URL"""
        model_config = self.get_model(model_name)
        return model_config.get("api_base") or model_config.get("base_url")
    
    def get_model_api_key_env(self, model_name: str) -> Optional[str]:
        """Return model's API key environment variable name"""
        model_config = self.get_model(model_name)
        return model_config.get("api_key_env")
    
    def get_model_api_key(self, model_name: str) -> Optional[str]:
        """Return model's API key (read from environment variable)"""
        env_name = self.get_model_api_key_env(model_name)
        if env_name:
            return os.environ.get(env_name)
        return None
    
    def get_inspect_model_args(self, model_name: str) -> dict:
        """
        Return arguments needed for Inspect AI model creation
        
        Returns:
            {
                "model": "openai/gpt-4o",
                "model_base_url": "https://...",
                ...
            }
        """
        model_config = self.get_model(model_name)
        
        args = {}
        
        # Model ID (provider/model format)
        if "model_id" in model_config:
            args["model"] = model_config["model_id"]
        else:
            args["model"] = model_name
        
        # Base URL
        base_url = self.get_model_api_base(model_name)
        if base_url:
            args["model_base_url"] = base_url
        
        # Additional settings
        for key in ["temperature", "max_tokens", "top_p"]:
            if key in model_config:
                args[key] = model_config[key]
        
        return args


# Global instance
_config_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Return global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(model_name: Optional[str] = None) -> dict:
    """Load configuration (convenience function)"""
    return get_config().load(model_name)
