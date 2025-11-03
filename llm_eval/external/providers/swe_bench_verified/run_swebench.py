#!/usr/bin/env python3
import argparse
from dotenv import load_dotenv, find_dotenv
import os
import sys
import json
import wandb
from pathlib import Path

# SSL 인증서 경로 설정 (certifi 사용)
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass

# Ensure local imports resolve when run directly
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))
if str(CUR_DIR / 'evaluator') not in sys.path:
    sys.path.insert(0, str(CUR_DIR / 'evaluator'))

from config_singleton import WandbConfigSingleton
from evaluator.swe_bench import evaluate as swe_evaluate
from llm_inference_adapter import get_llm_inference_engine


def main():
    # .env 자동 로드 (프로젝트 루트/현재 디렉토리 탐색)
    try:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
        else:
            # 상위 디렉토리도 탐색
            load_dotenv(override=False)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation (vendor, standalone)")
    parser.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "llm-leaderboard"))
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--run-name", type=str, default=os.environ.get("WANDB_RUN_NAME", "swebench-run"))
    parser.add_argument("--config", type=str, help="Path to a YAML/JSON config file to load into wandb.config", required=False)
    args = parser.parse_args()

    cfg_obj = None
    cfg_path: Path | None = None
    if args.config:
        cfg_path = Path(args.config)
    else:
        # Fallback to base_config.yaml in project root
        project_root = CUR_DIR.parent.parent.parent.parent
        candidates = [
            Path(os.environ.get("SWE_CONFIG")) if os.environ.get("SWE_CONFIG") else None,
            project_root / "configs" / "base_config.yaml",
            Path.cwd() / "configs" / "base_config.yaml",
        ]
        for cand in candidates:
            if cand and cand.exists():
                cfg_path = cand
                break

    if cfg_path:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        text = cfg_path.read_text(encoding="utf-8")
        if cfg_path.suffix.lower() in [".yaml", ".yml"]:
            try:
                import yaml
                cfg_obj = yaml.safe_load(text)
            except Exception as e:
                raise RuntimeError(f"Failed to parse YAML config: {e}")
        else:
            try:
                cfg_obj = json.loads(text)
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON config: {e}")

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name, config=cfg_obj)

    # Initialize singleton first (llm will be attached after creation)
    WandbConfigSingleton.initialize(run=run, llm=None)
    # Build LLM from current config
    llm = get_llm_inference_engine()
    # Attach llm to singleton
    inst = WandbConfigSingleton.get_instance()
    inst.llm = llm

    # Run evaluation (may return callback when background_eval is true)
    result_or_callback = swe_evaluate()
    if callable(result_or_callback):
        # If background mode, wait and aggregate
        result_or_callback()


if __name__ == "__main__":
    main()


