from typing import Any, Dict, Optional, List
from pathlib import Path
import tempfile
import yaml

from datasets import load_from_disk
from llm_eval.models import load_model

from llm_eval.utils.util import EvaluationResult
from llm_eval.utils.logging import get_logger
from llm_eval.wandb_singleton import WandbConfigSingleton

from .swe_bench import (
    generate_predictions,
    run_swebench_evaluation,
    calculate_metrics,
)


logger = get_logger(name="swebench_verified_evaluator", level=20)


def _infer_max_tokens(model_cfg: Dict[str, Any], default_val: int = 32768) -> int:
    try:
        if "max_tokens" in model_cfg['model']['params']:
            return int(model_cfg['model']['params'].get("max_tokens", default_val))
        elif "max_completion_tokens" in model_cfg['model']['params']:
            return int(model_cfg['model']['params'].get("max_completion_tokens", default_val))
        else:
            return default_val
        # return int(((model_cfg.get("model") or {}).get("params") or {}).get("max_tokens", default_val))
    except Exception:
        return default_val


def run_swebench_verified_from_configs(*, base_config_path: str, model_config_path: str) -> Dict[str, EvaluationResult]:
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}

    # 모델 설정 유효성 확인
    model_block: Dict[str, Any] = model_cfg.get("model") or {}
    model_name: Optional[str] = model_block.get("name")
    if not model_name:
        raise ValueError("model_config.yaml must contain 'model.name'")

    # swe_bench_verified 블록 확인
    ds_key = "swe_bench_verified"
    sb_cfg: Dict[str, Any] = base_cfg.get(ds_key) or {}
    split: str = sb_cfg.get("split", "test")
    subset: str = sb_cfg.get("subset", "official_80")
    limit: Optional[int] = sb_cfg.get("limit")

    artifact_dir = WandbConfigSingleton.download_artifact("swebench_verified_official_80")

    hf_ds = load_from_disk(artifact_dir)
    if hasattr(hf_ds, "keys") and split in hf_ds:
        hf_ds = hf_ds[split]

    # 샘플 목록 작성 및 제한 적용
    samples: List[Dict[str, Any]] = [row for row in hf_ds]
    if limit is not None:
        samples = samples[: int(limit)]

    # 실행 준비: 모델 백엔드 인스턴스화 및 싱글톤 보강
    instance = WandbConfigSingleton.get_instance()
    model_params: Dict[str, Any] = (model_block.get("params") or {}).copy()
    # OpenAI 백엔드 등은 __init__ 시 필요한 키를 직접 인자로 받음
    llm = load_model(
        model_name,  # e.g., "openai"
        **model_params,
    )
    # 싱글톤에 llm 주입 (기존에 문자열이 들어있던 문제 수정)
    try:
        instance.llm = llm
    except Exception:
        pass

    # swe_bench.py 내부에서 참조하는 cfg.swebench 키 구성
    sb_server = sb_cfg.get("server", {}) or {}
    swebench_cfg: Dict[str, Any] = {
        "fc_enabled": False,
        "prebuild_images": False,
        "private_registry": None,
        "max_samples": int(sb_cfg.get("limit") or 500),
        "max_workers": int(sb_cfg.get("max_workers", 4)),
        "background_eval": False,
        "images": {"namespace": "swebench", "tag": "latest"},
        "api_server": {
            "enabled": bool(sb_server.get("url")),
            "endpoint": (sb_server.get("url") or "").strip(),
            "api_key": (sb_server.get("token") or "").strip(),
            # optional tuning
            "concurrency": None,
            "timeout_sec": 1200,
            "namespace": "swebench",
            "tag": "latest",
        },
    }
    for key in ["max_tokens", "max_completion_tokens"]:
        if key in model_params:
            swebench_cfg[key] = int(model_params[key])
    try:
        # OmegaConf or dict both supported
        instance.config.swebench = swebench_cfg
    except Exception:
        try:
            instance.config["swebench"] = swebench_cfg
        except Exception:
            pass

    # model.pretrained_model_name_or_path 주입 (calculate_metrics에서 사용)
    try:
        instance.config.model = {
            "pretrained_model_name_or_path": model_params.get("model_name") or model_name
        }
    except Exception:
        try:
            instance.config["model"] = {
                "pretrained_model_name_or_path": model_params.get("model_name") or model_name
            }
        except Exception:
            pass

    max_tokens = _infer_max_tokens(model_cfg)
    generator_config = model_params.copy()
    generator_config.pop('model_name', None)
    generator_config.pop('api_base', None)
    generator_config.pop('batch_size', None)
    

    # 임시 디렉터리 및 파일 경로
    temp_dir = Path(tempfile.mkdtemp(prefix="swebench_eval_"))
    predictions_file = temp_dir / "predictions.jsonl"

    # 패치 생성 → 평가 실행 → 메트릭 집계/로그
    generate_predictions(samples, llm, generator_config, predictions_file, model_block.get("params", {}).get("model_name") or model_name)

    instance_ids = [s.get("instance_id") for s in samples]
    results = run_swebench_evaluation(predictions_file, max_workers=sb_cfg.get("max_workers", 4), instance_ids=instance_ids, samples=samples)
    calculate_metrics(samples, results, temp_dir)

    # 리더보드용 껍데기 결과 반환 (세부 로깅은 내부에서 처리)
    return {
        ds_key: EvaluationResult(
            metrics={},
            samples=[],
            info={
                "dataset_name": ds_key,
                "subset": subset,
                "split": split,
                "model_backend_name": model_name,
                "status": "completed",
            },
        )
    }


