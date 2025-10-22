import sys
import yaml
from typing import Any, Dict, List, Optional
from pathlib import Path
from types import SimpleNamespace

# Add the package root (parent of this file) to sys.path for import safety
bfcl_provider_dir = Path(__file__).parent
if str(bfcl_provider_dir) not in sys.path:
    sys.path.insert(0, str(bfcl_provider_dir))

from llm_eval.datasets import load_datasets
from llm_eval.utils.logging import get_logger
from llm_eval.utils.util import EvaluationResult

# Import BFCL evaluation components
from .bfcl_eval.constants.category_mapping import (
    ALL_SCORING_CATEGORIES,
    TEST_COLLECTION_MAPPING,
)
from .bfcl_eval.utils import parse_test_category_argument

logger = get_logger(name="bfcl_evaluator", level=20)


def run_bfcl_from_configs(
    *,
    base_config_path: str,
    model_config_path: str,
) -> Dict[str, EvaluationResult]:
    """
    BFCL runner. Reads base_config.yaml and model_config.yaml,
    executes the configured sub-tasks for the BFCL benchmark,
    and returns an aggregated EvaluationResult.
    """
    # Load configuration files
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}

    # Model setup
    model_block: Dict[str, Any] = model_cfg.get("model") or {}
    model_name: Optional[str] = model_block.get("name")
    model_params: Dict[str, Any] = model_block.get("params") or {}
    if not model_name:
        raise ValueError("model_config.yaml must contain 'model.name'")

    # Extract bfcl block (case-insensitive)
    ds_key = "bfcl" if "bfcl" in base_cfg else ("BFCL" if "BFCL" in base_cfg else None)
    if not ds_key:
        logger.warning("No 'bfcl' block in base_config. Nothing to run.")
        return {
            "bfcl": EvaluationResult(
                metrics={},
                samples=[],
                info={"status": "skipped"},
            )
        }

    ds_cfg = base_cfg.get(ds_key) or {}
    subset = ds_cfg.get("subset", "all")
    split = ds_cfg.get("split", "test")
    limit = ds_cfg.get("limit", None)

    # Set dataset-specific parameters (limit is deprecated in favor of limit_per_test_category)
    dataset_params = dict(ds_cfg.get("params") or {})
    if "limit" not in dataset_params:
        dataset_params["limit"] = limit
        logger.info("Currently, the limit parameter is not used. Use limit_per_test_category instead.")

    testmode = base_cfg.get("testmode", False)
    if testmode:
        dataset_params["limit_per_test_category"] = 1
        logger.info("testmode enabled: overriding bfcl limit_per_test_category to 1 per test category")

    # Determine BFCL model name
    bfcl_model = (model_params or {}).get("model_name") or model_name

    # Determine the list of test categories
    if isinstance(subset, list):
        test_categories = subset
    elif subset in TEST_COLLECTION_MAPPING:
        test_categories = TEST_COLLECTION_MAPPING[subset]
    else:
        test_categories = [subset]

    # Prepare result and score directories (relative to this file)
    current_dir = Path(__file__).parent
    if testmode:
        result_dir = current_dir / "result_temp"
        score_dir = current_dir / "score_temp"
    else:
        result_dir = current_dir / "result"
        score_dir = current_dir / "score"
    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)


    # ---------- Model Inference ----------
    try:
        logger.info(f"[BFCL] Running model inference for {len(test_categories)} categories")
        from .bfcl_eval._llm_response_generation import main as generation_main

        # Read concurrency from config (default: 32 if not provided)
        bfcl_params: Dict[str, Any] = (ds_cfg.get("params") or {})
        num_threads_cfg: Optional[int] = bfcl_params.get("num_threads") or ds_cfg.get("num_threads")
        num_threads = int(num_threads_cfg) if num_threads_cfg is not None else 32

        args = SimpleNamespace(
            model=[bfcl_model],
            test_category=test_categories,
            temperature=0.001,  # Default value
            include_input_log=False,
            exclude_state_log=False,
            num_gpus=1,
            num_threads=num_threads,
            gpu_memory_utilization=0.9,
            backend="sglang",
            skip_server_setup=False,
            local_model_path=None,
            result_dir=str(result_dir),
            allow_overwrite=True,
            run_ids=True,  # Currently always True, so the test_case_ids_to_generate.json file is used to run the test cases
            limit_per_test_category=dataset_params.get("limit_per_test_category", None),
        )
        generation_main(args)
        logger.info(f"[BFCL] Successfully completed model inference")

    except Exception as e:
        logger.error(f"[BFCL] Failed to run model inference: {e}")
        return {
            "bfcl": EvaluationResult(
                metrics={},
                samples=[],
                info={
                    "dataset_name": "bfcl",
                    "subset": subset,
                    "split": split,
                    "model_backend_name": model_name,
                    "status": "failed",
                    "error": f"Model inference failed: {e}",
                },
            )
        }

    # ---------- Evaluation ----------
    try:
        logger.info(f"[BFCL] Running evaluation for {len(test_categories)} categories")
        from .bfcl_runner import runner as evaluation_main

        leaderboard_table = evaluation_main(
            model_names=[bfcl_model],
            test_categories=test_categories,
            result_dir=result_dir,
            score_dir=score_dir,
        )
        logger.info(f"[BFCL] Successfully completed evaluation for all categories")

    except Exception as e:
        logger.error(f"[BFCL] Failed to run evaluation: {e}")
        return {
            "bfcl": EvaluationResult(
                metrics={},
                samples=[],
                info={
                    "dataset_name": "bfcl",
                    "subset": subset,
                    "split": split,
                    "model_backend_name": model_name,
                    "status": "failed",
                    "error": f"Evaluation failed: {e}",
                },
            )
        }

    # Return aggregate result (actual scores are logged in the respective runner)
    return {
        "bfcl": EvaluationResult(
            metrics={},
            samples=[],
            info={
                "dataset_name": "bfcl",
                "subset": subset,
                "split": split,
                "model_backend_name": model_name,
                "status": "completed",
                "note": "Executed BFCL runners (results logged by BFCL).",
            },
        )
    }


def get_bfcl_test_categories() -> List[str]:
    """
    Returns all available BFCL test categories.
    TODO: Can be used as task_list in the runner.
    """
    return ALL_SCORING_CATEGORIES


def get_bfcl_category_groups() -> Dict[str, List[str]]:
    """
    Returns mapping of BFCL test category groups.
    """
    return TEST_COLLECTION_MAPPING

