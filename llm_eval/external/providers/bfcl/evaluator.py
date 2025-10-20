from typing import Any, Dict, List, Optional
import yaml
import tempfile
import shutil
from pathlib import Path

from llm_eval.utils.logging import get_logger
from llm_eval.utils.util import EvaluationResult
from llm_eval.datasets import load_datasets

# Import BFCL evaluation components
from .eval_runner import evaluate_task, get_handler
from .bfcl_eval.constants.category_mapping import ALL_SCORING_CATEGORIES, TEST_COLLECTION_MAPPING
from .bfcl_eval.utils import parse_test_category_argument

logger = get_logger(name="bfcl_evaluator", level=20)

def run_bfcl_from_configs(
    *,
    base_config_path: str,
    model_config_path: str,
) -> Dict[str, EvaluationResult]:
    """
    BFCL 전용 실행기. base_config.yaml과 model_config.yaml을 읽어
    BFCL 벤치마크의 설정된 서브태스크들을 실행하고, 집계용 EvaluationResult를 반환합니다.
    """
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}

    # 모델 설정
    model_block: Dict[str, Any] = model_cfg.get("model") or {}
    model_name: Optional[str] = model_block.get("name")
    model_params: Dict[str, Any] = model_block.get("params") or {}
    if not model_name:
        raise ValueError("model_config.yaml must contain 'model.name'")

    # bfcl 블록 추출 (대소문자 혼용 대응)
    ds_key = "bfcl" if "bfcl" in base_cfg else ("BFCL" if "BFCL" in base_cfg else None)
    if not ds_key:
        logger.warning("No 'bfcl' block in base_config. Nothing to run.")
        return {"bfcl": EvaluationResult(metrics={}, samples=[], info={"status": "skipped"})}

    ds_cfg = base_cfg.get(ds_key) or {}
    subset = ds_cfg.get("subset", "all")
    split = ds_cfg.get("split", "test")
    limit = ds_cfg.get("limit", None)

    # dataset-specific params
    dataset_params = dict(ds_cfg.get("params") or {})
    if "limit" not in dataset_params:
        dataset_params["limit"] = limit

    # BFCL 모델명 설정
    bfcl_model = (model_params or {}).get("model_name") or model_name

    # 테스트 카테고리 결정
    if isinstance(subset, list):
        test_categories = subset
    elif subset in TEST_COLLECTION_MAPPING:
        test_categories = TEST_COLLECTION_MAPPING[subset]
    else:
        # 개별 카테고리명인 경우
        test_categories = [subset]

    # 실제 BFCL 평가 실행을 위한 임시 디렉토리 설정
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        result_dir = temp_path / "result" / bfcl_model.replace("/", "_")
        score_dir = temp_path / "score"
        
        # 디렉토리 생성
        result_dir.mkdir(parents=True, exist_ok=True)
        score_dir.mkdir(parents=True, exist_ok=True)

        # 리더보드 테이블 초기화
        leaderboard_table = {}
        
        # 모델 핸들러 생성
        try:
            handler = get_handler(bfcl_model)
        except Exception as e:
            logger.error(f"Failed to create handler for model {bfcl_model}: {e}")
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
                        "error": f"Handler creation failed: {e}",
                    },
                )
            }

        # 각 테스트 카테고리에 대해 평가 실행
        completed_tasks = []
        failed_tasks = []
        
        for test_category in test_categories:
            try:
                logger.info(f"[BFCL] Running evaluation for category: {test_category}")
                
                # 모델 결과 로드 (실제 구현에서는 모델 추론 결과가 있어야 함)
                # 현재는 placeholder로 빈 결과 사용
                model_result = []  # TODO: 실제 모델 추론 결과 로드
                
                if not model_result:
                    logger.warning(f"[BFCL] No model results found for {test_category}, skipping")
                    continue
                
                # BFCL 평가 실행
                leaderboard_table = evaluate_task(
                    test_category=test_category,
                    result_dir=result_dir,
                    score_dir=score_dir,
                    model_result=model_result,
                    model_name=bfcl_model.replace("/", "_"),
                    handler=handler,
                    leaderboard_table=leaderboard_table,
                )
                
                completed_tasks.append(test_category)
                logger.info(f"[BFCL] Successfully completed evaluation for {test_category}")
                
            except Exception as e:
                logger.error(f"[BFCL] Failed to evaluate {test_category}: {e}")
                failed_tasks.append({"category": test_category, "error": str(e)})

        # 결과 집계
        total_tasks = len(test_categories)
        success_count = len(completed_tasks)
        
        metrics = {
            "total_categories": total_tasks,
            "completed_categories": success_count,
            "success_rate": success_count / total_tasks if total_tasks > 0 else 0,
        }
        
        # 리더보드 테이블에서 정확도 정보 추출
        if leaderboard_table and bfcl_model.replace("/", "_") in leaderboard_table:
            model_results = leaderboard_table[bfcl_model.replace("/", "_")]
            for category, result in model_results.items():
                if isinstance(result, dict) and "accuracy" in result:
                    metrics[f"{category}_accuracy"] = result["accuracy"]
                    metrics[f"{category}_total_count"] = result.get("total_count", 0)

        info = {
            "dataset_name": "bfcl",
            "subset": subset,
            "split": split,
            "model_backend_name": model_name,
            "bfcl_model": bfcl_model,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "status": "completed" if success_count > 0 else "failed",
        }

        return {
            "bfcl": EvaluationResult(
                metrics=metrics,
                samples=[],  # TODO: 실제 샘플 결과 추가
                info=info,
            )
        }


def get_bfcl_test_categories() -> List[str]:
    """
    BFCL에서 사용 가능한 모든 테스트 카테고리 목록을 반환합니다.
    TODO runner에서 task_list로 사용할 수 있습니다.
    """
    return ALL_SCORING_CATEGORIES


def get_bfcl_category_groups() -> Dict[str, List[str]]:
    """
    BFCL 테스트 카테고리 그룹을 반환합니다.
    """
    return TEST_COLLECTION_MAPPING

