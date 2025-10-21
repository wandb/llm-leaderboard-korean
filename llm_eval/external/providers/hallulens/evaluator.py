from typing import Any, Dict, List, Optional
import yaml

from llm_eval.utils.logging import get_logger
from llm_eval.utils.util import EvaluationResult
from llm_eval.datasets import load_datasets
from llm_eval.external.providers.hallulens.hallulens_runner import (
    precise_wikiqa_runner,
    longwiki_runner,
    non_mixed_entity_runner,
    non_generated_entity_runner,
)

logger = get_logger(name="hallulens_evaluator", level=20)


def run_hallulens_from_configs(
    *,
    base_config_path: str,
    model_config_path: str,
) -> Dict[str, EvaluationResult]:
    """
    halluLens 전용 실행기. base_config.yaml과 model_config.yaml을 읽어
    HalluLens 관련 서브태스크들을 실행하고, 집계용 EvaluationResult를 반환합니다.
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

    # halluLens 블록 추출 (대소문자 혼용 대응)
    ds_key = "halluLens" if "halluLens" in base_cfg else ("hallulens" if "hallulens" in base_cfg else None)
    if not ds_key:
        logger.warning("No 'halluLens' block in base_config. Nothing to run.")
        return {"halluLens": EvaluationResult(metrics={}, samples=[], info={"status": "skipped"})}

    ds_cfg = base_cfg.get(ds_key) or {}
    subset = ds_cfg.get("subset")
    split = ds_cfg.get("split", "test")
    limit = ds_cfg.get("limit", None)
    
    # testmode 체크 - testmode가 true면 각 서브셋당 10개로 제한 (기존 limit 무시)
    testmode = base_cfg.get("testmode", False)
    if testmode:
        limit = 10
        logger.info(f"testmode enabled: overriding halluLens limit to {limit} per subset")

    # dataset-specific params
    dataset_params = dict(ds_cfg.get("params") or {})
    if "limit" not in dataset_params:
        dataset_params["limit"] = limit

    # evaluator/evaluator-roles configuration from base_config
    hl_params = ds_cfg
    # Inference method은 base_config가 아니라 모델 설정의 model.name을 그대로 사용
    inference_method_global = model_name
    eval_cfg = hl_params.get("evaluation", {}) or {}
    abstention_eval_model = (eval_cfg.get("abstention", {}) or {}).get("model")
    hallucination_eval_model = (eval_cfg.get("hallucination", {}) or {}).get("model")
    # longwiki role overrides
    longwiki_roles = hl_params.get("longwiki_roles", {}) or {}
    longwiki_claim_extractor = (longwiki_roles.get("claim_extractor", {}) or {}).get("model")
    longwiki_abstain_evaluator = (longwiki_roles.get("abstain_evaluator", {}) or {}).get("model")
    longwiki_verifier = (longwiki_roles.get("verifier", {}) or {}).get("model")

    # HalluLens 아티팩트 경로 로더로 가져오기
    try:
        hallu_loader = load_datasets(name="halluLens", subset=subset, split=split, **dataset_params)
        paths_dict: Dict[str, str] = hallu_loader.load()  # { task_name: file_path }
    except Exception as e:
        logger.warning(f"Falling back to static paths due to loader error: {e}")
        paths_dict = {}

    # paths_dict = {
    #     'precise_wikiqa': '/Users/hyunwoooh/workspace/llm-leaderboard-korean/artifacts/precise_wikiqa:v0/precise_wikiqa.jsonl',
    #     'longwiki': '/Users/hyunwoooh/workspace/llm-leaderboard-korean/artifacts/longwiki:v0/longwiki.jsonl',
    #     'mixed_entities': '/Users/hyunwoooh/workspace/llm-leaderboard-korean/artifacts/non_entity_refusal:v0/mixed_entity_2000.csv',
    #     'generated_entities': '/Users/hyunwoooh/workspace/llm-leaderboard-korean/artifacts/non_entity_refusal:v0/generated_entity_1950.csv'
    # }

    # 모델 문자열 결정
    hl_model = (model_params or {}).get("model_name") or model_name

    # subset 순회 (리스트가 아니면 paths_dict 키 기준 실행)
    if isinstance(subset, list):
        task_list: List[str] = subset
    else:
        task_list = list(paths_dict.keys())
        

    for subset_name in task_list:
        task_path = paths_dict.get(subset_name)
        if not task_path:
            logger.warning(f"[HalluLens] No path for subtask '{subset_name}', skipping.")
            continue
        logger.info(f"[HalluLens] Running task '{subset_name}' with model='{hl_model}', limit='{dataset_params.get('limit')}'")
        if subset_name == "precise_wikiqa":
            precise_wikiqa_runner(
                qa_dataset_path=task_path,
                model=hl_model,
                limit=dataset_params.get("limit"),
                inference_method=inference_method_global,
                evaluator_abstention_model=abstention_eval_model,
                evaluator_halu_model=hallucination_eval_model,
            )
        elif subset_name == "longwiki":
            longwiki_runner(
                benchmark_dataset_path=task_path,
                model=hl_model,
                limit=dataset_params.get("limit"),
                inference_method=inference_method_global,
                claim_extractor=longwiki_claim_extractor or hl_model,
                abstain_evaluator=longwiki_abstain_evaluator or hl_model,
                verifier=longwiki_verifier or hl_model,
            )
        elif subset_name == "mixed_entities":
            non_mixed_entity_runner(
                prompt_path=task_path,
                tested_model=hl_model,
                limit=dataset_params.get("limit"),
                inference_method=inference_method_global,
                evaluator_model=abstention_eval_model or "gpt-4o",
            )
        elif subset_name == "generated_entities":
            non_generated_entity_runner(
                prompt_path=task_path,
                generate_model=hl_model,
                limit=dataset_params.get("limit"),
                inference_method=inference_method_global,
                evaluator_model=abstention_eval_model or "gpt-4o",
            )
        else:
            logger.warning(f"[HalluLens] Unknown subtask '{subset_name}', skipping.")

    # 집계용 결과 반환 (실제 점수 로깅은 각 러너에서 수행)
    return {
        "halluLens": EvaluationResult(
            metrics={},
            samples=[],
            info={
                "dataset_name": "halluLens",
                "subset": subset,
                "split": split,
                "model_backend_name": model_name,
                "status": "completed",
                "note": "Executed HalluLens runners (results logged by HalluLens).",
            },
        )
    }
