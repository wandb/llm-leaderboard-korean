from typing import Any, Dict, List, Optional
import wandb
import weave

def run_all_from_configs(
    base_config_path: str,
    model_config_path: str,
    selected_datasets: Optional[List[str]] = None,
    *,
    language_penalize: Optional[bool] = None,
    target_lang: Optional[str] = None,
):
    """
    Convenience wrapper that creates a shared W&B run (singleton) and then
    calls Evaluator.run_multiple_from_configs so that all leaderboard tables
    are logged under the same run.
    """
    import yaml
    from llm_eval.evaluator import run_multiple_from_configs
    from llm_eval.wandb_singleton import WandbConfigSingleton

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}
    model_name: Optional[str] = model_cfg.get("model").get("params").get("model_name")

    # Load base to extract wandb settings
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    wandb_params: Dict[str, Any] = ((base_cfg.get("wandb") or {}).get("params") or {})
    weave.init(f"{wandb_params.get('entity')}/{wandb_params.get('project')}")
    run = wandb.init(entity=wandb_params.get("entity"), project=wandb_params.get("project"), name=model_name)
    WandbConfigSingleton.initialize(run, model_name, wandb_params)

    result = run_multiple_from_configs(
        base_config_path=base_config_path,
        model_config_path=model_config_path,
        selected_datasets=selected_datasets,
        language_penalize=language_penalize,
        target_lang=target_lang,
    )
    run.finish()
    return result


if __name__ == "__main__":
    # make parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default="configs/gpt-4o-2024-11-20.yaml")
    # parser.add_argument("--model_config_path", type=str, default="configs/gpt-4-1-2025-04-14.yaml")
    # parser.add_argument("--model_config_path", type=str, default="configs/gpt-5-mini-2025-08-07.yaml")
    parser.add_argument("--base_config_path", type=str, default="configs/base_config.yaml")
    parser.add_argument("--dataset", type=str, default="haerae_bench_v1")
    args = parser.parse_args()

    result = run_all_from_configs(
        base_config_path=args.base_config_path,
        model_config_path=args.model_config_path,
        selected_datasets=[
            args.dataset
            # Done: "halluLens", "ifeval_ko", "komoral", "korean_hate_speech", "korean_parallel_corpora", "mrcr", "haerae_bench_v1", "squad_kor_v1", "kobbq", "kmmlu", "kmmlu_pro", "kobalt_700", "hle", "arc_agi"
            # Need to check: "aime2025", "hrm8k", 
            # Need to add: "bfcl", "swe_bench_verified", "ko_dark_bench", "MT bench"
            # Not use: arc_agi2, kmmlu_hard
        ],
    )
    print(result)