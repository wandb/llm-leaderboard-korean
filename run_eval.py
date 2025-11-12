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
    use_standard_weave: bool = False,
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
    release_date: Optional[str] = model_cfg.get("model").get("release_date")
    size_category: Optional[str] = model_cfg.get("model").get("size_category")
    model_size: Optional[str] = model_cfg.get("model").get("model_size")

    # Load base to extract wandb settings
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    wandb_params: Dict[str, Any] = ((base_cfg.get("wandb") or {}).get("params") or {})
    weave.init(f"{wandb_params.get('entity')}/{wandb_params.get('project')}")
    run = wandb.init(
        entity=wandb_params.get("entity"),
        project=wandb_params.get("project"),
        name=model_name,
        config={"model_config": model_cfg, "base_config": base_cfg},
        )
    WandbConfigSingleton.initialize(run, model_name, wandb_params)
    if model_cfg.get("model").get("params").get("provider") == "hosted_vllm":
        from llm_eval.models.vllm_server_manager import start_vllm_server
        start_vllm_server(model_cfg.get("model").get("vllm_params"))

    result = run_multiple_from_configs(
        base_config_path=base_config_path,
        model_config_path=model_config_path,
        selected_datasets=selected_datasets,
        language_penalize=language_penalize,
        target_lang=target_lang,
        use_standard_weave=use_standard_weave,
    )

    WandbConfigSingleton.log_overall_leaderboard_table(model_name, release_date, size_category, model_size, selected_datasets)
    if model_cfg.get("model").get("params").get("provider") == "hosted_vllm":
        from llm_eval.models.vllm_server_manager import shutdown_vllm_server
        shutdown_vllm_server()
    run.finish()
    WandbConfigSingleton.finish()

    # Clean up any pending asyncio tasks from weave
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
    except RuntimeError:
        # No running event loop
        pass

    # Force garbage collection to clean up resources
    import gc
    gc.collect()

    return result


if __name__ == "__main__":
    import argparse
    import warnings
    import sys

    # Suppress ResourceWarning for unclosed aiohttp sessions from weave library
    warnings.filterwarnings("ignore", category=ResourceWarning, message="Unclosed.*")

    # Alternative: Set environment variable to suppress asyncio debug mode
    import os
    os.environ["PYTHONWARNINGS"] = "ignore::ResourceWarning"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--base_config_path", type=str, default="base_config.yaml")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--use-standard-weave", default=True, action="store_true",
                        help="Use Standard Weave Evaluation Framework with scorers")
    args = parser.parse_args()

    if args.dataset:
        selected_datasets = [args.dataset]
    else:
        selected_datasets = ["mt_bench", "halluLens", "ifeval_ko", "komoral", "korean_hate_speech", "mrcr_2_needles", "haerae_bench_v1_w_RC", "haerae_bench_v1_wo_RC", "squad_kor_v1", "kobbq", "kmmlu", "kmmlu_pro", "kobalt_700_syntax", "kobalt_700_semantic", "hle", "arc_agi", "aime2025", "hrm8k", "bfcl", "swebench", "korean_parallel_corpora"]
        # selected_datasets = ["mt_bench", "halluLens", "hle", "aime2025", "hrm8k", "bfcl"]#, "swe_bench_verified"]
        # selected_datasets = ["hle", "aime2025", "hrm8k", "komoral", "kmmlu", "halluLens", "bfcl", "swebench", "mt_bench"]#, "swe_bench_verified"]
        # selected_datasets = ["halluLens", "bfcl", "swebench", "mt_bench"]#, "swe_bench_verified"]
        # selected_datasets = ["mt_bench", "haerae_bench_v1"]#, "swe_bench_verified"]
        selected_datasets = ["mt_bench", "halluLens", "ifeval_ko", "komoral", "korean_hate_speech", "mrcr_2_needles", "haerae_bench_v1_w_RC", "haerae_bench_v1_wo_RC", "squad_kor_v1", "kobbq", "kmmlu", "kmmlu_pro", "kobalt_700_syntax", "kobalt_700_semantic", "hle", "arc_agi", "aime2025", "hrm8k", "bfcl", "swebench", "korean_parallel_corpora"]

    if args.config:
        configs = [args.config]
    else:
        configs = ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", "o4-mini-2025-04-16", "gpt-4o-2024-11-20", "gpt-4.1-2025-04-14"]
    for config in configs:
        print(f"Running {config}...")
        result = run_all_from_configs(
            base_config_path=f"configs/{args.base_config_path}",
            model_config_path=f"configs/{config}.yaml",
            selected_datasets=selected_datasets,
            use_standard_weave=args.use_standard_weave,
        )