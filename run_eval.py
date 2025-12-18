#!/usr/bin/env python3

import argparse
import locale
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent / ".env")

# Set English locale to fix inspect_evals date parsing issue
try:
    locale.setlocale(locale.LC_TIME, "en_US.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, "C")
    except locale.Error:
        pass  # Continue even if locale setting fails

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import wandb
from core.config_loader import get_config

# All benchmarks list (active ones only)
ALL_BENCHMARKS = [
    "ko_hellaswag",
    "ko_aime2025",
    "ifeval_ko",
    "ko_balt_700",
    # "ko_balt_700_syntax",
    # "ko_balt_700_semantic",
    "haerae_bench_v1",
    # "haerae_bench_v1_rc",
    # "haerae_bench_v1_wo_rc",
    "kmmlu",
    "kmmlu_pro",
    "squad_kor_v1",
    "ko_truthful_qa",
    "ko_moral",
    "ko_arc_agi",
    "ko_gsm8k",
    "korean_hate_speech",
    "kobbq",
    "ko_hle",
    "ko_hallulens_wikiqa",
    # "ko_hallulens_longwiki",
    "ko_hallulens_nonexistent",
    "bfcl",
    "ko_mtbench",
    "swebench_verified_official_80",
]

# Quick test benchmarks (lightweight ones only)
QUICK_BENCHMARKS = [
    "ko_hellaswag",
    "kmmlu",
    "kobbq",
    "korean_hate_speech",
    "ifeval_ko",
    "ko_moral",
]


def get_model_env(config_name: str) -> dict[str, str]:
    """
    Generate API environment variables from model config file
    
    From configs/models/<config_name>.yaml:
    - base_url → OPENAI_BASE_URL (or provider-specific environment variable)
    - api_key_env → Read API key from that environment variable
    
    Returns:
        Environment variable dictionary
    """
    config = get_config()
    model_config = config.get_model(config_name)
    
    if not model_config:
        return {}
    
    env = {}
    
    # Check provider (based on model_id: openai/solar-pro2 → openai)
    model_id = model_config.get("model_id") or config_name
    provider = model_id.split("/")[0] if "/" in model_id else "openai"
    provider_upper = provider.upper()
    
    # Set base URL
    base_url = model_config.get("base_url") or model_config.get("api_base")
    if base_url:
        # OpenAI-compatible APIs use OPENAI_BASE_URL
        if provider in ["openai", "together", "groq", "fireworks"]:
            env["OPENAI_BASE_URL"] = base_url
        else:
            env[f"{provider_upper}_BASE_URL"] = base_url
    
    # Set API key
    api_key_env = model_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            # OpenAI-compatible APIs
            if provider in ["openai", "together", "groq", "fireworks"]:
                env["OPENAI_API_KEY"] = api_key
            else:
                env[f"{provider_upper}_API_KEY"] = api_key
    
    return env


def get_model_metadata(model: str) -> dict:
    """
    Load metadata from model config file
    
    Returns:
        {
            "release_date": "2024-05-13",
            "size_category": "flagship",
            "model_size": "unknown",
            ...
        }
    """
    config = get_config()
    model_config = config.get_model(model)
    
    if not model_config:
        return {}
    
    metadata = model_config.get("metadata", {})
    return {
        "release_date": metadata.get("release_date", "unknown"),
        "size_category": metadata.get("size_category", "unknown"),
        "model_size": metadata.get("model_size") or metadata.get("parameters", "unknown"),
    }


def run_benchmark(
    benchmark: str, 
    config_name: str,
    limit: int | None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> tuple[str, bool, str, dict | None]:
    """
    Run a single benchmark
    
    Automatically applies API settings from model config file (configs/models/<model>.yaml).
    
    Returns:
        (benchmark_name, success, error_message, scores)
    """
    cmd = ["uv", "run", "horangi", benchmark, "--config", config_name]
    
    # Add limit only if specified (null = all)
    if limit is not None:
        cmd.extend(["-T", f"limit={limit}"])
    
    # Load environment variables from model config
    model_env = get_model_env(config_name)
    
    # Merge with current environment (model config takes precedence)
    env = os.environ.copy()
    env.update(model_env)
    
    # Set English locale to fix inspect_evals date parsing issue
    env["LC_TIME"] = "en_US.UTF-8"

    # Force W&B/Weave project for each benchmark subprocess (inspect eval)
    # (Without this, it may log to wandb's default project like horangi-dev)
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
    
    print(f"\n{'='*60}")
    print(f"🏃 Running: {benchmark}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr to stdout
            text=True,
            bufsize=1,  # Line buffering
            env=env,
        )
        
        # Collect output while printing in real-time
        output_lines = []
        weave_eval_url: str | None = None
        hook_noise_patterns = (
            r"^inspect_ai v",
            r"^- hooks enabled:",
            r"^\s*inspect_wandb/weave_evaluation_hooks:",
            r"^\s*inspect_wandb/wandb_models_hooks:",
        )
        for line in process.stdout:
            # Capture Weave Eval URL (show only once after benchmark completes)
            m = re.search(r"🔗\s*Weave Eval:\s*(https?://\S+)", line)
            if m:
                weave_eval_url = m.group(1)
            
            # Filter unnecessary noise logs/intermediate URL lines
            suppress = False
            if m:
                suppress = True
            else:
                for pat in hook_noise_patterns:
                    if re.search(pat, line):
                        suppress = True
                        break
            
            if not suppress:
                print(line, end="", flush=True)  # Real-time output
            output_lines.append(line)
        
        process.wait(timeout=1800)  # 30 minute timeout
        full_output = "".join(output_lines)
        
        success = process.returncode == 0
        
        # Print Weave Eval URL after benchmark completes
        if weave_eval_url:
            print(f"\n🔗 Weave Eval: {weave_eval_url}")
        
        # Try to parse scores
        scores = None
        if success:
            scores = parse_scores_from_output(full_output, benchmark)
        
        return benchmark, success, "" if success else f"Exit code: {process.returncode}", scores
    
    except subprocess.TimeoutExpired:
        process.kill()
        return benchmark, False, "Timeout (30m)", None
    except Exception as e:
        if 'process' in locals():
            process.kill()
        return benchmark, False, str(e), None


def parse_scores_from_output(output: str, benchmark: str) -> dict | None:
    """
    Parse scores from Inspect AI output
    
    Inspect AI output format example:
        accuracy  0.600
        stderr    0.245
        
        or
        
        mean    0.640
        writing_score  0.640
    
    Returns:
        {"score": main_score, "details": {metric_name: value, ...}}
    """
    all_metrics = {}
    
    # Parse all "name  number" patterns (line start, underscore/alphanumeric names)
    # Exclude stderr
    pattern = r"^([a-zA-Z][a-zA-Z0-9_]*)\s+([\d.-]+)\s*$"
    for match in re.finditer(pattern, output, re.MULTILINE):
        metric_name = match.group(1)
        # Exclude meta info like stderr, samples, tokens
        if metric_name.lower() in ["stderr", "samples", "tokens", "total"]:
            continue
        try:
            all_metrics[metric_name] = float(match.group(2))
        except ValueError:
            pass
    
    if not all_metrics:
        return None
    
    # Select main score
    main_score = None
    
    # IFEval: use final_acc or prompt_strict_acc
    if benchmark == "ifeval_ko":
        main_score = all_metrics.get("final_acc") or all_metrics.get("prompt_strict_acc")
    
    # KoBBQ: use kobbq_avg
    elif benchmark == "kobbq":
        main_score = all_metrics.get("kobbq_avg")
    
    # HLE: use hle_accuracy
    elif benchmark == "ko_hle":
        main_score = all_metrics.get("hle_accuracy") or all_metrics.get("accuracy")
    
    # HalluLens: use correct_rate or refusal_rate
    elif "hallulens" in benchmark:
        main_score = all_metrics.get("correct_rate") or all_metrics.get("refusal_rate")
    
    # MT-Bench: use mean (10-point scale → 0-1 scale)
    elif benchmark == "ko_mtbench":
        if "mean" in all_metrics:
            main_score = all_metrics["mean"] / 10.0
    
    # BFCL: use accuracy
    elif benchmark == "bfcl":
        main_score = all_metrics.get("accuracy")
    
    # SQuAD: f1 > exact priority
    elif benchmark == "squad_kor_v1":
        main_score = all_metrics.get("mean")  # f1.mean
    
    # General metric priority
    if main_score is None:
        for metric in ["accuracy", "mean", "macro_f1", "f1", "resolved"]:
            if metric in all_metrics:
                main_score = all_metrics[metric]
                if metric == "mean" and benchmark == "ko_mtbench":
                    main_score = main_score / 10.0  # mtbench scale
                break
    
    return {
        "score": main_score,
        "details": all_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks and create leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default run (entity/project loaded from configs/base_config.yaml)
    uv run python run_eval.py --config gpt-4o

    # Quick test (lightweight benchmarks only)
    uv run python run_eval.py --config gpt-4o --quick
    
    # Run specific benchmarks only
    uv run python run_eval.py --config gpt-4o --only ko_hellaswag,kmmlu
"""
    )
    
    # Basic options
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config name (configs/models/<name>.yaml, e.g., gpt-4o, solar_pro2)",
    )
    parser.add_argument("--limit", type=int,
                        help="Number of samples per benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Run only quick/light benchmarks")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated list of benchmarks to run (exclusive)")
    
    args = parser.parse_args()
    
    # Load W&B settings from environment variables (.env file already loaded)
    entity = os.environ.get("WANDB_ENTITY", "")
    project = os.environ.get("WANDB_PROJECT", "")
    
    if not entity or not project:
        print("❌ WANDB_ENTITY and WANDB_PROJECT environment variables are required for W&B logging.")
        print("   Add the following to your .env file:")
        print("     WANDB_ENTITY=your-entity")
        print("     WANDB_PROJECT=your-project")
        sys.exit(1)
    
    config = get_config()
    
    # Filter benchmarks
    if args.quick:
        benchmarks = QUICK_BENCHMARKS
    elif args.only:
        benchmarks = [b.strip() for b in args.only.split(",") if b.strip()]
        # Validate
        invalid = [b for b in benchmarks if b not in ALL_BENCHMARKS]
        if invalid:
            print(f"❌ Unknown benchmarks: {invalid}")
            print(f"   Available: {ALL_BENCHMARKS}")
            sys.exit(1)
    else:
        benchmarks = ALL_BENCHMARKS
    
    # Load model config (configs/models/<name>.yaml)
    model_cfg = config.get_model(args.config)
    if not model_cfg:
        print(f"❌ Model configuration not found: {args.config}")
        print("   Check if YAML file exists in configs/models/ directory.")
        sys.exit(1)

    model_id = model_cfg.get("model_id") or args.config

    # Display model name (openai/solar-pro2 → solar-pro2)
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    
    wandb_run = wandb.init(
        entity=entity,
        project=project,
        name=f"eval-{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        job_type="evaluation",
        config={
            "config": args.config,
            "model": model_id,
            "model_name": model_name,
            "limit": args.limit,
            "benchmarks": benchmarks,
        },
    )
    print(f"✅ W&B run started: {wandb_run.url}")
    
    
    print(f"\n🐯 Horangi Benchmark Runner")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Model: {model_id}")
    print(f"Limit: {args.limit} samples per benchmark")
    print(f"Benchmarks: {len(benchmarks)} / {len(ALL_BENCHMARKS)}")
    print(f"Leaderboard: {entity}/{project}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Track execution results
    results = []
    benchmark_scores = {}
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] ", end="")
        name, success, error, scores = run_benchmark(
            benchmark, 
            args.config,
            args.limit,
            wandb_entity=entity,
            wandb_project=project,
        )
        results.append((name, success, error))
        
        if scores:
            benchmark_scores[name] = scores
    
    # Results summary
    print(f"\n\n{'='*60}")
    print(f"📊 Results Summary")
    print(f"{'='*60}")
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"\n✅ Successful: {len(successful)} / {len(results)}")
    for name, _, _ in successful:
        score_info = benchmark_scores.get(name, {})
        score_str = f" (score: {score_info.get('score', 'N/A')})" if score_info else ""
        print(f"   - {name}{score_str}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} / {len(results)}")
        for name, _, error in failed:
            print(f"   - {name}: {error}")
    
    # Print detailed results table by category
    print(f"\n{'='*60}")
    print(f"📋 Detailed Results by Category")
    print(f"{'='*60}")
    
    for benchmark_name, score_info in benchmark_scores.items():
        details = score_info.get("details", {})
        if len(details) > 1:  # Only if there are detailed results
            print(f"\n📌 {benchmark_name}")
            print(f"   {'─'*40}")
            
            # Separate main metrics and category metrics
            main_metrics = []
            category_metrics = []
            
            for metric, value in sorted(details.items()):
                if "_score" in metric or "_accuracy" in metric or "_rate" in metric or "_acc" in metric:
                    category_metrics.append((metric, value))
                else:
                    main_metrics.append((metric, value))
            
            # Print main metrics
            for metric, value in main_metrics:
                print(f"   {metric:<30} {value:.4f}")
            
            # Print category metrics (table format)
            if category_metrics:
                print(f"   {'─'*40}")
                for metric, value in category_metrics:
                    print(f"   {metric:<30} {value:.4f}")
    
    # Create Weave Leaderboard (if there are successful benchmarks)
    if benchmark_scores and entity and project:
        try:
            from core.weave_leaderboard import create_weave_leaderboard
            # Pass only successful benchmark list
            successful_benchmarks = list(benchmark_scores.keys())
            leaderboard_url = create_weave_leaderboard(
                entity=entity,
                project=project,
                benchmarks=successful_benchmarks,
            )
            if leaderboard_url:
                print(f"\n🏆 Leaderboard URL: {leaderboard_url}")
        except Exception as e:
            print(f"⚠️ Weave Leaderboard creation failed: {e}")
    
    # End W&B run
    if wandb_run is not None:
        print(f"\n📊 Ending W&B run...")
        wandb_run.finish()
        print(f"✅ W&B run completed!")
    
    print(f"\n{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Exit code 0 even if some failures, as long as leaderboard was created
    # (partial benchmark failures still have results saved)
    sys.exit(0)


if __name__ == "__main__":
    main()
