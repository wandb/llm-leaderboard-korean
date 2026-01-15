"""
Weave Leaderboard Auto-generation Module

Automatically generates Weave Leaderboard from Inspect AI evaluation results.
This leaderboard enables model performance comparison in the Weave UI.

Usage:
    # Automatically called from run_eval.py
    from core.weave_leaderboard import create_weave_leaderboard
    
    create_weave_leaderboard(
        entity="wandb-korea",
        project="korean-llm-eval",
        model_name="gpt-4o",
    )

Note:
    - Uses Weave UI's Leaderboard feature.
"""

from __future__ import annotations

import weave
from weave.flow import leaderboard
from weave.trace import urls as weave_urls


# Leaderboard configuration
LEADERBOARD_REF = "Korean-LLM-Leaderboard"
LEADERBOARD_NAME = "Korean LLM Leaderboard"
LEADERBOARD_DESCRIPTION = """Korean LLM Benchmark Model Performance Comparison Leaderboard

This leaderboard is automatically generated from Inspect AI evaluation results.
Models are compared on two axes: General Language Performance (GLP) and Alignment Performance (ALT).

📊 General Language Performance (GLP):
- Syntax/Semantics: ko_balt_700, haerae_bench_v1
- General/Expert Knowledge: kmmlu, kmmlu_pro, ko_hle
- Common Sense/Math/Abstract Reasoning: ko_hellaswag, hrm8k, ko_aime2025, ko_arc_agi
- Information Retrieval: squad_kor_v1
- Expression: ko_mtbench
- Coding: swebench_verified_official_80
- Function Calling: bfcl

🛡️ Alignment Performance (ALT):
- Controllability: ifeval_ko
- Ethics/Morality: ko_moral
- Toxicity/Bias Prevention: korean_hate_speech, kobbq
- Hallucination Prevention: ko_truthful_qa, ko_hallulens_wikiqa, ko_hallulens_nonexistent
"""


def get_evaluation_ref(entity: str, project: str, benchmark: str) -> str | None:
    """
    Get the actual ref of the evaluation object for a benchmark.
    
    :latest tag doesn't work in Leaderboard,
    so returns ref with actual digest.
    """
    from weave.trace.ref_util import get_ref
    
    try:
        eval_name = f"{benchmark}-evaluation"
        eval_obj = weave.ref(f"{eval_name}:latest").get()
        ref = get_ref(eval_obj)
        if ref:
            return ref.uri()
    except Exception:
        pass
    
    return None


def build_columns_from_benchmarks(
    benchmarks: list[str],
    entity: str,
    project: str,
) -> list[leaderboard.LeaderboardColumn]:
    """
    Create LeaderboardColumns from benchmark name list
    
    Dynamically fetches evaluation ref for each benchmark to create columns.
    
    Args:
        benchmarks: List of benchmark names
        entity: Weave entity
        project: Weave project name
    
    Returns:
        List of LeaderboardColumn
    """
    # Benchmark-specific metric mapping
    # Format: (scorer_name, summary_metric_path)
    # output structure: {"scorer_name": {"metric": value, ...}, ...}
    BENCHMARK_METRICS = {
        # Default choice scorer
        "ko_hellaswag": ("choice", "true_fraction"),
        "ko_balt_700": ("choice", "true_fraction"),
        "haerae_bench_v1": ("choice", "true_fraction"),
        "kmmlu": ("choice", "true_fraction"),
        "kmmlu_pro": ("choice", "true_fraction"),
        "ko_truthful_qa": ("choice", "true_fraction"),
        "ko_moral": ("choice", "true_fraction"),
        "korean_hate_speech": ("choice", "true_fraction"),
        
        # model_graded_qa scorer
        "ko_aime2025": ("math_grader", "accuracy"),
        "hrm8k": ("math_grader", "accuracy"),
        
        # Special scorers
        "ifeval_ko": ("instruction_following", "prompt_level_strict.true_fraction"),
        "ko_arc_agi": ("grid_match", "true_fraction"),
        "squad_kor_v1": ("f1", "mean"),
        
        # KoBBQ
        "kobbq": ("kobbq_scorer", "true_fraction"),
        
        # HLE
        "ko_hle": ("hle_grader", "true_fraction"),
        
        # HalluLens
        "ko_hallulens_wikiqa": ("hallulens_qa", "true_fraction"),
        "ko_hallulens_nonexistent": ("hallulens_refusal", "true_fraction"),
        
        # BFCL
        "bfcl": ("bfcl_scorer", "true_fraction"),
        
        # MT-Bench
        "ko_mtbench": ("mtbench_scorer", "mean"),
        
        # SWE-bench
        "swebench_verified_official_80": ("swebench_server_scorer", "true_fraction"),
    }
    
    columns = []
    
    for benchmark in benchmarks:
        # Get actual evaluation ref (with digest)
        eval_ref = get_evaluation_ref(entity, project, benchmark)
        
        if not eval_ref:
            print(f"   ⚠️ {benchmark}-evaluation object not found")
            continue
        
        # Get metric for this benchmark
        scorer_name, metric_path = BENCHMARK_METRICS.get(
            benchmark, ("output", "true_fraction")
        )
        
        columns.append(
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=eval_ref,
                scorer_name=scorer_name,
                summary_metric_path=metric_path,
                should_minimize=False,
            )
        )
        print(f"   ✓ {benchmark}: {scorer_name}.{metric_path}")
    
    return columns


def create_weave_leaderboard(
    entity: str,
    project: str,
    benchmarks: list[str] | None = None,
    name: str = LEADERBOARD_NAME,
    description: str = LEADERBOARD_DESCRIPTION,
) -> str | None:
    """
    Create/Update Weave Leaderboard
    
    Creates a Weave Leaderboard from benchmark list.
    Merges new columns with existing leaderboard if it exists.
    
    Args:
        entity: Weave entity (team or username)
        project: Weave project name
        benchmarks: List of benchmark names (uses default list if None)
        name: Leaderboard name
        description: Leaderboard description
    
    Returns:
        Leaderboard URL (on success) or None (on failure)
    """
    print(f"\n{'='*60}")
    print(f"🏆 Weave Leaderboard Creation")
    print(f"{'='*60}")
    
    # Default benchmark list
    DEFAULT_BENCHMARKS = [
        "ko_hellaswag",
        "ko_aime2025",
        "ifeval_ko",
        "ko_balt_700",
        "haerae_bench_v1",
        "kmmlu",
        "kmmlu_pro",
        "squad_kor_v1",
        "ko_truthful_qa",
        "ko_moral",
        "ko_arc_agi",
        "hrm8k",
        "korean_hate_speech",
        "kobbq",
        "ko_hle",
        "ko_hallulens_wikiqa",
        "ko_hallulens_nonexistent",
        "bfcl",
        "ko_mtbench",
        "swebench_verified_official_80",
    ]
    
    benchmarks = benchmarks or DEFAULT_BENCHMARKS
    
    # Initialize Weave
    client = weave.get_client()
    if client is None:
        weave.init(f"{entity}/{project}")
        client = weave.get_client()
    
    if client is None:
        print("❌ Failed to initialize Weave client")
        return None
    
    try:
        # 1. Create LeaderboardColumns
        print(f"📊 Creating LeaderboardColumns from {len(benchmarks)} benchmarks...")
        new_columns = build_columns_from_benchmarks(benchmarks, entity, project)
        
        if not new_columns:
            print("⚠️ No columns to create.")
            return None
        
        print(f"   New columns: {len(new_columns)}")
        
        # 3. Get existing leaderboard (if exists)
        existing_columns: list[leaderboard.LeaderboardColumn] = []
        try:
            existing = weave.ref(LEADERBOARD_REF).get()
            cols = getattr(existing, "columns", None)
            if cols:
                existing_columns = list(cols)
                print(f"   Existing columns: {len(existing_columns)}")
        except Exception:
            print("   No existing leaderboard - creating new")
        
        # 4. Merge columns (remove duplicates)
        merged_columns = list(
            {
                (
                    column.evaluation_object_ref,
                    column.scorer_name,
                    column.summary_metric_path,
                    column.should_minimize,
                ): column
                for column in (existing_columns or []) + new_columns
            }.values()
        )
        
        print(f"\n📈 Creating leaderboard with {len(merged_columns)} total columns")
        
        # 5. Create and publish leaderboard
        spec = leaderboard.Leaderboard(
            name=name,
            description=description,
            columns=merged_columns,
        )
        ref = weave.publish(spec, name=LEADERBOARD_REF)
        
        url = weave_urls.leaderboard_path(
            ref.entity,
            ref.project,
            ref.name,
        )
        
        print(f"\n✅ Weave Leaderboard created!")
        print(f"🔗 URL: {url}")
        
        return url
        
    except Exception as e:
        print(f"❌ Leaderboard creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_weave_leaderboard_from_active_loggers(
    name: str = LEADERBOARD_NAME,
    description: str = LEADERBOARD_DESCRIPTION,
) -> str | None:
    """
    Create Weave Leaderboard from active EvaluationLoggers
    
    This function only works when evaluation is run in the same process.
    For subprocess execution, use create_weave_leaderboard().
    
    Args:
        name: Leaderboard name
        description: Leaderboard description
    
    Returns:
        Leaderboard URL (on success) or None (on failure)
    """
    from weave.evaluation.eval_imperative import _active_evaluation_loggers
    from weave.trace.ref_util import get_ref
    
    client = weave.get_client()
    if client is None:
        print("❌ Weave client is not initialized.")
        return None
    
    try:
        # Build columns from active loggers
        new_columns: list[leaderboard.LeaderboardColumn] = []
        
        for eval_logger in _active_evaluation_loggers:
            eval_output = eval_logger._evaluate_call and (eval_logger._evaluate_call.output or {})
            output_scorer = eval_output.get("output", {})
            
            for metric_name, metric_values in output_scorer.items():
                if not isinstance(metric_values, dict):
                    continue
                    
                for m_value in metric_values.keys():
                    if "err" in m_value.lower():
                        continue
                    
                    new_columns.append(
                        leaderboard.LeaderboardColumn(
                            evaluation_object_ref=get_ref(
                                eval_logger._pseudo_evaluation
                            ).uri(),
                            scorer_name="output",
                            summary_metric_path=f"{metric_name}.{m_value}",
                            should_minimize=False,
                        )
                    )
        
        if not new_columns:
            print("⚠️ No active evaluation loggers.")
            return None
        
        # Merge with existing leaderboard
        existing_columns: list[leaderboard.LeaderboardColumn] = []
        try:
            existing = weave.ref(LEADERBOARD_REF).get()
            cols = getattr(existing, "columns", None)
            if cols:
                existing_columns = list(cols)
        except Exception:
            pass
        
        merged_columns = list(
            {
                (
                    column.evaluation_object_ref,
                    column.scorer_name,
                    column.summary_metric_path,
                    column.should_minimize,
                ): column
                for column in (existing_columns or []) + new_columns
            }.values()
        )
        
        # Publish leaderboard
        spec = leaderboard.Leaderboard(
            name=name,
            description=description,
            columns=merged_columns,
        )
        ref = weave.publish(spec, name=LEADERBOARD_REF)
        
        url = weave_urls.leaderboard_path(
            ref.entity,
            ref.project,
            ref.name,
        )
        
        print(f"✅ Weave Leaderboard created!")
        print(f"🔗 URL: {url}")
        
        return url
        
    except Exception as e:
        print(f"❌ Leaderboard creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Weave Leaderboard")
    parser.add_argument("--entity", "-e", required=True, help="Weave entity")
    parser.add_argument("--project", "-p", required=True, help="Weave project")
    parser.add_argument("--benchmarks", "-b", nargs="+", help="Benchmark list (default: all)")
    parser.add_argument("--name", default=LEADERBOARD_NAME, help="Leaderboard name")
    
    args = parser.parse_args()
    
    create_weave_leaderboard(
        entity=args.entity,
        project=args.project,
        benchmarks=args.benchmarks,
        name=args.name,
    )
