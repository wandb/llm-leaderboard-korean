"""
Korean LLM Evaluation Benchmark Definitions

Each benchmark is managed in an individual file.
To add a new benchmark:
1. Create a new file in this folder (e.g., ko_new.py)
2. Define CONFIG dictionary
3. Add to BENCHMARKS in this file
4. Add @task function to horangi.py
"""

from benchmarks.ko_hellaswag import CONFIG as ko_hellaswag
from benchmarks.ko_aime2025 import CONFIG as ko_aime2025
from benchmarks.ifeval_ko import CONFIG as ifeval_ko
from benchmarks.ko_balt_700 import CONFIG as ko_balt_700
from benchmarks.ko_balt_700_syntax import CONFIG as ko_balt_700_syntax
from benchmarks.ko_balt_700_semantic import CONFIG as ko_balt_700_semantic
from benchmarks.haerae_bench_v1 import CONFIG as haerae_bench_v1
from benchmarks.haerae_bench_v1_rc import CONFIG as haerae_bench_v1_rc
from benchmarks.haerae_bench_v1_wo_rc import CONFIG as haerae_bench_v1_wo_rc
from benchmarks.kmmlu import CONFIG as kmmlu
from benchmarks.kmmlu_pro import CONFIG as kmmlu_pro
from benchmarks.squad_kor_v1 import CONFIG as squad_kor_v1
from benchmarks.ko_truthful_qa import CONFIG as ko_truthful_qa
from benchmarks.ko_moral import CONFIG as ko_moral
from benchmarks.ko_arc_agi import CONFIG as ko_arc_agi
from benchmarks.korean_hate_speech import CONFIG as korean_hate_speech
from benchmarks.kobbq import CONFIG as kobbq
from benchmarks.ko_hle import CONFIG as ko_hle
# HalluLens benchmarks
from benchmarks.ko_hallulens_wikiqa import CONFIG as ko_hallulens_wikiqa
from benchmarks.ko_hallulens_longwiki import CONFIG as ko_hallulens_longwiki
from benchmarks.ko_hallulens_generated import CONFIG as ko_hallulens_generated
from benchmarks.ko_hallulens_mixed import CONFIG as ko_hallulens_mixed
from benchmarks.ko_hallulens_nonexistent import CONFIG as ko_hallulens_nonexistent
# BFCL benchmarks
from benchmarks.bfcl import CONFIG as bfcl
from benchmarks.bfcl_extended import CONFIG as bfcl_extended  # backward compatibility
from benchmarks.bfcl_text import CONFIG as bfcl_text  # backward compatibility
# MT-Bench
from benchmarks.ko_mtbench import CONFIG as ko_mtbench
# SWE-bench
from benchmarks.swebench_verified_official_80 import CONFIG as swebench_verified_official_80
# HRM8K
from benchmarks.hrm8k import CONFIG as hrm8k

# Benchmark descriptions
BENCHMARK_DESCRIPTIONS: dict[str, str] = {
    # General
    "ko_hellaswag": "Common sense reasoning (sentence completion)",
    "ko_aime2025": "AIME 2025 math problems",
    "ifeval_ko": "Instruction following evaluation",
    "ko_balt_700": "Language understanding and reasoning",
    # Knowledge
    "haerae_bench_v1": "HAERAE v1",
    "haerae_bench_v1_rc": "HAERAE v1 (with reading comprehension)",
    "haerae_bench_v1_wo_rc": "HAERAE v1 (without reading comprehension)",
    "kmmlu": "Korean MMLU",
    "kmmlu_pro": "Korean MMLU Pro (advanced)",
    "squad_kor_v1": "Korean reading comprehension QA",
    "ko_truthful_qa": "Truthfulness evaluation",
    # Reasoning
    "ko_moral": "Moral judgment",
    "ko_arc_agi": "ARC-AGI reasoning (grid)",
    # Bias/Safety
    "korean_hate_speech": "Hate speech detection",
    "kobbq": "Bias judgment (BBQ)",
    "ko_hle": "Humanity's Last Exam",
    # HalluLens (Hallucination)
    "ko_hallulens_wikiqa": "Wiki QA hallucination evaluation",
    "ko_hallulens_longwiki": "Long document QA hallucination evaluation",
    "ko_hallulens_generated": "Generated entity refusal (generated)",
    "ko_hallulens_mixed": "Generated entity refusal (mixed)",
    "ko_hallulens_nonexistent": "Generated entity refusal (combined)",
    # Function Calling
    "bfcl": "Function calling (unified, auto-select based on model config)",
    "bfcl_extended": "Function calling (Native Tool) [deprecated]",
    "bfcl_text": "Function calling (Text-based) [deprecated]",
    # Conversation
    "ko_mtbench": "Multi-turn conversation evaluation",
    # Coding
    "swebench_verified_official_80": "SWE-bench bug fix (80 tasks)",
    # Math Reasoning
    "hrm8k": "Korean math reasoning (HRM8K)",
}

# All benchmark configurations
BENCHMARKS: dict = {
    "ko_hellaswag": ko_hellaswag,
    "ko_aime2025": ko_aime2025,
    "ifeval_ko": ifeval_ko,
    "ko_balt_700": ko_balt_700,
    "ko_balt_700_syntax": ko_balt_700_syntax,
    "ko_balt_700_semantic": ko_balt_700_semantic,
    "haerae_bench_v1": haerae_bench_v1,
    "haerae_bench_v1_rc": haerae_bench_v1_rc,
    "haerae_bench_v1_wo_rc": haerae_bench_v1_wo_rc,
    "kmmlu": kmmlu,
    "kmmlu_pro": kmmlu_pro,
    "squad_kor_v1": squad_kor_v1,
    "ko_truthful_qa": ko_truthful_qa,
    "ko_moral": ko_moral,
    "ko_arc_agi": ko_arc_agi,
    "korean_hate_speech": korean_hate_speech,
    "kobbq": kobbq,
    "ko_hle": ko_hle,
    # HalluLens benchmarks
    "ko_hallulens_wikiqa": ko_hallulens_wikiqa,
    "ko_hallulens_longwiki": ko_hallulens_longwiki,
    "ko_hallulens_generated": ko_hallulens_generated,
    "ko_hallulens_mixed": ko_hallulens_mixed,
    "ko_hallulens_nonexistent": ko_hallulens_nonexistent,
    # BFCL benchmarks
    "bfcl": bfcl,
    "bfcl_extended": bfcl_extended,  # backward compatibility
    "bfcl_text": bfcl_text,  # backward compatibility
    # MT-Bench
    "ko_mtbench": ko_mtbench,
    # SWE-bench
    "swebench_verified_official_80": swebench_verified_official_80,
    # HRM8K
    "hrm8k": hrm8k,
}

def get_benchmark_config(name: str) -> dict:
    """Get benchmark configuration (returns dict)"""
    if name not in BENCHMARKS:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    
    config = BENCHMARKS[name]
    
    # Convert BenchmarkConfig dataclass to dict if needed
    if hasattr(config, "to_dict"):
        return config.to_dict()
    
    # Return dict as-is
    return config


def list_benchmarks() -> list[str]:
    """List available benchmarks"""
    return list(BENCHMARKS.keys())


def list_benchmarks_with_descriptions() -> list[tuple[str, str]]:
    """Return benchmark list with descriptions"""
    return [
        (name, BENCHMARK_DESCRIPTIONS.get(name, ""))
        for name in BENCHMARKS.keys()
    ]


__all__ = [
    "BENCHMARKS",
    "BENCHMARK_DESCRIPTIONS",
    "get_benchmark_config",
    "list_benchmarks",
    "list_benchmarks_with_descriptions",
]
