"""
HumanEval 100 - Stratified sampled HumanEval benchmark

100 samples from OpenAI HumanEval, stratified by code complexity (solution length).
Uses inspect_evals.humaneval as base for solver/scorer.
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    base="inspect_evals.humaneval.humaneval",
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/HumanEval_100:3uFy8Hxjt2bcb5jBD7hlnQr4ZpNa03T81euMB0Wsh14",
    sandbox="local",  # Run code locally without Docker
)
