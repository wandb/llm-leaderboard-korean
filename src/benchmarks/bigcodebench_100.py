"""
BigCodeBench 100 - Stratified sampled BigCodeBench benchmark

100 samples from BigCodeBench, stratified by library usage (pandas, numpy, sklearn, other).
Uses inspect_evals.bigcodebench as base for solver/scorer.
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    base="inspect_evals.bigcodebench.bigcodebench",
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/BigCodeBench_100:YIZRzAx2mGCbXZarXh5TQkDDmwK2AsKfwuxOO9BlhcQ",
    sandbox="local",  # Run code locally without Docker
)
