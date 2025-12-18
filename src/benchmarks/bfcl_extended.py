"""
BFCL Extended - Extended Function Calling Benchmark (Independent Version)

Implemented independently without inheriting from inspect_evals.bfcl.
Uses custom bfcl_solver and bfcl_scorer.

Supported splits (150 samples):
- simple: Single function call (30)
- multiple: Choose from multiple functions (30)
- exec_simple: Executable simple call (30)
- exec_multiple: Executable multiple call (30)
- irrelevance: Reject irrelevant function (30)

Excluded:
- parallel*: Parallel calls
- multi_turn*: Multi-turn conversation
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    # No base - independent benchmark
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/BFCL_Extended:latest",
    field_mapping={
        "id": "id",
        "input": "input",
        # No target - uses ground_truth from metadata
    },
    answer_format="identity",
    solver="bfcl_solver",  # Custom solver
    scorer="bfcl_scorer",  # Custom scorer
    # Balanced sampling to extract evenly from each category
    sampling="balanced",
    sampling_by="category",
)
