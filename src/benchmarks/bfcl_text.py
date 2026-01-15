"""
BFCL Text-based - Function Calling Benchmark for Models Without Tool Calling Support

Induces function calls through prompts.
Used for EXAONE, some open-source models, etc.

Usage:
    uv run horangi bfcl_text --model vllm/LGAI-EXAONE/EXAONE-3.5-32B-Instruct
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/BFCL_Extended:latest",
    field_mapping={
        "id": "id",
        "input": "input",
        # tools, ground_truth, category are automatically stored in metadata
    },
    answer_format="identity",
    solver="bfcl_text_solver",  # Text-based solver
    scorer="bfcl_scorer",
    sampling="balanced",
    sampling_by="category",
    # System prompt is set in solver
)
