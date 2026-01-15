"""
BFCL - Berkeley Function Calling Leaderboard Benchmark (Integrated Version)

Automatically selects appropriate solver based on model's tool calling support.
- Native Tool Calling (default): Models that support tool calling (OpenAI, Claude, Gemini, etc.)
- Text-based: Models without tool calling support (EXAONE, some open-source models, etc.)

Model configuration (configs/models/<model>.yaml):
    benchmarks:
      bfcl:
        use_native_tools: true  # or false

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
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/BFCL_mini:ODywz9h7BWEfpYfAmkqjwLXQYxrsRWlPXCXNMoo3jTg",
    field_mapping={
        "id": "id",
        "input": "input",
        # No target - uses ground_truth from metadata
    },
    answer_format="identity",
    # solver is determined dynamically (based on use_native_tools setting)
    # default: bfcl_solver (native tool calling)
    solver="bfcl_solver",
    scorer="bfcl_scorer",
    # Balanced sampling to extract evenly from each category
    sampling="balanced",
    sampling_by="category",
    # Metadata: indicates this benchmark supports dynamic solver selection
    metadata={
        "supports_dynamic_solver": True,
        "native_solver": "bfcl_solver",
        "text_solver": "bfcl_text_solver",
    },
)
