"""
KoMT-Bench - Korean Multi-turn Conversation Evaluation Benchmark

Source: https://huggingface.co/datasets/LGAI-EXAONE/KoMT-Bench (LG AI Research)

8 categories (writing, roleplay, reasoning, math, coding, extraction, stem, humanities)
10 questions per category, 80 total
2-turn conversation format evaluated by LLM Judge on 1-10 scale
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoMTBench_mini:GY9L798k1ezXyTlk7ILVZtAK0c3ii1ysPM7y1ahmCag",
    field_mapping={
        "id": "id",
        "input": "turn1",  # First question is input
        # No target - evaluated by LLM Judge
    },
    answer_format="identity",
    solver="mtbench_solver",
    scorer="mtbench_scorer",
    # system_message is not handled in solver (only question is passed)
)
