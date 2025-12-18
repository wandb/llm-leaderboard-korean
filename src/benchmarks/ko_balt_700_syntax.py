"""
KoBALT-700 Syntax - Korean MCQA Benchmark

Independent benchmark (no base) - solver/scorer directly specified
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/KoBALT-700-syntax:UkRzrRi96jX1YIXN0TV065Ssy8IiSkQ9FngkCIR9O7E",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="letter",  # Data's answer is already a letter (A, B, C, ...)
    solver="multiple_choice",
    scorer="choice",
    system_message="주어진 질문에 가장 적절한 답을 선택하세요.",
)
