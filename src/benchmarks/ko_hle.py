"""
KoHLE Standalone - Korean Humanity's Last Exam Benchmark (Independent Version)

Implemented independently without inheriting from inspect_evals.hle.
Uses custom hle_grader (can specify separate judge model).
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    # No base - independent benchmark
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoHLE_mini:UrNXEnhaUHDoqButTAy204OEEevet6Pa1iSRYfnnnPY",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
    },
    answer_format="identity",  # Use answer as-is
    solver="generate",
    scorer="hle_grader",
    system_message="""답변은 다음 형식으로 작성해 주십시오.
설명: {답변 선택에 대한 설명}
답변: {선택한 답변}
확신도: {답변에 대한 확신도 (0%~100%)}""",
)
