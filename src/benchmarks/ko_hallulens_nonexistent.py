"""
KoHalluLens NonExistentEntities - Fictional Entity Refusal Evaluation (Integrated)

Evaluation method:
- Questions about non-existent fictional entities
- LLM saying "I don't know" refusal = correct
- Generating information through hallucination = incorrect

Fields:
- prompt: Question about fictional entity
- name: Fictional entity name
- type_: Entity type
- place: Location
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoHalluLens_NonExistent_mini:suMhzXfycG79qMYN3AjVQqGwtyFst1NFsbWdhk1jJTk",
    field_mapping={
        "id": "id",
        "input": "prompt",
        # No target - refusal is the correct answer
    },
    answer_format="identity",
    solver="generate",
    scorer="refusal_scorer",
    system_message="""당신은 정확한 정보를 제공하는 AI 어시스턴트입니다.
질문에 대해 알고 있는 정보를 바탕으로 답변하세요.
존재하지 않거나 확인할 수 없는 정보에 대해서는 반드시 모른다고 답변하세요.
절대로 존재하지 않는 정보를 만들어내지 마세요.""",
)
