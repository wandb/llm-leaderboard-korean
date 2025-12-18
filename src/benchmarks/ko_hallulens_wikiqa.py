"""
KoHalluLens PreciseWikiQA - Korean Wikipedia Based QA Hallucination Evaluation

Evaluation method:
- Provide prompt only to LLM without reference
- Compare LLM answer with reference + answer to evaluate accuracy
- LLM evaluator (GPT-4o-mini) classifies as Correct/Hallucinated/Refused

Fields:
- prompt: Question
- answer: Correct answer
- reference: Wikipedia original text (translated to Korean)
- reference_en: Original English Wikipedia
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoHalluLens_WikiQA_mini:rU9poRP5fcXtp7mZsuRYYDNKPK51OkMRJTuXjyXP9WI",
    field_mapping={
        "id": "id",
        "input": "prompt",
        "target": "answer",
    },
    answer_format="identity",
    solver="generate",
    scorer="hallulens_qa_scorer",
    system_message="""당신은 정확한 정보를 제공하는 AI 어시스턴트입니다.
질문에 대해 알고 있는 정보를 바탕으로 답변하세요.
확실하지 않은 경우 모른다고 말하세요.""",
)
