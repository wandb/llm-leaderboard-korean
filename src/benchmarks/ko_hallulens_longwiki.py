"""
KoHalluLens LongWiki - Korean Long Wikipedia Document Based QA Hallucination Evaluation

Evaluation method:
- Provide prompt only to LLM without reference (contains multiple questions)
- Compare LLM answer with reference + answer to evaluate accuracy
- LLM evaluator (GPT-4o-mini) classifies as Correct/Hallucinated/Refused

Fields:
- prompt: Multiple questions (numbered format)
- answer: Multiple answers (numbered format)
- reference: Wikipedia original text (translated to Korean)
- reference_en: Original English Wikipedia
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoHalluLens_LongWiki_mini:VktVotlYffXkFz0VT5sKgXrEmItplwFb3R97zb6syEA",
    field_mapping={
        "id": "id",
        "input": "prompt",
        "target": "answer",
    },
    answer_format="identity",
    solver="generate",
    scorer="hallulens_qa_scorer",
    system_message="""당신은 정확한 정보를 제공하는 AI 어시스턴트입니다.
각 질문에 대해 알고 있는 정보를 바탕으로 답변하세요.
확실하지 않은 경우 모른다고 말하세요.
답변은 질문 번호에 맞춰서 작성해 주세요.""",
)
