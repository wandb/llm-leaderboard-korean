"""
한국어 지식 (Korean Knowledge) 벤치마크

한국 역사, 문화, 사회 등에 대한 지식을 평가합니다.
"""

from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import match, choice
from inspect_ai.solver import generate, system_message, multiple_choice

# 데이터 경로
DATA_DIR = Path(__file__).parent / "data"


def korean_knowledge_dataset():
    """한국어 지식 데이터셋 로드"""
    return json_dataset(str(DATA_DIR / "korean_knowledge.jsonl"))


KOREAN_KNOWLEDGE_SYSTEM_PROMPT = """당신은 한국에 대한 전문 지식을 갖춘 전문가입니다.
주어진 질문에 대해 정확하게 답변해주세요.
객관식 문제의 경우 정답 번호만 답변해주세요."""


@task
def korean_knowledge(
    is_multiple_choice: bool = True,
):
    """
    한국어 지식 벤치마크

    Args:
        is_multiple_choice: 객관식 문제 여부

    Returns:
        Task: Inspect AI Task 객체
    """
    solvers = [system_message(KOREAN_KNOWLEDGE_SYSTEM_PROMPT)]
    
    if is_multiple_choice:
        solvers.append(multiple_choice())
        scorer = choice()
    else:
        solvers.append(generate())
        scorer = match()

    return Task(
        dataset=korean_knowledge_dataset(),
        solver=solvers,
        scorer=scorer,
        name="korean_knowledge",
        metadata={
            "benchmark": "korean_knowledge",
            "language": "ko",
            "task_type": "knowledge",
            "is_multiple_choice": is_multiple_choice,
        },
    )


@task
def korean_history():
    """한국 역사 지식 벤치마크"""
    return korean_knowledge(is_multiple_choice=True)


@task
def korean_culture():
    """한국 문화 지식 벤치마크"""
    return korean_knowledge(is_multiple_choice=True)

