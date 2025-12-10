"""
한국어 상식 추론 (Korean Commonsense) 벤치마크

한국어 상식 추론 및 사회적 맥락 이해 능력을 평가합니다.
HellaSwag, WinoGrande 스타일의 한국어 버전입니다.
"""

from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import choice, model_graded_fact
from inspect_ai.solver import generate, system_message, multiple_choice

# 데이터 경로
DATA_DIR = Path(__file__).parent / "data"


def korean_commonsense_dataset():
    """한국어 상식 데이터셋 로드"""
    return json_dataset(str(DATA_DIR / "korean_commonsense.jsonl"))


KOREAN_COMMONSENSE_SYSTEM_PROMPT = """당신은 한국어와 한국 문화에 대한 깊은 이해를 가진 전문가입니다.
주어진 상황에서 가장 자연스럽고 상식적인 답변을 선택해주세요."""


@task
def korean_commonsense(
    use_cot: bool = False,
):
    """
    한국어 상식 추론 벤치마크

    Args:
        use_cot: Chain-of-thought 프롬프팅 사용 여부

    Returns:
        Task: Inspect AI Task 객체
    """
    from inspect_ai.solver import chain_of_thought

    solvers = [system_message(KOREAN_COMMONSENSE_SYSTEM_PROMPT)]
    
    if use_cot:
        solvers.append(chain_of_thought())
    
    solvers.append(multiple_choice())

    return Task(
        dataset=korean_commonsense_dataset(),
        solver=solvers,
        scorer=choice(),
        name="korean_commonsense",
        metadata={
            "benchmark": "korean_commonsense",
            "language": "ko",
            "task_type": "commonsense_reasoning",
            "use_cot": use_cot,
        },
    )


@task
def korean_hellaswag():
    """
    한국어 HellaSwag 스타일 벤치마크
    문장 완성 상식 추론
    """
    return korean_commonsense(use_cot=False)


@task
def korean_winogrande():
    """
    한국어 WinoGrande 스타일 벤치마크
    대명사 해소 상식 추론
    """
    return korean_commonsense(use_cot=True)

