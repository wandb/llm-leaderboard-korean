"""
한국어 추론 (Korean Reasoning) 벤치마크

한국어 논리적 추론 및 수리 추론 능력을 평가합니다.
"""

from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import match, model_graded_fact
from inspect_ai.solver import generate, system_message, chain_of_thought

# 데이터 경로
DATA_DIR = Path(__file__).parent / "data"


def korean_reasoning_dataset():
    """한국어 추론 데이터셋 로드"""
    return json_dataset(str(DATA_DIR / "korean_reasoning.jsonl"))


KOREAN_REASONING_SYSTEM_PROMPT = """당신은 논리적 추론 전문가입니다.
주어진 문제를 단계별로 분석하고 논리적으로 답을 도출해주세요.
최종 답변은 명확하게 제시해주세요."""


@task
def korean_reasoning(
    use_cot: bool = True,
    use_self_critique: bool = False,
):
    """
    한국어 추론 벤치마크

    Args:
        use_cot: Chain-of-thought 프롬프팅 사용 여부 (기본: True)
        use_self_critique: Self-critique 사용 여부

    Returns:
        Task: Inspect AI Task 객체
    """
    from inspect_ai.solver import self_critique

    # Solver 구성
    solvers = [system_message(KOREAN_REASONING_SYSTEM_PROMPT)]
    
    if use_cot:
        solvers.append(chain_of_thought())
    
    solvers.append(generate())
    
    if use_self_critique:
        solvers.append(self_critique())

    return Task(
        dataset=korean_reasoning_dataset(),
        solver=solvers,
        scorer=match(),
        name="korean_reasoning",
        metadata={
            "benchmark": "korean_reasoning",
            "language": "ko",
            "task_type": "reasoning",
            "use_cot": use_cot,
            "use_self_critique": use_self_critique,
        },
    )


@task
def korean_math_reasoning():
    """
    한국어 수학 추론 벤치마크 (Chain-of-thought 필수)
    """
    return korean_reasoning(use_cot=True, use_self_critique=False)


@task
def korean_logical_reasoning():
    """
    한국어 논리 추론 벤치마크
    """
    return korean_reasoning(use_cot=True, use_self_critique=True)

