"""
Inspect CLI에서 직접 실행 가능한 Task 정의 파일

사용법:
    # 단일 벤치마크 실행
    inspect eval eval_tasks.py@korean_qa --model openai/gpt-4o

    # 모든 벤치마크 실행
    inspect eval eval_tasks.py --model openai/gpt-4o

inspect-wandb가 설치되어 있으면 자동으로 WandB/Weave에 로깅됩니다.
"""

from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import match, choice
from inspect_ai.solver import generate, system_message, multiple_choice, chain_of_thought

# 데이터 경로
DATA_DIR = Path(__file__).parent / "src" / "horangi" / "benchmarks" / "data"


# =============================================================================
# 한국어 QA 벤치마크
# =============================================================================

KOREAN_QA_SYSTEM_PROMPT = """당신은 한국어 질의응답 전문가입니다.
주어진 지문을 읽고 질문에 대해 정확하게 답변해주세요.
답변은 간결하고 명확하게 작성해주세요."""


@task
def korean_qa(use_cot: bool = False):
    """한국어 질의응답 벤치마크"""
    solvers = [system_message(KOREAN_QA_SYSTEM_PROMPT)]
    if use_cot:
        solvers.append(chain_of_thought())
    solvers.append(generate())

    return Task(
        dataset=json_dataset(str(DATA_DIR / "korean_qa.jsonl")),
        solver=solvers,
        scorer=match(),
        name="korean_qa",
    )


# =============================================================================
# 한국어 추론 벤치마크
# =============================================================================

KOREAN_REASONING_SYSTEM_PROMPT = """당신은 논리적 추론 전문가입니다.
주어진 문제를 단계별로 분석하고 논리적으로 답을 도출해주세요.
최종 답변은 명확하게 제시해주세요."""


@task
def korean_reasoning(use_cot: bool = True):
    """한국어 추론 벤치마크"""
    solvers = [system_message(KOREAN_REASONING_SYSTEM_PROMPT)]
    if use_cot:
        solvers.append(chain_of_thought())
    solvers.append(generate())

    return Task(
        dataset=json_dataset(str(DATA_DIR / "korean_reasoning.jsonl")),
        solver=solvers,
        scorer=match(),
        name="korean_reasoning",
    )


@task
def korean_math_reasoning():
    """한국어 수학 추론 벤치마크"""
    return korean_reasoning(use_cot=True)


# =============================================================================
# 한국어 지식 벤치마크
# =============================================================================

KOREAN_KNOWLEDGE_SYSTEM_PROMPT = """당신은 한국에 대한 전문 지식을 갖춘 전문가입니다.
주어진 질문에 대해 정확하게 답변해주세요.
객관식 문제의 경우 정답 번호만 답변해주세요."""


@task
def korean_knowledge():
    """한국어 지식 벤치마크"""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "korean_knowledge.jsonl")),
        solver=[
            system_message(KOREAN_KNOWLEDGE_SYSTEM_PROMPT),
            multiple_choice(),
        ],
        scorer=choice(),
        name="korean_knowledge",
    )


# =============================================================================
# 한국어 상식 벤치마크
# =============================================================================

KOREAN_COMMONSENSE_SYSTEM_PROMPT = """당신은 한국어와 한국 문화에 대한 깊은 이해를 가진 전문가입니다.
주어진 상황에서 가장 자연스럽고 상식적인 답변을 선택해주세요."""


@task
def korean_commonsense(use_cot: bool = False):
    """한국어 상식 추론 벤치마크"""
    solvers = [system_message(KOREAN_COMMONSENSE_SYSTEM_PROMPT)]
    if use_cot:
        solvers.append(chain_of_thought())
    solvers.append(multiple_choice())

    return Task(
        dataset=json_dataset(str(DATA_DIR / "korean_commonsense.jsonl")),
        solver=solvers,
        scorer=choice(),
        name="korean_commonsense",
    )
