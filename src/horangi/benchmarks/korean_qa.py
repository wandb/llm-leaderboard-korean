"""
한국어 질의응답 (Korean QA) 벤치마크

한국어 읽기 이해 및 질의응답 능력을 평가합니다.
"""

from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.scorer import match, model_graded_qa
from inspect_ai.solver import generate, system_message

# 데이터 경로
DATA_DIR = Path(__file__).parent / "data"


def korean_qa_dataset():
    """한국어 QA 데이터셋 로드"""
    return json_dataset(str(DATA_DIR / "korean_qa.jsonl"))


KOREAN_QA_SYSTEM_PROMPT = """당신은 한국어 질의응답 전문가입니다.
주어진 지문을 읽고 질문에 대해 정확하게 답변해주세요.
답변은 간결하고 명확하게 작성해주세요."""


@task
def korean_qa(
    use_cot: bool = False,
    use_model_grading: bool = False,
):
    """
    한국어 질의응답 벤치마크

    Args:
        use_cot: Chain-of-thought 프롬프팅 사용 여부
        use_model_grading: 모델 기반 채점 사용 여부 (False면 exact match 사용)

    Returns:
        Task: Inspect AI Task 객체
    """
    from inspect_ai.solver import chain_of_thought

    # Solver 구성
    solvers = [system_message(KOREAN_QA_SYSTEM_PROMPT)]
    
    if use_cot:
        solvers.append(chain_of_thought())
    
    solvers.append(generate())

    # Scorer 선택
    if use_model_grading:
        scorer = model_graded_qa(
            template="""다음 질문에 대한 정답과 모델의 답변을 비교하세요.

질문: {question}
정답: {target}
모델 답변: {answer}

모델의 답변이 정답과 의미적으로 일치하면 CORRECT, 그렇지 않으면 INCORRECT로 판정하세요.
답변의 형식이나 표현이 다르더라도 의미가 같으면 CORRECT입니다.

판정:""",
        )
    else:
        scorer = match()

    return Task(
        dataset=korean_qa_dataset(),
        solver=solvers,
        scorer=scorer,
        name="korean_qa",
        metadata={
            "benchmark": "korean_qa",
            "language": "ko",
            "task_type": "question_answering",
            "use_cot": use_cot,
            "use_model_grading": use_model_grading,
        },
    )

