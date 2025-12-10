"""
한국어 평가를 위한 커스텀 Solver

한국어에 최적화된 프롬프팅 전략을 제공합니다.
"""

from inspect_ai.solver import Solver, TaskState, solver, Generate
from inspect_ai.model import ChatMessageUser, ChatMessageSystem


@solver
def korean_system_message(
    persona: str = "전문가",
    task_description: str = "",
) -> Solver:
    """
    한국어 시스템 메시지 Solver

    Args:
        persona: AI의 페르소나 (예: "전문가", "교수", "연구원")
        task_description: 추가 작업 설명

    Returns:
        Solver: 시스템 메시지를 추가하는 Solver
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        system_content = f"""당신은 {persona}입니다.
모든 답변은 한국어로 작성해주세요.
{task_description}

답변 시 다음 사항을 지켜주세요:
- 정확하고 사실에 기반한 답변을 제공하세요.
- 불확실한 경우 그 점을 명시하세요.
- 간결하고 명확하게 답변하세요."""

        state.messages.insert(0, ChatMessageSystem(content=system_content))
        return state

    return solve


@solver
def korean_cot_prompt(
    instruction: str = "단계별로 생각해보세요.",
) -> Solver:
    """
    한국어 Chain-of-Thought 프롬프트 Solver

    Args:
        instruction: CoT 유도 지시문

    Returns:
        Solver: CoT 프롬프트를 추가하는 Solver
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cot_prompt = f"""
{instruction}

문제를 해결하기 위해:
1. 먼저 주어진 정보를 정리하세요.
2. 단계별로 논리적으로 분석하세요.
3. 최종 답변을 명확하게 제시하세요.

답변 형식:
[분석]
(여기에 단계별 분석을 작성)

[최종 답변]
(여기에 최종 답변을 작성)
"""
        if state.messages and isinstance(state.messages[-1], ChatMessageUser):
            state.messages[-1] = ChatMessageUser(
                content=state.messages[-1].content + "\n\n" + cot_prompt
            )
        return state

    return solve


@solver
def korean_few_shot(
    examples: list[dict],
) -> Solver:
    """
    한국어 Few-shot 프롬프트 Solver

    Args:
        examples: 예시 목록 [{"question": "...", "answer": "..."}]

    Returns:
        Solver: Few-shot 예시를 추가하는 Solver
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        examples_text = "다음은 예시입니다:\n\n"
        
        for i, example in enumerate(examples, 1):
            examples_text += f"예시 {i}:\n"
            examples_text += f"질문: {example.get('question', '')}\n"
            examples_text += f"답변: {example.get('answer', '')}\n\n"
        
        examples_text += "이제 다음 질문에 답해주세요:\n"
        
        if state.messages and isinstance(state.messages[-1], ChatMessageUser):
            original_content = state.messages[-1].content
            state.messages[-1] = ChatMessageUser(
                content=examples_text + original_content
            )
        
        return state

    return solve


@solver
def korean_answer_format(
    format_instruction: str = "답변만 간단히 작성해주세요.",
) -> Solver:
    """
    한국어 답변 형식 지정 Solver

    Args:
        format_instruction: 답변 형식 지시문

    Returns:
        Solver: 답변 형식 지시를 추가하는 Solver
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        format_prompt = f"\n\n주의: {format_instruction}"
        
        if state.messages and isinstance(state.messages[-1], ChatMessageUser):
            state.messages[-1] = ChatMessageUser(
                content=state.messages[-1].content + format_prompt
            )
        
        return state

    return solve

