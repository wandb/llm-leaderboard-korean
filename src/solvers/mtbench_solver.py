"""
MT-Bench Solver - 2턴 대화 처리

MT-Bench는 2턴 대화 형식:
1. Turn 1: 첫 번째 질문에 응답
2. Turn 2: 첫 번째 응답을 바탕으로 두 번째 질문에 응답
"""

from inspect_ai.solver import Solver, solver, Generate, TaskState
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant


@solver
def mtbench_solver() -> Solver:
    """
    MT-Bench 2턴 대화 Solver
    
    Turn 1: state.input (첫 번째 질문)
    Turn 2: metadata["turn2"] (두 번째 질문)
    
    결과는 metadata에 response_turn1, response_turn2로 저장됩니다.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata or {}
        
        # Turn 1: state.input에서 가져옴 (field_mapping["input"] = "turn1")
        turn1 = state.input if isinstance(state.input, str) else str(state.input)
        turn2 = metadata.get("turn2", "")
        
        # metadata에 turn1도 저장 (scorer에서 사용)
        state.metadata["turn1"] = turn1
        
        # Turn 1: 첫 번째 질문
        state.messages = [ChatMessageUser(content=turn1)]
        state = await generate(state)
        
        # Turn 1 응답 저장
        response_turn1 = state.output.completion if state.output else ""
        state.metadata["response_turn1"] = response_turn1
        
        # Turn 2가 있으면 계속
        if turn2:
            # 대화 히스토리에 추가
            state.messages.append(ChatMessageAssistant(content=response_turn1))
            state.messages.append(ChatMessageUser(content=turn2))
            
            # Turn 2 생성
            state = await generate(state)
            response_turn2 = state.output.completion if state.output else ""
            state.metadata["response_turn2"] = response_turn2
        else:
            state.metadata["response_turn2"] = ""
        
        return state
    
    return solve

