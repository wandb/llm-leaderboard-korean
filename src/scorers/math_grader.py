"""
Math Grader - 수학 문제용 스코어러

수학 문제의 정답을 정확한 숫자 비교로 채점합니다.
- 모델 출력에서 "\\boxed{X}" 형식으로 숫자 추출
- 정답과 숫자 비교 (정수/소수점 허용, 소수점 이하 오차 허용)
- LLM as judge 없이 정확한 숫자 매칭
"""

import re
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    scorer,
    accuracy,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import TaskState


def _extract_boxed_answer(response: str) -> str | None:
    """
    응답에서 '\\boxed{X}' 형식의 답을 추출
    
    지원 형식:
    - \\boxed{7}
    - \\boxed{70.0}
    - \\boxed{-3.14}
    - \\boxed{1/2}
    - \\boxed{A} (객관식)
    """
    # \boxed{...} 패턴 (중첩 괄호 처리)
    patterns = [
        r"\\boxed\{([^{}]+)\}",  # \boxed{7}
        r"\\boxed\s*\{([^{}]+)\}",  # \boxed {7}
        r"boxed\{([^{}]+)\}",  # boxed{7} (백슬래시 없이)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            # 마지막 \boxed{} 값 반환
            return matches[-1].strip()
    
    return None


def _extract_fallback_answer(response: str) -> str | None:
    """fallback: 다른 형식에서 답 추출"""
    patterns = [
        r"(?:정답|답|answer|Answer)[:\s]+(-?[\d.]+(?:/[\d.]+)?)",  # 정답: 7
        r"(?:정답|답|answer|Answer)[:\s]+([A-Za-z])\b",  # 정답: A (객관식)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def _extract_last_number(response: str) -> str | None:
    """응답에서 마지막 숫자를 추출 (최후 fallback)"""
    numbers = re.findall(r"-?[\d.]+(?:/[\d.]+)?", response)
    if numbers:
        return numbers[-1]
    return None


def _parse_number(value: str) -> float | None:
    """문자열을 숫자로 변환 (분수 지원)"""
    try:
        value = value.strip()
        # 분수 처리
        if "/" in value:
            parts = value.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def _numbers_equal(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """두 숫자가 같은지 비교 (허용 오차 포함)"""
    # 정수 비교 (둘 다 정수인 경우)
    if a == int(a) and b == int(b):
        return int(a) == int(b)
    # 소수점 비교 (오차 허용)
    return abs(a - b) < tolerance


@scorer(metrics=[accuracy()])
def math_grader(tolerance: float = 1e-6) -> Scorer:
    """
    수학 문제용 스코어러
    
    모델 출력에서 \\boxed{X} 형식의 답을 추출하고 정답과 비교합니다.
    
    Args:
        tolerance: 소수점 비교 시 허용 오차 (기본값: 1e-6)
    
    채점 방식:
    1. "\\boxed{X}" 형식에서 답 추출 시도
    2. 실패시 "정답: X" 등 다른 형식 시도
    3. 최후로 마지막 숫자 추출 (fallback)
    4. 정답과 비교 (숫자면 숫자 비교, 아니면 문자열 비교)
    """
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion if state.output else ""
        target_text = (target.text if target else "").strip()
        
        # 입력 문제 (input)
        input_text = state.input_text if hasattr(state, 'input_text') else ""
        if not input_text and state.messages:
            # 첫 번째 user 메시지에서 입력 추출
            for msg in state.messages:
                if msg.role == "user":
                    input_text = msg.text if hasattr(msg, 'text') else str(msg.content)
                    break
        
        # 1. 응답에서 답 추출 (우선순위: boxed > fallback > last number)
        extraction_method = None
        extracted = _extract_boxed_answer(response)
        if extracted is not None:
            extraction_method = "boxed"
        
        if extracted is None:
            extracted = _extract_fallback_answer(response)
            if extracted is not None:
                extraction_method = "fallback"
        
        if extracted is None:
            extracted = _extract_last_number(response)
            if extracted is not None:
                extraction_method = "last_number"
        
        if extracted is None:
            return Score(
                value=INCORRECT,
                answer=None,
                explanation="응답에서 답을 찾을 수 없음",
                metadata={
                    "target": target_text,
                    "extracted": None,
                    "extraction_method": None,
                    "model_response": response[:500] if response else None,  # 처음 500자
                },
            )
        
        # 2. 정답과 비교
        target_num = _parse_number(target_text)
        response_num = _parse_number(extracted)
        
        # 둘 다 숫자인 경우: 숫자 비교
        if target_num is not None and response_num is not None:
            is_correct = _numbers_equal(response_num, target_num, tolerance)
            return Score(
                value=CORRECT if is_correct else INCORRECT,
                answer=extracted,
                explanation=f"정답: {target_num}, 추출된 답: {response_num}, 일치: {'✓' if is_correct else '✗'}",
                metadata={
                    "target": target_text,
                    "target_parsed": target_num,
                    "extracted": extracted,
                    "extracted_parsed": response_num,
                    "extraction_method": extraction_method,
                    "is_correct": is_correct,
                },
            )
        
        # 문자열 비교 (객관식 등)
        is_correct = extracted.strip().upper() == target_text.strip().upper()
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=extracted,
            explanation=f"정답: {target_text}, 추출된 답: {extracted}, 일치: {'✓' if is_correct else '✗'}",
            metadata={
                "target": target_text,
                "extracted": extracted,
                "extraction_method": extraction_method,
                "is_correct": is_correct,
            },
        )
    
    return score

