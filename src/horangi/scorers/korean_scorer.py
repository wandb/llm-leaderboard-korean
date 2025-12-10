"""
한국어 평가를 위한 커스텀 Scorer

한국어 특성을 고려한 채점 로직을 제공합니다.
"""

import re
from inspect_ai.scorer import (
    Scorer,
    Score,
    Target,
    scorer,
    accuracy,
    CORRECT,
    INCORRECT,
    PARTIAL,
)
from inspect_ai.solver import TaskState


def normalize_korean_text(text: str) -> str:
    """
    한국어 텍스트 정규화

    - 공백 정규화
    - 특수문자 제거
    - 조사 변형 처리
    """
    # 공백 정규화
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 특수문자 제거 (한글, 영문, 숫자만 유지)
    text = re.sub(r'[^\w\s가-힣]', '', text)
    
    # 소문자 변환 (영문의 경우)
    text = text.lower()
    
    return text


def extract_korean_answer(text: str) -> str:
    """
    응답에서 최종 답변 추출

    [최종 답변], [답], 답: 등의 패턴을 찾아 답변 추출
    """
    # 패턴들 (우선순위 순)
    patterns = [
        r'\[최종\s*답변\]\s*:?\s*(.+?)(?:\n|$)',
        r'\[답\]\s*:?\s*(.+?)(?:\n|$)',
        r'최종\s*답변\s*:?\s*(.+?)(?:\n|$)',
        r'답\s*:?\s*(.+?)(?:\n|$)',
        r'정답\s*:?\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # 패턴을 찾지 못하면 마지막 문장 반환
    sentences = text.strip().split('\n')
    return sentences[-1].strip() if sentences else text


@scorer(metrics=[accuracy()])
def korean_semantic_match() -> Scorer:
    """
    한국어 의미적 일치 Scorer

    정규화된 텍스트 비교를 수행합니다.
    """
    async def score(state: TaskState, target: Target) -> Score:
        # 모델 응답 추출
        model_answer = state.output.completion if state.output else ""
        extracted_answer = extract_korean_answer(model_answer)
        normalized_answer = normalize_korean_text(extracted_answer)
        
        # 정답 정규화
        target_text = target.text if target.text else ""
        normalized_target = normalize_korean_text(target_text)
        
        # 완전 일치 확인
        if normalized_answer == normalized_target:
            return Score(
                value=CORRECT,
                answer=extracted_answer,
                explanation="정확히 일치합니다.",
            )
        
        # 부분 일치 확인 (정답이 응답에 포함된 경우)
        if normalized_target in normalized_answer:
            return Score(
                value=PARTIAL,
                answer=extracted_answer,
                explanation=f"부분 일치: 정답 '{target_text}'이(가) 응답에 포함되어 있습니다.",
            )
        
        return Score(
            value=INCORRECT,
            answer=extracted_answer,
            explanation=f"불일치: 정답은 '{target_text}'입니다.",
        )

    return score


@scorer(metrics=[accuracy()])
def korean_keyword_match(
    keywords_weight: float = 0.5,
) -> Scorer:
    """
    한국어 키워드 일치 Scorer

    정답의 주요 키워드가 응답에 포함되어 있는지 확인합니다.

    Args:
        keywords_weight: 키워드 일치 시 부여할 점수 가중치
    """
    async def score(state: TaskState, target: Target) -> Score:
        model_answer = state.output.completion if state.output else ""
        normalized_answer = normalize_korean_text(model_answer)
        
        target_text = target.text if target.text else ""
        
        # 정답에서 키워드 추출 (2글자 이상의 단어)
        keywords = [
            word for word in target_text.split()
            if len(word) >= 2
        ]
        
        if not keywords:
            keywords = [target_text]
        
        # 키워드 일치 확인
        matched_keywords = [
            kw for kw in keywords
            if normalize_korean_text(kw) in normalized_answer
        ]
        
        match_ratio = len(matched_keywords) / len(keywords) if keywords else 0
        
        if match_ratio >= 0.8:
            return Score(
                value=CORRECT,
                answer=model_answer,
                explanation=f"키워드 일치율: {match_ratio:.0%} ({len(matched_keywords)}/{len(keywords)})",
            )
        elif match_ratio >= keywords_weight:
            return Score(
                value=PARTIAL,
                answer=model_answer,
                explanation=f"부분 키워드 일치: {match_ratio:.0%}",
            )
        
        return Score(
            value=INCORRECT,
            answer=model_answer,
            explanation=f"키워드 일치율 낮음: {match_ratio:.0%}",
        )

    return score


@scorer(metrics=[accuracy()])
def korean_model_grader(
    grader_model: str | None = None,
) -> Scorer:
    """
    한국어 모델 기반 Scorer

    LLM을 사용하여 응답의 정확성을 평가합니다.

    Args:
        grader_model: 채점에 사용할 모델 (None이면 평가 모델 사용)
    """
    from inspect_ai.model import get_model
    
    GRADING_PROMPT = """당신은 한국어 답변을 채점하는 전문가입니다.

주어진 질문에 대해 정답과 모델의 답변을 비교하여 평가해주세요.

질문: {question}
정답: {target}
모델 답변: {answer}

평가 기준:
- 의미적으로 정답과 일치하면 CORRECT
- 부분적으로 맞으면 PARTIAL
- 완전히 틀리면 INCORRECT

답변 형식:
[평가]: CORRECT/PARTIAL/INCORRECT
[설명]: (평가 이유)"""

    async def score(state: TaskState, target: Target) -> Score:
        model_answer = state.output.completion if state.output else ""
        
        # 질문 추출 (첫 번째 사용자 메시지)
        question = ""
        for msg in state.messages:
            if hasattr(msg, 'role') and msg.role == "user":
                question = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        
        prompt = GRADING_PROMPT.format(
            question=question,
            target=target.text or "",
            answer=model_answer,
        )
        
        # 채점 모델 호출
        model = get_model(grader_model) if grader_model else get_model()
        response = await model.generate(prompt)
        grading_result = response.completion
        
        # 결과 파싱
        if "CORRECT" in grading_result.upper():
            if "INCORRECT" in grading_result.upper():
                value = INCORRECT
            else:
                value = CORRECT
        elif "PARTIAL" in grading_result.upper():
            value = PARTIAL
        else:
            value = INCORRECT
        
        return Score(
            value=value,
            answer=model_answer,
            explanation=grading_result,
        )

    return score

