"""
MT-Bench Scorer - LLM Judge 기반 평가

각 턴별로 LLM Judge가 1-10점 평가를 수행합니다.
최종 점수는 두 턴의 평균입니다.

한국어 점수 파싱: 평가: [[N]] 또는 [[N]] 형식
"""

import re
from inspect_ai.scorer import (
    Scorer,
    Score,
    Target,
    scorer,
    metric,
    mean,
    Metric,
    SampleScore,
)
from inspect_ai.solver import TaskState
from inspect_ai.model import get_model, ChatMessageSystem, ChatMessageUser, GenerateConfig


def _extract_score(text: str) -> float | None:
    """응답에서 점수 추출 (1-10)
    
    한국어 MT-Bench 형식:
    - 평가: [[N]]
    - [[N]]
    """
    # 한국어 형식: 평가: [[N]]
    m = re.search(r"평가\s*:\s*\[\[(\d+(?:\.\d+)?)\]\]", text)
    if m:
        score = float(m.group(1))
        if 1 <= score <= 10:
            return score
    
    # 기본 형식: [[N]]
    m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", text)
    if m:
        score = float(m.group(1))
        if 1 <= score <= 10:
            return score
    
    return None


def _build_single_turn_prompt(
    template: str,
    question: str,
    answer: str,
    ref_answer: str = "",
) -> str:
    """Single turn judge 프롬프트 생성"""
    prompt = template
    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{answer}", answer)
    prompt = prompt.replace("{ref_answer_1}", ref_answer)
    return prompt


def _build_multi_turn_prompt(
    template: str,
    question_1: str,
    answer_1: str,
    question_2: str,
    answer_2: str,
    ref_answer_1: str = "",
    ref_answer_2: str = "",
) -> str:
    """Multi turn judge 프롬프트 생성"""
    prompt = template
    prompt = prompt.replace("{question_1}", question_1)
    prompt = prompt.replace("{answer_1}", answer_1)
    prompt = prompt.replace("{question_2}", question_2)
    prompt = prompt.replace("{answer_2}", answer_2)
    prompt = prompt.replace("{ref_answer_1}", ref_answer_1)
    prompt = prompt.replace("{ref_answer_2}", ref_answer_2)
    return prompt


# =============================================================================
# 카테고리별 메트릭
# =============================================================================

@metric
def writing_score() -> Metric:
    """Writing 카테고리 평균 점수"""
    def metric_fn(scores: list[SampleScore]) -> float:
        category_scores = [
            s.score.metadata.get("avg_score", 0.0) / 10.0
            for s in scores
            if s.score.metadata and s.score.metadata.get("category") == "writing"
        ]
        return sum(category_scores) / len(category_scores) if category_scores else 0.0
    return metric_fn


@metric
def roleplay_score() -> Metric:
    """Roleplay 카테고리 평균 점수"""
    def metric_fn(scores: list[SampleScore]) -> float:
        category_scores = [
            s.score.metadata.get("avg_score", 0.0) / 10.0
            for s in scores
            if s.score.metadata and s.score.metadata.get("category") == "roleplay"
        ]
        return sum(category_scores) / len(category_scores) if category_scores else 0.0
    return metric_fn


@metric
def reasoning_score() -> Metric:
    """Reasoning 카테고리 평균 점수"""
    def metric_fn(scores: list[SampleScore]) -> float:
        category_scores = [
            s.score.metadata.get("avg_score", 0.0) / 10.0
            for s in scores
            if s.score.metadata and s.score.metadata.get("category") == "reasoning"
        ]
        return sum(category_scores) / len(category_scores) if category_scores else 0.0
    return metric_fn


@metric
def math_score() -> Metric:
    """Math 카테고리 평균 점수"""
    def metric_fn(scores: list[SampleScore]) -> float:
        category_scores = [
            s.score.metadata.get("avg_score", 0.0) / 10.0
            for s in scores
            if s.score.metadata and s.score.metadata.get("category") == "math"
        ]
        return sum(category_scores) / len(category_scores) if category_scores else 0.0
    return metric_fn


@metric
def coding_score() -> Metric:
    """Coding 카테고리 평균 점수"""
    def metric_fn(scores: list[SampleScore]) -> float:
        category_scores = [
            s.score.metadata.get("avg_score", 0.0) / 10.0
            for s in scores
            if s.score.metadata and s.score.metadata.get("category") == "coding"
        ]
        return sum(category_scores) / len(category_scores) if category_scores else 0.0
    return metric_fn


@metric
def extraction_score() -> Metric:
    """Extraction 카테고리 평균 점수"""
    def metric_fn(scores: list[SampleScore]) -> float:
        category_scores = [
            s.score.metadata.get("avg_score", 0.0) / 10.0
            for s in scores
            if s.score.metadata and s.score.metadata.get("category") == "extraction"
        ]
        return sum(category_scores) / len(category_scores) if category_scores else 0.0
    return metric_fn


@metric
def stem_score() -> Metric:
    """STEM 카테고리 평균 점수"""
    def metric_fn(scores: list[SampleScore]) -> float:
        category_scores = [
            s.score.metadata.get("avg_score", 0.0) / 10.0
            for s in scores
            if s.score.metadata and s.score.metadata.get("category") == "stem"
        ]
        return sum(category_scores) / len(category_scores) if category_scores else 0.0
    return metric_fn


@metric
def humanities_score() -> Metric:
    """Humanities 카테고리 평균 점수"""
    def metric_fn(scores: list[SampleScore]) -> float:
        category_scores = [
            s.score.metadata.get("avg_score", 0.0) / 10.0
            for s in scores
            if s.score.metadata and s.score.metadata.get("category") == "humanities"
        ]
        return sum(category_scores) / len(category_scores) if category_scores else 0.0
    return metric_fn


# =============================================================================
# MT-Bench Scorer
# =============================================================================

@scorer(metrics=[
    mean(),
    writing_score(),
    roleplay_score(),
    reasoning_score(),
    math_score(),
    coding_score(),
    extraction_score(),
    stem_score(),
    humanities_score(),
])
def mtbench_scorer(
    judge_model: str = "openai/gpt-4o-mini",
) -> Scorer:
    """
    MT-Bench LLM Judge Scorer
    
    Args:
        judge_model: 평가에 사용할 LLM 모델
    
    평가 기준:
    - Turn 1: 첫 번째 응답의 품질 (1-10점)
    - Turn 2: 두 번째 응답의 품질 (1-10점)
    - 최종 점수: 두 턴의 평균 (0-1 정규화)
    """
    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        
        # 필요한 데이터 추출
        turn1 = metadata.get("turn1", "")
        turn2 = metadata.get("turn2", "")
        response_turn1 = metadata.get("response_turn1", "")
        response_turn2 = metadata.get("response_turn2", "")
        reference_turn1 = metadata.get("reference_turn1", "")
        reference_turn2 = metadata.get("reference_turn2", "")
        category = metadata.get("category", "general")
        
        # Judge 프롬프트 정보
        judge_single = metadata.get("judge_prompt_single", {})
        judge_multi = metadata.get("judge_prompt_multi", {})
        
        system_prompt_single = judge_single.get("system_prompt", "당신은 유익한 조수입니다.")
        system_prompt_multi = judge_multi.get("system_prompt", system_prompt_single)
        prompt_template_single = judge_single.get("prompt_template", "")
        prompt_template_multi = judge_multi.get("prompt_template", "")
        
        # Judge 모델 초기화
        judge = get_model(
            judge_model,
            config=GenerateConfig(temperature=0.0, max_tokens=1024),
        )
        
        # Turn 2만 평가 (Multi-turn 맥락에서 최종 평가)
        # Turn 2 프롬프트에 Turn 1 대화 맥락이 포함됨
        turn2_score = None
        judge_text = ""
        
        if response_turn2 and turn2 and prompt_template_multi:
            judge_prompt = _build_multi_turn_prompt(
                template=prompt_template_multi,
                question_1=turn1,
                answer_1=response_turn1,
                question_2=turn2,
                answer_2=response_turn2,
                ref_answer_1=reference_turn1,
                ref_answer_2=reference_turn2,
            )
            full_prompt = f"{system_prompt_multi}\n\n{judge_prompt}"
            
            judge_response = await judge.generate([
                ChatMessageUser(content=full_prompt),
            ])
            judge_text = judge_response.completion
            turn2_score = _extract_score(judge_text)
        
        # 결과 처리
        if turn2_score:
            final_score = turn2_score
            explanation = f"Turn2: {turn2_score}/10"
        else:
            final_score = 0.0
            explanation = "Turn2: 점수 추출 실패"
        
        normalized_score = final_score / 10.0  # 0-1 정규화
        
        return Score(
            value=normalized_score,
            answer=f"Turn1: {response_turn1[:100]}... | Turn2: {response_turn2[:100] if response_turn2 else 'N/A'}",
            explanation=explanation,
            metadata={
                "category": category,
                "avg_score": final_score,
                "turn2_score": turn2_score,
                "judge_response": judge_text[:500] if judge_text else "",
            },
        )
    
    return score
