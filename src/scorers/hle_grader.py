"""
HLE Grader - Humanity's Last Exam style LLM-based scorer

Features:
- Judge model can be specified separately (default: same as evaluation model)
- Different scoring methods based on answer_type:
  - exactMatch: LLM determines semantic equivalence
  - multipleChoice: Exact matching
"""

import re
from inspect_ai.scorer import (
    Score,
    SampleScore,
    Scorer,
    Target,
    scorer,
    metric,
    Metric,
    accuracy,
    model_graded_qa,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import TaskState


JUDGE_PROMPT = """Based on the precise and clear [correct answer] below, determine whether the following [response] is correct for the [question].

[Question]: {question}

[Response]: {answer}

[Correct Answer]: {criterion}

Focus only on whether there is a meaningful difference between the [correct answer] and [response].
Do not mention background information about the problem or attempts to solve it, and do not argue for an answer different from the [correct answer]. Focus only on whether the two answers match.
Explain in one or two sentences why the [response] is correct or incorrect based on the [correct answer].
Finally, write 'GRADE: C' if correct, or 'GRADE: I' if incorrect.

"""

JUDGE_PROMPT2 = """Determine if the following [response] matches the [correct answer].

[Question]: {question}

[Response]: {answer}

[Correct Answer]: {criterion}

[Instructions]:
- Only determine if there is a semantic difference between [correct answer] and [response].
- It is correct if the meaning is the same even if the expression is different (e.g., "true" = "True" = "correct").
- Explain the reason for your judgment in one or two sentences.
- End with 'GRADE: C' (correct) or 'GRADE: I' (incorrect).
"""


def _is_correct(value) -> bool:
    """Determine if score.value is correct (supports both number/string)"""
    if value == CORRECT or value == "C":
        return True
    if isinstance(value, (int, float)) and value == 1:
        return True
    return False


@metric
def hle_accuracy() -> Metric:
    """HLE accuracy (overall, without answer_type distinction)"""
    def metric_fn(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        correct = sum(1 for s in scores if _is_correct(s.score.value))
        return correct / len(scores)
    return metric_fn


@metric
def hle_exact_match_accuracy() -> Metric:
    """Accuracy for exactMatch type only"""
    def metric_fn(scores: list[SampleScore]) -> float:
        exact_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("answer_type") == "exactMatch"]
        if not exact_scores:
            return 0.0
        correct = sum(1 for s in exact_scores if _is_correct(s.score.value))
        return correct / len(exact_scores)
    return metric_fn


@metric
def hle_multiple_choice_accuracy() -> Metric:
    """Accuracy for multipleChoice type only"""
    def metric_fn(scores: list[SampleScore]) -> float:
        mc_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("answer_type") == "multipleChoice"]
        if not mc_scores:
            return 0.0
        correct = sum(1 for s in mc_scores if _is_correct(s.score.value))
        return correct / len(mc_scores)
    return metric_fn


def _extract_answer(response: str, answer_type: str) -> str:
    """Extract answer from model response"""
    if answer_type == "multipleChoice":
        # Find "Answer: B" or "(B)" format
        match = re.search(r"Answer:\s*([A-Za-z])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"\(([A-Za-z])\)", response)
        if match:
            return match.group(1).upper()
        # Find single alphabet
        match = re.search(r"\b([A-Za-z])\b", response)
        if match:
            return match.group(1).upper()
    else:
        # exactMatch: Find "Exact Answer: ..." format
        match = re.search(r"Exact Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return response.strip()


@scorer(metrics=[
    accuracy(),  # Add default accuracy
    hle_accuracy(),
    hle_exact_match_accuracy(),
    hle_multiple_choice_accuracy(),
])
def hle_grader(judge_model: str | None = None) -> Scorer:
    """
    HLE style scorer.
    
    Args:
        judge_model: Model for scoring (None uses same model as evaluation)
                     e.g., "openai/gpt-4o-mini"
    
    Scoring method:
    - multipleChoice: Exact match (A, B, C, etc.)
    - exactMatch: LLM determines semantic equivalence
    """
    # Create LLM grader
    llm_grader = model_graded_qa(
        template=JUDGE_PROMPT,
        model=judge_model,
        partial_credit=False,
    )
    
    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        answer_type = metadata.get("answer_type", "exactMatch")
        
        response = state.output.completion if state.output else ""
        extracted_answer = _extract_answer(response, answer_type)
        
        # Same as original HLE: LLM scores all questions
        base_score = await llm_grader(state, target)
        
        return Score(
            value=base_score.value if base_score else INCORRECT,
            answer=extracted_answer,
            explanation=base_score.explanation if base_score else "LLM grading failed",
            metadata={
                "answer_type": answer_type,
                "llm_explanation": base_score.explanation if base_score else None,
            },
        )
    
    return score
