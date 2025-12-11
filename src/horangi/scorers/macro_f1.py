"""
Macro F1 scorer for multi-class classification.

Hate speech detection 등 다중 클래스 분류 문제에 사용.
"""

from collections import Counter
from inspect_ai.scorer import (
    Score,
    SampleScore,
    Scorer,
    Target,
    scorer,
    metric,
    Metric,
    accuracy,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import TaskState


def _calculate_f1(tp: int, fp: int, fn: int) -> float:
    """Calculate F1 score from TP, FP, FN."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


@metric
def macro_f1_metric() -> Metric:
    """
    Macro F1 metric - 각 클래스별 F1의 평균.
    
    scorer에서 metadata에 저장한 prediction/target을 사용.
    """
    def metric_fn(scores: list[SampleScore]) -> float:
        # 예측과 정답 수집
        predictions = []
        targets = []
        
        for sample_score in scores:
            metadata = sample_score.score.metadata or {}
            pred = metadata.get("prediction")
            target = metadata.get("target")
            if pred is not None and target is not None:
                predictions.append(pred)
                targets.append(target)
        
        if not predictions:
            return 0.0
        
        # 모든 클래스 수집
        classes = sorted(set(targets) | set(predictions))
        
        # 클래스별 F1 계산
        f1_scores = []
        for cls in classes:
            tp = sum(1 for p, t in zip(predictions, targets) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(predictions, targets) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(predictions, targets) if p != cls and t == cls)
            f1_scores.append(_calculate_f1(tp, fp, fn))
        
        # Macro F1 = 클래스별 F1의 평균
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return metric_fn


@scorer(metrics=[accuracy(), macro_f1_metric()])
def macro_f1() -> Scorer:
    """
    Multi-class classification scorer with Macro F1 and Accuracy.
    
    모델의 선택(A, B, C 등)을 정답과 비교하고,
    전체 결과에 대해 Accuracy와 Macro F1을 계산합니다.
    """
    async def score(state: TaskState, target: Target) -> Score:
        # 모델 답변에서 선택지 추출 (A, B, C 등)
        answer = state.output.completion.strip()
        
        # 첫 번째 대문자 알파벳 찾기
        prediction = None
        for char in answer:
            if char.upper() in "ABCDEFGHIJ":
                prediction = char.upper()
                break
        
        # target도 문자로 변환 (이미 A, B, C 형식이어야 함)
        target_letter = target.text.strip().upper()
        
        correct = prediction == target_letter
        
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=prediction or answer[:50],
            explanation=f"Predicted: {prediction}, Target: {target_letter}",
            metadata={
                "prediction": prediction,
                "target": target_letter,
            }
        )
    
    return score

