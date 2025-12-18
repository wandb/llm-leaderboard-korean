"""
Macro F1 scorer for multi-class classification.

Used for multi-class classification problems such as hate speech detection.
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
    Macro F1 metric - average of F1 scores for each class.
    
    Uses prediction/target stored in metadata from scorer.
    """
    def metric_fn(scores: list[SampleScore]) -> float:
        # Collect predictions and targets
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
        
        # Collect all classes
        classes = sorted(set(targets) | set(predictions))
        
        # Calculate F1 for each class
        f1_scores = []
        for cls in classes:
            tp = sum(1 for p, t in zip(predictions, targets) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(predictions, targets) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(predictions, targets) if p != cls and t == cls)
            f1_scores.append(_calculate_f1(tp, fp, fn))
        
        # Macro F1 = average of F1 scores for each class
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return metric_fn


@scorer(metrics=[accuracy(), macro_f1_metric()])
def macro_f1() -> Scorer:
    """
    Multi-class classification scorer with Macro F1 and Accuracy.
    
    Compares the model's selection (A, B, C, etc.) with the correct answer,
    and calculates Accuracy and Macro F1 for the overall results.
    """
    async def score(state: TaskState, target: Target) -> Score:
        # Extract selection from model answer (A, B, C, etc.)
        answer = state.output.completion.strip()
        
        # Find the first uppercase letter
        prediction = None
        for char in answer:
            if char.upper() in "ABCDEFGHIJ":
                prediction = char.upper()
                break
        
        # Convert target to letter (should already be in A, B, C format)
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
