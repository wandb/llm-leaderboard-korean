from typing import List, Dict, Any, Optional
from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.logging import get_logger
import logging
from tqdm import tqdm

logger = get_logger(name="log_likelihood_eval", level=logging.INFO)

@register_evaluator("log_likelihood")
class LogProbEvaluator(BaseEvaluator):
    """
    MCQA Log Probability Evaluator.

    This evaluator expects each sample to include an "options" field (a list of candidate options)
    and a "logits" field containing "option_log_probs" (a list of log probability scores for each option).

    Instead of solely relying on the stored "prediction" field, it recalculates the predicted option
    based on the log probability scores (if available) and then compares it with the ground truth reference
    to compute accuracy.
    """
    name = "log_prob_mcqa"
    requires_logits = True

    def evaluate_predictions(self, subsets: Optional[List[str]], samples: List[Dict[str, Any]]) -> Dict[str, float]:
        correct = 0
        total = 0
        # 요청된 subsets 기준 초기화
        stats: Dict[str, Dict[str, int]] = {}
        if subsets:
            for subset in subsets:
                stats[subset] = {"total": 0, "correct": 0}
        for sample in samples:
            subset_name = sample.get("_subset_name")
            options = sample.get("options", [])
            reference = sample.get("reference", "").strip()
            logits_info = sample.get("logits", {})
            option_log_probs = logits_info.get("option_log_probs", None)
            # If option log probabilities are available and valid, use them to determine the prediction.
            if option_log_probs is not None and options and len(option_log_probs) == len(options):
                best_idx = option_log_probs.index(max(option_log_probs))
                predicted_option = options[best_idx].strip()
            else:
                # Fallback: use the stored prediction.
                predicted_option = sample.get("prediction", "").strip()

            if predicted_option == reference:
                correct += 1
                if subset_name:
                    stats[subset_name]["correct"] += 1
            total += 1
            if subset_name:
                stats[subset_name]["total"] += 1
        # 전체 메트릭 유지 (accuracy/AVG)
        overall_acc = correct / total if total > 0 else 0.0
        metrics: Dict[str, float] = {"AVG": overall_acc, "accuracy": overall_acc}
        if subsets:
            for sname, st in stats.items():
                s_total = st["total"]
                s_correct = st["correct"]
                s_acc = (s_correct / s_total) if s_total > 0 else 0.0
                metrics[f"{sname}/AVG"] = s_acc
        return metrics