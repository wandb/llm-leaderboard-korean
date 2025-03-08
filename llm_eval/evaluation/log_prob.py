from typing import List, Dict, Any
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

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        correct = 0
        total = 0
        for sample in samples:
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
            total += 1
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}