from typing import List, Dict, Any
from llm_eval.models.base import BaseModel
from .base import BaseScalingMethod
from . import register_scaling_method
from llm_eval.utils.logging import get_logger
from tqdm import tqdm
import logging

logger = get_logger(name="best_of_n", level=logging.INFO)

@register_scaling_method("best_of_n")
class BestOfN(BaseScalingMethod):
    """
    A scaling method that samples N times for each sample and selects the best answer
    according to a reward or log-prob score.

    Example for batch_size=3, n=3:
      - We have 3 samples in a batch, and we run generate_batch 3 times (N=3).
      - That yields 9 candidate answers (3 for each sample).
      - We store the candidate texts in candidates_texts and their scores in candidates_scores.
      - For sample #0, the relevant indices are (0, 3, 6); 
        for sample #1, they are (1, 4, 7); 
        for sample #2, they are (2, 5, 8).
      - We select the candidate with the highest score for each sample.
    """

    def __init__(self, model: BaseModel = None, n: int = 5, batch_size: int = 1, **kwargs):
        """
        Args:
            model (BaseModel): The model backend (could be MultiModel).
            n (int): Number of sampling iterations per sample.
            batch_size (int): Number of samples to process in one batch.
            kwargs: Additional parameters that may be passed from BaseScalingMethod.
        """
        super().__init__(model=model, **kwargs)
        self.n = n
        self.batch_size = batch_size

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for Best-of-N scaling.

        Args:
            data: A list of samples, each a dict with at least {"input":..., "reference":...}.

        Returns:
            A list of the same structure with "prediction" updated to the selected best candidate
            for each sample.
        """
        if self.model is None:
            raise ValueError("BestOfN scaling requires a 'model' instance.")
        
        final_outputs = []
        total_samples = len(data)
        logger.info("Starting Best-Of-N Scaling")
        # Process data in batches
        for batch_start in tqdm(range(0, total_samples, self.batch_size)):
            # 1) Prepare batch
            batch_samples = []
            for local_idx in range(self.batch_size):
                idx = batch_start + local_idx
                if idx < total_samples:
                    batch_samples.append(data[idx])

            num_batch_samples = len(batch_samples)
            if num_batch_samples == 0:
                continue

            # 2) Accumulate candidates over N iterations
            candidates_texts: List[str] = []
            candidates_scores: List[float] = []

            # For each iteration, generate 1 candidate per sample
            for _ in range(self.n):
                outputs = self.model.generate_batch(
                    batch_samples,
                    batch_size=self.batch_size,
                    show_progress=False
                )

                # Collect candidate texts
                for out_item in outputs:
                    candidates_texts.append(out_item.get("prediction", ""))

                # Check if the model can provide a reward score
                if hasattr(self.model, "score_batch"):
                    # Using a model that provides a `score_batch` method (e.g., MultiModel with a reward model)
                    scored_outputs = self.model.score_batch(outputs)
                    for s_out in scored_outputs:
                        # Expecting s_out to have "reward" field
                        score = float(s_out.get("reward", 0.0))
                        candidates_scores.append(score)
                else:
                    # Fallback: use log-prob from "logits" if available
                    for out_item in outputs:
                        logit_info = out_item.get("logits", {})
                        sum_log_prob = float(logit_info.get("sum_log_prob", 0.0))
                        candidates_scores.append(sum_log_prob)

            # 3) For each sample in the batch, pick the best candidate
            # total candidates = n * num_batch_samples
            # The indices for sample i within [candidates_texts, candidates_scores] are:
            # i, i + num_batch_samples, i + 2*num_batch_samples, ...
            for local_idx, sample_item in enumerate(batch_samples):
                best_score = float("-inf")
                best_text = ""

                for round_idx in range(self.n):
                    c_idx = round_idx * num_batch_samples + local_idx
                    score = candidates_scores[c_idx]
                    if score > best_score:
                        best_score = score
                        best_text = candidates_texts[c_idx]

                sample_item["prediction"] = best_text
                final_outputs.append(sample_item)

        return final_outputs
