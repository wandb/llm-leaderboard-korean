import copy
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseScalingMethod
from . import register_scaling_method
from llm_eval.models.base import BaseModel
from llm_eval.utils.logging import get_logger

logger = get_logger(name="beam_search", level=logging.INFO)


@dataclass
class Beam:
    """
    Data class representing a single beam candidate.
    Attributes:
        prompt: The original prompt (e.g., the question).
        index: Beam index identifier.
        current_text: The generated text accumulated so far.
        completed: Flag indicating if generation is finished (e.g., EOS encountered or max_tokens reached).
        pruned: Flag indicating if the beam has been pruned due to low score or duplicate text.
        score_history: List of log probability scores (or any score) recorded at each generation step.
        completion_tokens: Number of tokens generated.
    """
    prompt: str
    index: int
    current_text: str = ""
    completed: bool = False
    pruned: bool = False
    score_history: List[float] = field(default_factory=list)
    completion_tokens: int = 0

    def aggregate_score(self, agg_fn: Callable[[List[float]], float]) -> float:
        """
        Aggregate the recorded scores using the provided aggregation function.
        Args:
            agg_fn: A callable function such as sum, mean, or max.
        Returns:
            A single float representing the aggregated score.
        """
        if not self.score_history:
            return 0.0
        return agg_fn(self.score_history)


@register_scaling_method("beam_search")
class BeamSearch(BaseScalingMethod):
    """
    Production-level Beam Search scaling method.
    
    For each sample, an initial set of beams is created (based on beam_size). 
    Then, for up to max_tokens iterations, the model generates one token per beam.
    The log probabilities (if available) for each token are recorded, and beams are 
    updated accordingly. Duplicate beams are pruned, and once a beam meets a termination
    condition (empty token generated or reaching max_tokens), it is marked as completed.
    Finally, the beam with the highest aggregated score is selected as the final prediction.
    """

    def __init__(
        self,
        model: BaseModel = None,
        beam_size: int = 4,
        max_tokens: int = 50,
        agg_strategy: str = "sum",  # Options: "sum", "mean", "max"
        **kwargs
    ):
        """
        Args:
            model: An instance of BaseModel providing generate_batch().
            beam_size: Number of beams to maintain per sample.
            max_tokens: Maximum number of tokens to generate per sample.
            agg_strategy: Aggregation strategy to combine scores (e.g., sum, mean, max).
            kwargs: Additional parameters.
        """
        super().__init__(model=model, **kwargs)
        self.beam_size = beam_size
        self.max_tokens = max_tokens
        self.agg_strategy = agg_strategy

        # Determine aggregation function based on strategy.
        if agg_strategy == "sum":
            self.agg_fn = sum
        elif agg_strategy == "max":
            self.agg_fn = max
        elif agg_strategy == "mean":
            self.agg_fn = lambda scores: sum(scores) / len(scores)
        else:
            raise ValueError(f"Unknown aggregation strategy: {agg_strategy}")

    def _aggregate_scores(self, scores: List[float]) -> float:
        """Aggregate the list of scores using the chosen aggregation function."""
        return self.agg_fn(scores) if scores else 0.0

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Performs beam search on each sample and updates the "prediction" field.
        Args:
            data: A list of samples, each a dictionary containing at least {"input": str, "reference": str, ...}.
        Returns:
            The list with updated "prediction" field for each sample.
        """
        if self.model is None:
            raise ValueError("BeamSearch requires a 'model' instance.")

        logger.info("Starting Beam Search scaling method.")
        for sample in tqdm(data, desc="Beam Search", leave=False):
            original_prompt = sample["input"]
            # Create initial beams with an empty generated text and zero score.
            beams = [Beam(prompt=original_prompt, index=i) for i in range(self.beam_size)]
            finished_beams = []

            # Generate tokens until reaching max_tokens or all beams are completed.
            for _ in range(self.max_tokens):
                # Select active beams that are neither pruned nor completed.
                active_beams = [beam for beam in beams if not beam.pruned and not beam.completed]
                if not active_beams:
                    break

                # If the number of active beams is less than beam_size, duplicate beams to maintain count.
                if len(active_beams) < self.beam_size:
                    repeats = (self.beam_size // len(active_beams)) + 1
                    extended = (active_beams * repeats)[: self.beam_size]
                    active_beams = [copy.deepcopy(b) for b in extended]

                # Prepare a batch of inputs: for each active beam, combine prompt with current_text.
                batch_inputs = [{"input": beam.prompt + beam.current_text} for beam in active_beams]

                # Generate one token for each beam using the model.
                try:
                    generation_outputs = self.model.generate_batch(
                        batch_inputs,
                        return_logits=True,
                        max_new_tokens=1  # Generate one token per call.
                    )
                except Exception as e:
                    logger.error("Error during model generation.", exc_info=True)
                    raise e

                # Update each active beam with the generated token and its log probability.
                for beam, out in zip(active_beams, generation_outputs):
                    token_text = out.get("prediction", "")
                    beam.current_text += token_text
                    beam.completion_tokens += 1

                    # If logits are provided, compute the log probability for the generated token.
                    if "logits" in out and out["logits"] is not None:
                        token_log_probs = out["logits"].get("token_log_probs", [])
                        if token_log_probs:
                            beam.score_history.append(token_log_probs[0])
                    # Check termination conditions: if token is empty or max tokens reached.
                    if token_text.strip() == "" or beam.completion_tokens >= self.max_tokens:
                        beam.completed = True
                        finished_beams.append(beam)

                # Remove duplicate beams based on current_text if filter is enabled.
                unique_beams = {}
                for beam in active_beams:
                    if beam.current_text not in unique_beams:
                        unique_beams[beam.current_text] = beam
                    else:
                        beam.pruned = True
                active_beams = [beam for beam in active_beams if not beam.pruned]

                # Sort active beams by aggregated score and keep only the top beam_size beams.
                sorted_beams = sorted(
                    active_beams,
                    key=lambda b: self._aggregate_scores(b.score_history),
                    reverse=True
                )
                beams = sorted_beams[: self.beam_size]

            # After iterations, select the best candidate from finished beams or the remaining active beams.
            if finished_beams:
                best_beam = max(finished_beams, key=lambda b: b.aggregate_score(self.agg_fn))
            else:
                best_beam = max(beams, key=lambda b: b.aggregate_score(self.agg_fn))
            sample["prediction"] = best_beam.current_text

        logger.info("Beam Search scaling completed.")
        return data
