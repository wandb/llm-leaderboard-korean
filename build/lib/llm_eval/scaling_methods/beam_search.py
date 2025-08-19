import copy
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseScalingMethod
from . import register_scaling_method
from llm_eval.models.base import BaseModel
from llm_eval.models.multi import MultiModel
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
    Production-level Beam Search scaling method that supports batch processing.
    
    For each batch of samples, an initial set of beams is created (based on beam_size) 
    for each sample. Then, for up to max_tokens iterations, the model generates one token per beam.
    The log probabilities for each token are recorded, and beams are updated accordingly.
    Duplicate beams are pruned, and once a beam meets a termination condition, it is marked as completed.
    Finally, the beam with the highest aggregated score is selected as the final prediction for each sample.
    """

    def __init__(
        self,
        model: Union[BaseModel, MultiModel] = None,
        beam_size: int = 4,
        max_tokens: int = 50,
        agg_strategy: str = "sum",
        batch_size: int = 1,
        **kwargs
    ):
        """
        Args:
            model: An instance of BaseModel providing generate_batch().
            beam_size: Number of beams to maintain per sample.
            max_tokens: Maximum number of tokens to generate per sample.
            agg_strategy: Aggregation strategy to combine scores (e.g., sum, mean, max).
            batch_size: Number of samples to process in one batch.
            kwargs: Additional parameters.
        """
        super().__init__(model=model, **kwargs)
        self.beam_size = beam_size
        self.max_tokens = max_tokens
        self.agg_strategy = agg_strategy
        self.batch_size = batch_size

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
        Performs beam search on each sample in batches and updates the "prediction" field.
        Args:
            data: A list of samples, each a dictionary containing at least {"input": str, "reference": str, ...}.
        Returns:
            The list with updated "prediction" field for each sample.
        """
        if self.model is None:
            raise ValueError("BeamSearch requires a 'model' instance.")

        logger.info("Starting Beam Search scaling method with batch_size=%d.", self.batch_size)
        for batch_start in tqdm(range(0, len(data), self.batch_size), desc="Beam Search", leave=False):
            batch = data[batch_start: batch_start + self.batch_size]
            sample_beams = []
            sample_finished_beams = []
            for sample in batch:
                beams = [Beam(prompt=sample["input"], index=i) for i in range(self.beam_size)]
                sample_beams.append(beams)
                sample_finished_beams.append([])

            for _ in range(self.max_tokens):
                global_active_beams = []
                for s_idx, beams in enumerate(sample_beams):
                    active = [beam for beam in beams if not beam.pruned and not beam.completed]
                    if not active:
                        continue
                    if len(active) < self.beam_size:
                        repeats = (self.beam_size // len(active)) + 1
                        active = (active * repeats)[: self.beam_size]
                        active = [copy.deepcopy(b) for b in active]
                    for beam in active:
                        global_active_beams.append((s_idx, beam))
                if not global_active_beams:
                    break

                batch_inputs = [
                    {"input": beam.prompt + beam.current_text} for (_, beam) in global_active_beams
                ]
                try:
                    generation_outputs = self.model.generate_batch(
                        batch_inputs,
                        return_logits=True,
                        max_new_tokens=1,
                        show_progress = False,
                    )
                except Exception as e:
                    logger.error("Error during model generation.", exc_info=True)
                    raise e

                for (s_idx, beam), out in zip(global_active_beams, generation_outputs):
                    token_text = out.get("prediction", "")
                    beam.current_text += token_text
                    beam.completion_tokens += 1
                    if "logits" in out and out["logits"] is not None:
                        token_log_probs = out["logits"].get("token_log_probs", [])
                        if token_log_probs:
                            beam.score_history.append(token_log_probs[0])
                    if token_text.strip() == "" or beam.completion_tokens >= self.max_tokens:
                        beam.completed = True
                        sample_finished_beams[s_idx].append(beam)

                new_sample_beams = []
                for s_idx in range(len(batch)):
                    updated_beams = [beam for (idx, beam) in global_active_beams if idx == s_idx and not beam.pruned]
                    unique = {}
                    for beam in updated_beams:
                        if beam.current_text not in unique:
                            unique[beam.current_text] = beam
                        else:
                            beam.pruned = True
                    unique_beams = list(unique.values())
                    sorted_beams = sorted(unique_beams, key=lambda b: self._aggregate_scores(b.score_history), reverse=True)
                    new_sample_beams.append(sorted_beams[: self.beam_size])
                sample_beams = new_sample_beams

            for s_idx, sample in enumerate(batch):
                beams = sample_beams[s_idx]
                finished = sample_finished_beams[s_idx]
                if finished:
                    best_beam = max(finished, key=lambda b: b.aggregate_score(self.agg_fn))
                elif beams:
                    best_beam = max(beams, key=lambda b: b.aggregate_score(self.agg_fn))
                else:
                    best_beam = Beam(prompt=sample["input"], index=0)
                sample["prediction"] = best_beam.current_text

        logger.info("Beam Search scaling completed.")
        return data
