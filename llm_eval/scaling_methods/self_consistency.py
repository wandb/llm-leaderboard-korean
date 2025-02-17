from typing import List, Dict, Any, Optional, Callable
from collections import Counter
import re
from llm_eval.models.base import BaseModel
from .base import BaseScalingMethod
from . import register_scaling_method
from llm_eval.utils.prompt_template import extract_final_answer # default parser
from llm_eval.utils.logging import get_logger
import logging
from tqdm import tqdm

logger = get_logger(name="runner", level=logging.INFO)

@register_scaling_method("self_consistency")
class SelfConsistencyScalingMethod(BaseScalingMethod):
    """
    Self-Consistency Chain-of-Thought (CoT) scaling method using the default parser.
    
    Args:
        model (BaseModel): LLM model providing generate_batch().
        n_paths (int): Number of reasoning paths (i.e., sampling iterations).
        aggregator_fn (Callable): Function to decide the final answer among multiple candidates.
            Default: majority_voting.
        prompt_cot (str): Additional text to induce CoT reasoning.
        use_default_parser (bool): If True, use the default parser (extract_final_answer);
            otherwise, use raw text processing.
    """

    def __init__(
        self,
        model: BaseModel = None,
        n_paths: int = 5,
        aggregator_fn: Optional[Callable[[List[str]], str]] = None,
        prompt_cot: Optional[str] = None,
        use_default_parser: bool = True,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.n_paths = n_paths
        self.prompt_cot = prompt_cot or "\nLet's think step by step.\n"
        self.aggregator_fn = aggregator_fn if aggregator_fn else self._majority_voting
        self.use_default_parser = use_default_parser

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.model:
            raise ValueError("SelfConsistencyScalingMethod: model is not set.")

        logger.info("starting self-consistency voting")
        for sample in tqdm(data):
            original_prompt = sample["input"]
            prompt = original_prompt + self.prompt_cot

            candidates = []
            for _ in range(self.n_paths):
                outputs = self.model.generate_batch(
                    [{"input": prompt}],
                    return_logits=False,
                    show_progress=False
                )
                raw_text = outputs[0].get("prediction", "")

                # Use the default parser if enabled; otherwise, just trim the raw text.
                if self.use_default_parser:
                    parsed_answer = extract_final_answer(raw_text)
                else:
                    parsed_answer = raw_text.strip()

                candidates.append(parsed_answer)

            # Use the aggregator function to decide on the final answer.
            final_answer = self.aggregator_fn(candidates)
            sample["prediction"] = final_answer

        return data

    def _majority_voting(self, candidates: List[str]) -> str:
        """
        Simple majority voting: choose the answer that appears most frequently.
        """
        if not candidates:
            return ""
        counter = Counter(candidates)
        # most_common(1)[0] returns (answer, frequency)
        return counter.most_common(1)[0][0]
