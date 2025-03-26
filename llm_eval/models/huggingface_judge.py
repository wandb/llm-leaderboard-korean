import logging
from typing import List, Dict, Any, Optional, Union
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseJudge
from . import register_model
from tqdm import tqdm

from llm_eval.utils.logging import get_logger

logger = get_logger(name="huggingface_judge", level=logging.INFO)

@register_model("huggingface_judge")
class HuggingFaceJudge(BaseJudge):
    """
    A 'judge' model implementation based on HuggingFace Transformers.
    
    This class is designed for scenarios where an LLM is used to evaluate
    (or 'judge') existing answers by generating a short rating, correctness, or
    preference output. The 'judge_batch' method:
      - Expects each sample to have "input" containing the full judge prompt
        (rubric, reference, model output, etc.).
      - Generates a short response indicating a score, correctness, or comparison result.
      - Stores the raw generation in 'sample["prediction"]'.

    The higher-level evaluator (e.g., LLMJudgeEvaluator) parses these outputs
    (e.g., "[[score: 4.5]]") using a suitable parser (RubricScoreParser, etc.)
    to extract structured metrics.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        batch_size: int = 8,
        **kwargs
    ):
        """
        Args:
            model_name_or_path (str): A HuggingFace Hub model identifier or local path.
            device (str): Device to run on ('cpu', 'cuda', etc.).
            max_new_tokens (int): The maximum number of tokens to generate per sample.
            temperature (float): Sampling temperature for generation.
            top_p (float): Nucleus sampling probability.
            do_sample (bool): Whether to perform sampling (True) or greedy decoding (False).
            batch_size (int): Number of samples processed at once in judge_batch().
            **kwargs: Additional parameters (ignored or for extension).
        """
        super().__init__(**kwargs)
        logger.info(f"[HuggingFaceJudge] Initializing with model: {model_name_or_path}")

        # Update auth token parameter if it exists
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")

        # Load tokenizer with correct padding settings for decoder-only models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            truncation_side="left",
            **kwargs
        )
        
        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Ensure model knows about pad token
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Log tokenizer settings for debugging
        logger.info(f"Tokenizer settings - Pad token: {self.tokenizer.pad_token}, "
                    f"Pad token ID: {self.tokenizer.pad_token_id}, "
                    f"Padding side: {self.tokenizer.padding_side}")
        
        self.model.eval()

        if device != "cpu":
            self.model.to(device)
        self.device = device

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.batch_size = batch_size

    def judge_batch(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Processes a batch of judge prompts and generates a short response for each.
        """
        if not inputs:
            return inputs

        results: List[Dict[str, Any]] = []
        logger.info("starting llm-as-judge")
        for start_idx in tqdm(range(0, len(inputs), self.batch_size)):
            batch = inputs[start_idx : start_idx + self.batch_size]

            # Extract only the essential parts from the input
            prompts = []
            for item in batch:
                input_text = item["input"]
                # Add a clear instruction to generate evaluation
                prompt = f"{input_text}\n\nPlease provide your evaluation in the following format:\nFeedback: [your detailed feedback] [RESULT] [score between 1-5]"
                prompts.append(prompt)

            # Tokenize
            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            if self.device != "cpu":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**encoded, **gen_kwargs)

            if isinstance(outputs, dict):
                sequences = outputs.get("sequences", None)
                if sequences is None:
                    sequences = outputs
            else:
                sequences = outputs

            for i, seq_ids in enumerate(sequences):
                decoded = self.tokenizer.decode(seq_ids, skip_special_tokens=True)
                
                # Extract feedback and score using regex
                feedback_pattern = r"Feedback:(.*?)\[RESULT\]\s*(\d+(?:\.\d+)?)"
                result_only_pattern = r"\[RESULT\]\s*(\d+(?:\.\d+)?)"

                match = re.search(feedback_pattern, decoded, re.DOTALL)
                if match:
                    feedback = match.group(1).strip()
                    score = float(match.group(2))
                    # Format the output as expected
                    batch[i]["prediction"] = f"Feedback: {feedback} [RESULT] {score}"
                    batch[i]["judge_score"] = score
                    batch[i]["language_penalizer"] = 1.0
                    # Add fields to match OpenAI output format
                    batch[i]["reference"] = batch[i].get("reference", "")
                    batch[i]["_subset_name"] = "Pointwise"
                    batch[i]["judge_type"] = "rubric_and_response"
                else:
                    result_match = re.search(result_only_pattern, decoded)
                    if result_match:
                        score = float(result_match.group(1))
                        batch[i]["prediction"] = f"Feedback: {decoded} [RESULT] {score}"
                        batch[i]["judge_score"] = score
                        batch[i]["language_penalizer"] = 1.0
                        # Add fields to match OpenAI output format
                        batch[i]["reference"] = batch[i].get("reference", "")
                        batch[i]["_subset_name"] = "Pointwise"
                        batch[i]["judge_type"] = "rubric_and_response"
                    else:
                        error_msg = f"Failed to extract score from model output: {decoded}"
                        logger.error(error_msg)
                        batch[i]["prediction"] = decoded
                        batch[i]["judge_score"] = "error"
                        batch[i]["language_penalizer"] = 1.0
                        # Add fields to match OpenAI output format
                        batch[i]["reference"] = batch[i].get("reference", "")
                        batch[i]["_subset_name"] = "Pointwise"
                        batch[i]["judge_type"] = "rubric_and_response"

            results.extend(batch)

        return results