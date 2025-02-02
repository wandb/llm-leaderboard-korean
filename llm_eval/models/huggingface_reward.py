from typing import List, Dict, Any, Optional, Union
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_eval.models.base import BaseRewardModel
from . import register_model
from llm_eval.utils.logging import get_logger

logger = get_logger(name="huggingface_reward", level=logging.INFO)

@register_model("huggingface_reward")
class HuggingFaceReward(BaseRewardModel):
    """
    HuggingFaceReward computes a reward for a given sample by calculating the
    conditional log likelihood (average log probability) of the generated answer.

    For example, for a sample like:
      {
         "input": "Problem: ...\nAnswer:",
         "prediction": " generated answer text..."
      }

    The reward is computed as the average log probability of the tokens in the generated
    answer (i.e., the part after the prompt), as predicted by the model.
    """
    
    def __init__(self, model_name_or_path: str, device: str = "cpu", **kwargs):
        """
        Args:
            model_name_or_path (str): HuggingFace model identifier or local path (e.g., "gpt2", "EleutherAI/gpt-neox-20b", etc.)
            device (str): The device to run the model on ("cpu", "cuda", etc.)
            **kwargs: Additional arguments as needed.
        """
        super().__init__(**kwargs)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.eval()
        if device != "cpu":
            self.model.to(device)
    
    def score_batch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For each sample, combine the "input" and "prediction" fields to form the full text,
        and compute the conditional log likelihood for the tokens corresponding to the generated answer.
        The reward is computed as the average log probability over these tokens.

        The computation steps are as follows:
          1. For each sample, construct the full text as input + prediction.
          2. Calculate the number of tokens in the prompt (input) separately.
          3. Tokenize the full text in a batch.
          4. Pass the tokenized inputs through the model to obtain logits.
          5. Using the shifted logits, compute the log probabilities for each token,
             and then compute the average log probability over the generated part (after the prompt).
          6. Add the computed reward to each sample in the "reward" field.

        Args:
            inputs: List of samples, where each sample must contain at least the "input" and "prediction" fields.
            
        Returns:
            The list of input samples with an additional "reward" field added to each sample.
        """
        if not inputs:
            return inputs
        
        batch_size = len(inputs)
        full_texts = []       # Combined texts: input + prediction
        prompt_lengths = []   # Number of tokens in the prompt for each sample
        actual_lengths = []   # Actual token count (before padding) for the full text
        
        for sample in inputs:
            prompt = sample.get("input", "")
            prediction = sample.get("prediction", "")
            full_text = prompt + prediction
            full_texts.append(full_text)
            
            # Get the token count for the prompt using the tokenizer.
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_lengths.append(len(prompt_ids))
            
            # Get the token count for the full text (before padding).
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            actual_lengths.append(len(full_ids))
        
        # Batch tokenize the full texts with padding.
        encoded = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
        
        # Compute the log probabilities for the tokens after the prompt for each sample.
        rewards = []
        for i in range(batch_size):
            p_len = prompt_lengths[i]
            seq_len = actual_lengths[i]  # Actual sequence length (without padding)
            
            # If the full text length is not greater than the prompt length or the prompt is empty, set reward to 0.
            if seq_len <= p_len or p_len == 0:
                rewards.append(0.0)
                continue
            
            # For language models, the probability of token j is derived from logits at position j-1.
            # That is, for tokens full_text[p_len:], the corresponding logits are from logits[i, p_len-1: seq_len-1].
            logits_i = logits[i, p_len - 1: seq_len - 1, :]  # shape: (seq_len - p_len, vocab_size)
            target_ids = input_ids[i, p_len:seq_len]           # True token ids for the generated part
            
            # Compute log probabilities by applying log_softmax to the logits.
            log_probs = F.log_softmax(logits_i, dim=-1)         # shape: (seq_len - p_len, vocab_size)
            # Gather the log probability corresponding to the target token at each position.
            token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            # Compute the average log probability.
            avg_log_prob = token_log_probs.mean().item()
            rewards.append(avg_log_prob)
        
        # Add the computed reward to each sample.
        for i, sample in enumerate(inputs):
            sample["reward"] = rewards[i]
        
        return inputs
