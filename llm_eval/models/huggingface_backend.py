import logging
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger
from llm_eval.utils.prompt_template import default_cot_parser

logger = get_logger(name="huggingface", level=logging.INFO)

@register_model("huggingface")
class HuggingFaceModel(BaseModel):
    """
    A backend model class that uses HuggingFace Transformers.

    Main points:
      - If `return_logits=True`, we call `model.generate(..., return_dict_in_generate=True, output_scores=True)`
        so that "scores" (logits per step) are included in the output, allowing us to compute log probabilities.
      - If CoT is enabled (`cot=True`), the `cot_trigger` is appended to the prompt and the generated text
        can be parsed using `cot_parser` to separate the chain-of-thought from the final answer.
      - MCQA mode: If an input item contains an "options" field (list), each option's log likelihood is computed,
        and the option with the highest log probability is selected as the final prediction.

    Args:
        model_name_or_path (str): HF Hub model ID or local path.
        device (str): 'cpu', 'cuda', 'cuda:0', etc.
        max_new_tokens (int): Maximum new tokens to generate in one call.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        do_sample (bool): If True, sampling mode; if False, greedy generation.
        batch_size (int): Batch size for generation.
        cot (bool): Whether to use chain-of-thought prompting.
        cot_trigger (str|None): Optional CoT trigger string. If None, CoT is not triggered.
        cot_parser (callable|None): A function that takes a generated text string and returns a tuple 
                                    (chain_of_thought, final_answer). If None, no CoT parsing is applied.
        **kwargs: Additional parameters.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        batch_size: int = 1,
        cot: bool = False,
        cot_trigger: Optional[str] = "Let's think step by step.",
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = default_cot_parser,
        **kwargs
    ):
        super().__init__(**kwargs)
        logger.info(f"[HuggingFaceModel] Loading tokenizer/model from {model_name_or_path}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.eval()

        # Device setup
        self.device = device
        if self.device != "cpu":
            self.model.to(self.device)
            logger.info(f"[HuggingFaceModel] Model moved to {self.device}")

        # Inference hyperparameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.cot = cot
        self.cot_trigger = cot_trigger
        self.cot_parser = cot_parser

    def _score_option(self, prompt: str, option: str) -> float:
        """
        Computes the log likelihood score for a given prompt-option pair.
        Concatenates the prompt and option, tokenizes the combined text, and sums the log probabilities
        for the tokens corresponding to the option.
        """
        full_text = prompt + " " + option
        encoded_full = self.tokenizer(full_text, return_tensors="pt")
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = encoded_prompt.input_ids.shape[1]
        input_ids = encoded_full.input_ids.to(self.device)
        attention_mask = encoded_full.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        option_ids = input_ids[0, prompt_len:]
        total_log_prob = 0.0
        for t, token_id in enumerate(option_ids):
            # Adjust index if necessary; here we use a simple approach.
            index = prompt_len + t - 1 if prompt_len + t - 1 < logits.shape[1] else -1
            token_log_prob = log_probs[0, index, token_id].item()
            total_log_prob += token_log_prob
        return total_log_prob

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = True,
        batch_size: Optional[Union[int, str]] = None,
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Processes a batch of inputs to generate text outputs and updates each item with the final prediction.

        MCQA mode:
            If an input item contains an "options" field (list), compute the log likelihood for each option,
            store the scores in item["logits"]["option_log_probs"], and select the option with the highest score
            as the final prediction. Chain-of-thought (CoT) processing is applied if enabled.

        Args:
            inputs (List[Dict[str, Any]]): List of items, each with at least {"input": str, "reference": str, ...}.
            return_logits (bool): If True, compute and store log probabilities in item["logits"].
            batch_size (int|str): The batch size to use. If "auto", starts with a default size and reduces on OOM.
            until (str|List[str]|None): Optional stopping condition(s).
            show_progress (bool): Whether to display a progress bar.
            **kwargs: Additional arguments.

        Returns:
            List[Dict[str, Any]]: The updated list of items, with added fields:
                - "prediction": the final generated answer (or selected option in MCQA mode)
                - "chain_of_thought": (optional) parsed CoT text if applicable
                - "logits": (optional) dictionary containing log probability details if return_logits is True.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if isinstance(batch_size, str) and batch_size.lower() == "auto":
            auto_mode = True
            current_bs = 128  # Default starting batch size in auto mode
            logger.info(f"[HuggingFaceModel] Batch size set to 'auto'. Starting with {current_bs}.")
        else:
            auto_mode = False
            current_bs = batch_size if batch_size is not None else len(inputs)
            logger.info(f"[HuggingFaceModel] Batch size set to {current_bs}.")

        # Setup stopping criteria if provided
        stopping_criteria = None
        if until is not None:
            if isinstance(until, str):
                until = [until]

            class StoppingCriteriaSub(StoppingCriteria):
                def __init__(self, tokenizer, stops: List[str]):
                    super().__init__()
                    self.tokenizer = tokenizer
                    self.stops = stops

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    decoded = self.tokenizer.decode(input_ids[0])
                    return any(stop in decoded for stop in self.stops)

            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(self.tokenizer, stops=until)])

        while True:
            try:
                results = []
                for start in tqdm(range(0, len(inputs), current_bs), disable=not show_progress):
                    batch_items = inputs[start:start + current_bs]
                    
                    # Check for MCQA mode: if any item contains an "options" field as a list.
                    mcqa_flags = [("options" in item and isinstance(item["options"], list)) for item in batch_items]
                    
                    if any(mcqa_flags):
                        # Process each item individually in MCQA mode.
                        for item in batch_items:
                            if "options" in item and isinstance(item["options"], list):
                                prompt = item.get("input", "")
                                option_log_probs = []
                                for opt in item["options"]:
                                    lp = self._score_option(prompt, opt)
                                    option_log_probs.append(lp)
                                item["logits"] = {"option_log_probs": option_log_probs}
                                best_idx = option_log_probs.index(max(option_log_probs))
                                item["prediction"] = item["options"][best_idx]
                                # Optionally process chain-of-thought if enabled
                                if self.cot and self.cot_trigger:
                                    generated_text = f"{prompt}\n{self.cot_trigger}\n{item['prediction']}"
                                    cot, final_answer = self.cot_parser(generated_text)
                                    item["chain_of_thought"] = cot
                                    item["prediction"] = final_answer
                                results.append(item)
                            else:
                                # For items without options, fall back to normal generation.
                                results.extend(self._generate_normal(batch_items, **kwargs))
                        return results
                    else:
                        # Normal generation mode.
                        results = self._generate_normal(batch_items, **kwargs)
                return results
            except RuntimeError as e:
                # Handle out-of-memory (OOM) errors by reducing batch size in auto mode.
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if auto_mode:
                        if current_bs > 1:
                            current_bs = max(1, current_bs // 2)
                            logger.warning(f"[HuggingFaceModel] OOM detected: reducing batch size to {current_bs}.")
                        else:
                            logger.error("[HuggingFaceModel] Batch size is 1 but OOM persists.")
                            raise RuntimeError("Out of memory even with batch size=1.") from e
                    else:
                        logger.error("[HuggingFaceModel] OOM with specified batch size.")
                        raise RuntimeError("Out of memory with specified batch size.") from e
                else:
                    logger.error("[HuggingFaceModel] RuntimeError occurred:", exc_info=True)
                    raise

    def _generate_normal(self, batch_items: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Handles normal text generation (non-MCQA mode) for a batch of items.
        """
        results = []
        # Build prompts; append the CoT trigger if enabled.
        prompts = [
            f"{item['input']}\n{self.cot_trigger}\n" if (self.cot and self.cot_trigger) else item["input"]
            for item in batch_items
        ]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_lens = encoded["attention_mask"].sum(dim=1).tolist()
        if self.device != "cpu":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }
        if "stopping_criteria" in kwargs:
            gen_kwargs["stopping_criteria"] = kwargs["stopping_criteria"]
        # Ensure logits are returned if needed.
        if gen_kwargs.get("output_scores", False):
            gen_kwargs.update({
                "return_dict_in_generate": True,
                "output_scores": True,
            })

        with torch.no_grad():
            outputs = self.model.generate(**encoded, **gen_kwargs)
        if isinstance(outputs, dict):
            sequences = outputs.get("sequences", outputs)
            scores_list = outputs.get("scores", None)
        else:
            sequences = outputs
            scores_list = None

        batch_size_actual = sequences.shape[0]
        for i in range(batch_size_actual):
            item = batch_items[i]
            input_len = input_lens[i]
            gen_ids = sequences[i][input_len:]  # Extract the generated tokens.
            generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            final_answer = generated_text
            chain_of_thought = None
            if self.cot and self.cot_parser is not None:
                chain_of_thought, final_answer = self.cot_parser(generated_text)
            item["prediction"] = final_answer
            if chain_of_thought is not None:
                item["chain_of_thought"] = chain_of_thought
            if gen_kwargs.get("output_scores", False) and scores_list is not None:
                log_probs_per_token = []
                sum_log_prob = 0.0
                token_ids = gen_ids.tolist()
                for t, step_scores in enumerate(scores_list):
                    if t >= len(token_ids):
                        break
                    step_score = step_scores[i]
                    step_log_probs = F.log_softmax(step_score, dim=-1)
                    token_log_prob = step_log_probs[token_ids[t]].item()
                    sum_log_prob += token_log_prob
                    log_probs_per_token.append(token_log_prob)
                item["logits"] = {
                    "sum_log_prob": sum_log_prob,
                    "token_log_probs": log_probs_per_token,
                    "tokens": self.tokenizer.convert_ids_to_tokens(token_ids),
                }
            results.append(item)
        return results
