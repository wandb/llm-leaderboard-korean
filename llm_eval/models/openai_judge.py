import openai
import time
import logging
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import re

from .base import BaseJudge
from . import register_model
from llm_eval.utils.logging import get_logger
from tqdm import tqdm

logger = get_logger(name="openai_judge", level=logging.INFO)


@register_model("openai_judge")
class OpenAIJudge(BaseJudge):
    """
    OpenAIJudge is a production-grade judge backend that uses the OpenAI API to evaluate generated answers.
    It supports:
      - Both Chat and Completions APIs (controlled by the use_chat_api flag)
      - Chain-of-thought (CoT) prompting: if enabled, the CoT trigger is appended to the prompt and
        the generated output is parsed into a chain-of-thought and a final answer.
      - Batch processing with multi-threading to improve throughput.
      - Robust retry logic with exponential backoff on API errors.
    
    Args:
        api_key (str): OpenAI API key.
        api_base (str, optional): OpenAI API base URL. Defaults to "https://api.openai.com/v1".
        model_name (str): OpenAI model name (e.g., "gpt-4o", "gpt-4", "gpt-3.5-turbo").
        model_name_or_path (str, optional): Huggingface model name or path.
        system_message (Optional[str]): A system message to include in chat completions.
        use_chat_api (bool): If True, uses the Chat API; otherwise, uses the Completions API.
        max_tokens (int): Maximum tokens to generate for each judge response.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        do_sample (bool): Whether to use sampling (True) or greedy decoding (False).
        batch_size (int): Number of judge prompts to process concurrently.
        retry_attempts (int): Number of retry attempts for API calls.
        retry_delay (float): Delay between retry attempts in seconds.
        **kwargs: Additional parameters for the API (e.g., frequency_penalty, presence_penalty).
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        model_name_or_path: str = None,
        system_message: Optional[str] = None,
        use_chat_api: bool = True,
        max_tokens: int = 64,
        temperature: float = 0.1,
        top_p: float = 0.95,
        do_sample: bool = True,
        batch_size: int = 4,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        logger.info(f"[OpenAIJudge] Initializing with model: {model_name}")
        
        self.model_name = model_name_or_path if model_name_or_path else model_name
        self.system_message = system_message
        self.use_chat_api = use_chat_api
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Initialize OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["base_url"] = api_base
        
        self.client = openai.OpenAI(**client_kwargs)

    def _create_payload(
        self,
        prompt: str,
        until: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Constructs the API payload for the OpenAI judge call.
        Supports both Chat API and Completions API.
        
        Args:
            prompt (str): The judge prompt containing the evaluation context.
            until (str or List[str], optional): Stop sequence(s) for generation.
            **kwargs: Additional parameters to include in the payload.
        
        Returns:
            Dict[str, Any]: The payload dictionary for the API call.
        """
        params = kwargs.copy()
        payload = {"model": self.model_name}

        # Add stop sequences if provided.
        if until is not None:
            if isinstance(until, str):
                until = [until]
            payload["stop"] = until

        if self.use_chat_api:
            # Build message list for Chat API
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            messages.append({"role": "user", "content": prompt})
            payload["messages"] = messages
        else:
            # For Completions API, use the prompt field
            payload["prompt"] = prompt
            payload["logprobs"] = params.get("logprobs", None)

        # Add additional generation parameters
        for param in ["max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if param in params:
                payload[param] = params[param]

        # Remove keys with None values before returning
        return {k: v for k, v in payload.items() if v is not None}

    def _generate_single(
        self,
        input_item: Dict[str, Any],
        until: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates a judge response for a single input item using the OpenAI API.
        Implements retry logic with exponential backoff.
        
        Args:
            input_item (Dict[str, Any]): A dictionary containing at least an "input" field.
            until (str or List[str], optional): Stop sequence(s) for generation.
            **kwargs: Additional parameters for payload creation.
        
        Returns:
            Dict[str, Any]: The result dictionary with a "prediction" field containing the judge's output.
        """
        result = None
        prompt = input_item["input"]
        for attempt in range(self.retry_attempts):
            try:
                payload = self._create_payload(prompt, until=until, **kwargs)
                if self.use_chat_api:
                    response = self.client.chat.completions.create(**payload)
                    content = response.choices[0].message.content
                    # Parse score from content
                    score_pattern = r"\[RESULT\]\s*(\d+)"
                    score_match = re.search(score_pattern, content)
                    score = int(score_match.group(1)) if score_match else None
                    
                    result = {
                        "prediction": content,
                        "judge_score": score,  # Add explicit judge_score field
                        "finish_reason": response.choices[0].finish_reason,
                    }
                else:
                    response = self.client.completions.create(**payload)
                    content = response.choices[0].text
                    score_pattern = r"\[RESULT\]\s*(\d+)"
                    score_match = re.search(score_pattern, content)
                    score = int(score_match.group(1)) if score_match else None
                    
                    result = {
                        "prediction": content,
                        "judge_score": score,  # Add explicit judge_score field
                        "finish_reason": response.choices[0].finish_reason,
                    }
                break  # Exit loop if successful
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    error_msg = f"Error after {self.retry_attempts} attempts: {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                else:
                    wait_time = min(self.retry_delay ** attempt, 32)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        return result or {"error": "Generation failed", "prediction": None, "finish_reason": "error"}

    def judge_batch(
        self,
        inputs: List[Dict[str, Any]],
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Processes a batch of judge prompts concurrently using multi-threading.
        Each input dictionary should have an "input" field containing the judge prompt.
        
        Args:
            inputs (List[Dict[str, Any]]): List of judge prompt dictionaries.
            until (str or List[str], optional): Stop sequence(s) for generation.
            show_progress (bool): Whether to display a progress bar.
            **kwargs: Additional parameters for API calls.
        
        Returns:
            List[Dict[str, Any]]: The list of input dictionaries updated with a "prediction" field
                                  containing the judge model's raw output.
        """
        results = []

        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            # Work on a deep copy of the input item to preserve original data
            copy_item = deepcopy(item)
            try:
                res = self._generate_single(copy_item, until=until, **kwargs)
                copy_item.update(res)
            except Exception as e:
                logger.error(f"Error processing judge prompt: {str(e)}")
                copy_item.update({"prediction": f"Error: {str(e)}", "finish_reason": "error"})
            return copy_item

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in inputs}
            if show_progress:
                for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="OpenAI Judge Batch"):
                    results.append(future.result())
            else:
                for future in as_completed(future_to_item):
                    results.append(future.result())
        return results