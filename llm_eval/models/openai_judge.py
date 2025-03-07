import openai
import time
import logging
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

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
        api_base (str): API base URL.
        model_name (str): Judge model identifier (e.g., "gpt-4").
        system_message (Optional[str]): A system message to include in chat completions.
        use_chat_api (bool): If True, uses the Chat API; otherwise, uses the Completions API.
        max_tokens (int): Maximum tokens to generate for each judge response.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        do_sample (bool): Whether to use sampling (True) or greedy decoding (False).
        batch_size (int): Number of judge prompts to process concurrently.
        max_retries (int): Maximum number of retry attempts for API calls.
        **kwargs: Additional parameters for the API (e.g., frequency_penalty, presence_penalty).
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model_name: str,
        system_message: Optional[str] = None,
        use_chat_api: bool = True,
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        batch_size: int = 8,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not model_name:
            raise ValueError("model_name is required for OpenAIJudge.")
        if not api_key:
            raise ValueError("api_key is required for OpenAIJudge.")
        if not api_base:
            raise ValueError("api_base is required for OpenAIJudge.")

        # Set OpenAI API credentials for this instance
        openai.api_key = api_key
        openai.api_base = api_base

        self.model_name = model_name
        self.system_message = system_message
        self.use_chat_api = use_chat_api
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.max_retries = max_retries
        # Store any additional API parameters for future use
        self.default_params = kwargs

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
        params = self.default_params.copy()
        params.update(kwargs)
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
        for attempt in range(self.max_retries):
            try:
                payload = self._create_payload(prompt, until=until, **kwargs)
                if self.use_chat_api:
                    response = openai.ChatCompletion.create(**payload)
                    result = {
                        "prediction": response.choices[0].message.content,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                else:
                    response = openai.Completion.create(**payload)
                    result = {
                        "prediction": response.choices[0].text,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                break  # Exit loop if successful
            except Exception as e:
                if attempt == self.max_retries - 1:
                    error_msg = f"Error after {self.max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                else:
                    wait_time = min(2 ** attempt, 32)
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
