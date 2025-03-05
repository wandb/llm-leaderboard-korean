import logging
import time
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from tqdm import tqdm

from . import register_model
from .base import BaseJudge
from llm_eval.utils.logging import get_logger

logger = get_logger(name="litellm_judge", level=logging.INFO)

@register_model("litellm_judge")
class LiteLLMJudge(BaseJudge):
    """
    A judge backend implementation using LiteLLM API.

    This judge model is intended to evaluate generated answers by constructing judge prompts 
    and calling the LiteLLM API. It supports:
      - Basic retry logic with exponential backoff.
      - Batch processing via multithreading.
    
    Args:
        provider (str): LLM provider (e.g., "openai", "anthropic", "bedrock", "azure").
        model_name (str): Name of the judge model to use.
        api_key (Optional[str]): API key for the provider.
        api_base (Optional[str]): Base URL for the API.
        max_new_tokens (int): Maximum tokens to generate in judge response.
        temperature (float): Sampling temperature.
        batch_size (int): Number of judge prompts to process concurrently.
        **kwargs: Additional parameters.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        batch_size: int = 8,
        **kwargs
    ):
        super().__init__(**kwargs)
        logger.info(f"[LiteLLMJudge] Initializing judge model '{model_name}' for provider '{provider}'.")

        self.provider = provider.lower()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.extra_kwargs = kwargs

        # Configure API settings
        self.completion_kwargs = {
            "api_key": api_key,
            "api_base": api_base,
        }
        if self.provider == "bedrock":
            self.completion_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
            })
        elif self.provider == "anthropic":
            self.completion_kwargs["api_key"] = anthropic_api_key

    def _prepare_completion_kwargs(self, prompt: str, until: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Prepares the completion parameters for the LiteLLM judge API.

        Args:
            prompt (str): Judge prompt text.
            until (Optional[Union[str, List[str]]]): Optional stop sequence(s).

        Returns:
            Dict[str, Any]: Dictionary of parameters for the LiteLLM API.
        """
        completion_kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            **self.completion_kwargs,
            **self.extra_kwargs,
        }
        if until is not None:
            completion_kwargs["stop"] = until if isinstance(until, list) else [until]

        if self.provider == "azure":
            completion_kwargs.update({
                "model": self.model_name,
                "engine": self.model_name,
            })
        elif self.provider == "bedrock":
            completion_kwargs["model"] = f"bedrock/{self.model_name}"
        else:
            completion_kwargs["model"] = self.model_name

        logger.debug(f"[LiteLLMJudge] Prepared completion kwargs: {completion_kwargs}")
        return completion_kwargs

    def _generate_with_retry(
        self, 
        completion_kwargs: Dict[str, Any], 
        max_attempts: int = 3,
        initial_wait: int = 4
    ) -> str:
        """
        Calls the LiteLLM judge API with retry logic.

        Args:
            completion_kwargs (Dict[str, Any]): Parameters for the API call.
            max_attempts (int): Maximum retry attempts.
            initial_wait (int): Initial wait time (seconds) for exponential backoff.

        Returns:
            str: Generated judge response.

        Raises:
            Exception: If all attempts fail.
        """
        attempt = 0
        last_exception = None
        while attempt < max_attempts:
            try:
                response = litellm.completion(**completion_kwargs)
                return response.choices[0].message.content
            except Exception as e:
                attempt += 1
                last_exception = e
                if attempt < max_attempts:
                    wait_time = initial_wait * (2 ** (attempt - 1))
                    logger.warning(f"[LiteLLMJudge] Attempt {attempt} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        error_msg = f"[LiteLLMJudge] All {max_attempts} attempts failed. Last error: {str(last_exception)}"
        logger.error(error_msg)
        raise last_exception or Exception(error_msg)

    def judge_batch(
        self,
        inputs: List[Dict[str, Any]],
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Processes a batch of judge prompts concurrently using multithreading.

        Args:
            inputs (List[Dict[str, Any]]): A list of judge prompt items.
                Each item should have an "input" field containing the judge prompt.
            until (Optional[Union[str, List[str]]]): Optional stopping conditions.
            show_progress (bool): Whether to display a progress bar.
            **kwargs: Additional parameters.

        Returns:
            List[Dict[str, Any]]: The input list with each item updated with a "prediction"
            field containing the judge model's raw response.
        """
        results = []

        # Define the per-item processing function.
        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            prompt = item["input"]
            completion_kwargs = self._prepare_completion_kwargs(prompt, until=until)
            try:
                prediction = self._generate_with_retry(completion_kwargs)
                result_item = {
                    "input": item["input"],
                    "prediction": prediction
                }
                return result_item
            except Exception as e:
                logger.error(f"[LiteLLMJudge] Error generating judge response: {str(e)}")
                return {
                    "input": prompt,
                    "prediction": f"Error: {str(e)}"
                }

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in inputs}
            if show_progress:
                for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="LiteLLM Judge Batch"):
                    results.append(future.result())
            else:
                for future in as_completed(future_to_item):
                    results.append(future.result())

        return results
