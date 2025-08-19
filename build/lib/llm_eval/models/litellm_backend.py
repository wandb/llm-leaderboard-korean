import logging
import time
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from tqdm import tqdm

from . import register_model
from .base import BaseModel
from llm_eval.utils.prompt_template import default_cot_parser  
from llm_eval.utils.logging import get_logger

logger = get_logger(name="litellm", level=logging.INFO)

@register_model("litellm")
class LiteLLMBackend(BaseModel):
    """
    A backend model class that uses LiteLLM.
    
    Args:
        provider (str): Name of the LLM provider (e.g., "openai", "anthropic", "bedrock", "azure").
        model_name (str): The model name to use.
        api_key (Optional[str]): API key for providers like OpenAI or Azure.
        api_base (Optional[str]): Base URL for the API.
        aws_access_key_id (Optional[str]): AWS access key ID (for Bedrock).
        aws_secret_access_key (Optional[str]): AWS secret key (for Bedrock).
        anthropic_api_key (Optional[str]): Anthropic API key.
        max_new_tokens (int): Maximum number of tokens to generate per call.
        temperature (float): Sampling temperature (0.0 ~ 1.0).
        batch_size (int): Number of items to process concurrently (using multithreading).
        cot (bool): If True, appends a CoT trigger to the prompt.
        cot_trigger (Optional[str]): Trigger phrase for inducing CoT. If None, no trigger is used.
        cot_parser (Optional[Callable[[str], Tuple[str, str]]]): A callable to parse generated text into 
            (chain_of_thought, final_answer). Defaults to default_cot_parser.
        **kwargs: Additional configuration parameters.
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
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        batch_size: int = 1,
        cot: bool = False,
        cot_trigger: Optional[str] = None,
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = default_cot_parser,
        **kwargs
    ):
        super().__init__(**kwargs)
        logger.info(f"[LiteLLMBackend] Loading model settings for '{model_name}' from provider '{provider}'.")
        
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.cot = cot
        self.cot_trigger = cot_trigger
        self.cot_parser = cot_parser
        self.extra_kwargs = kwargs

        # Configure API settings specific to LiteLLM
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
        Prepares the parameters to pass to the LiteLLM completion function.
        
        Args:
            prompt (str): The input prompt text.
            until (Optional[Union[str, List[str]]]): Optional stop sequence(s).
        
        Returns:
            Dict[str, Any]: Dictionary of completion parameters.
        """
        # Append the CoT trigger if chain-of-thought is enabled.
        if self.cot and self.cot_trigger:
            prompt = f"{prompt}\n{self.cot_trigger}"
            
        completion_kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            **self.completion_kwargs,
            **self.extra_kwargs,
        }
        if until is not None:
            completion_kwargs["stop"] = until if isinstance(until, list) else [until]
        
        # Adjust model name formatting based on provider
        if self.provider == "azure":
            completion_kwargs.update({
                "model": self.model_name,
                "engine": self.model_name,
            })
        elif self.provider == "bedrock":
            completion_kwargs["model"] = f"bedrock/{self.model_name}"
        else:
            completion_kwargs["model"] = self.model_name
        
        logger.debug(f"[LiteLLMBackend] Prepared completion kwargs: {completion_kwargs}")
        return completion_kwargs

    def _generate_with_retry(
        self, 
        completion_kwargs: Dict[str, Any], 
        max_attempts: int = 3,
        initial_wait: int = 4
    ) -> str:
        """
        Calls the LiteLLM completion API with retry logic.
        
        Args:
            completion_kwargs (Dict[str, Any]): Parameters for the completion API.
            max_attempts (int): Maximum number of retry attempts.
            initial_wait (int): Initial wait time (in seconds) before the first retry.
        
        Returns:
            str: The generated text.
        
        Raises:
            Exception: If all retry attempts fail.
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
                    logger.warning(f"[LiteLLMBackend] Attempt {attempt} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        error_msg = f"[LiteLLMBackend] All {max_attempts} attempts failed. Last error: {str(last_exception)}"
        logger.error(error_msg)
        raise last_exception or Exception(error_msg)

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        batch_size: Optional[Union[int, str]] = None,
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generates text for a batch of inputs using multithreading.
        
        Args:
            inputs (List[Dict[str, Any]]): A list of input items. Each item should contain at least:
                - "input": The prompt text.
                - "reference": The reference text (optional).
            return_logits (bool): If True, returns log probabilities (not supported in LiteLLM; raises NotImplementedError).
            batch_size (int | str): Number of items to process concurrently. If "auto", uses the entire input list.
            until (Optional[Union[str, List[str]]]): Optional stopping condition.
            show_progress (bool): Whether to display a progress bar.
            **kwargs: Additional parameters.
        
        Returns:
            List[Dict[str, Any]]: A list of results, where each item includes:
                - "input": Original input text.
                - "reference": Original reference text.
                - "prediction": Generated text (final answer).
                - "chain_of_thought": (Optional) Chain-of-thought text if CoT parsing is enabled.
        """
        if return_logits:
            raise NotImplementedError("LiteLLM backend does not support logits calculation yet.")

        # Determine the batch size
        if batch_size is None:
            batch_size = self.batch_size
        if isinstance(batch_size, str) and batch_size.lower() == "auto":
            batch_size = min(len(inputs), 8)
            logger.info(f"[LiteLLMBackend] Using auto batch size: {batch_size}")

        results = []
        
        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            prompt = item["input"]
            reference = item.get("reference", "")
            # Prepare parameters for the API call
            completion_kwargs = self._prepare_completion_kwargs(prompt, until=until)
            try:
                prediction = self._generate_with_retry(completion_kwargs)
                result_item = {
                    "input": item["input"],
                    "reference": reference,
                    "prediction": prediction
                }
                # If CoT is enabled and a parser is provided, apply it
                if self.cot and self.cot_parser:
                    chain_of_thought, final_answer = self.cot_parser(prediction)
                    result_item["chain_of_thought"] = chain_of_thought
                    result_item["prediction"] = final_answer
                return result_item
            except Exception as e:
                logger.error(f"[LiteLLMBackend] Error generating completion: {str(e)}")
                return {
                    "input": prompt,
                    "reference": reference,
                    "prediction": f"Error: {str(e)}"
                }
        
        # Use ThreadPoolExecutor for concurrent processing (similar to batching)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in inputs}
            if show_progress:
                for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="LiteLLM Batch Generation"):
                    results.append(future.result())
            else:
                for future in as_completed(future_to_item):
                    results.append(future.result())
        
        return results
