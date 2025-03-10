import asyncio
import time
import logging
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import httpx
from tqdm import tqdm

from .base import BaseJudge
from . import register_model
from llm_eval.utils.logging import get_logger

logger = get_logger(name="openai_judge", level=logging.INFO)


@register_model("openai_judge")
class OpenAIJudge(BaseJudge):
    """
    OpenAIJudge is a production-grade judge backend that uses an HTTP-based API (via httpx)
    to evaluate generated answers. This implementation supports:
      - Both Chat and Completions API payload structures, controlled by the use_chat_api flag.
      - Chain-of-thought (CoT) prompting: if enabled, the CoT trigger is appended to the prompt,
        and the generated output is parsed into a chain-of-thought and a final answer.
      - Concurrent batch processing using asynchronous HTTP requests with httpx.
      - Robust retry logic with exponential backoff on API errors.
    
    Args:
        api_base (str): The base URL of the judge API endpoint.
        model_name (str): Judge model identifier (e.g., "gpt-4").
        system_message (Optional[str]): A system message to include in chat completions.
        use_chat_api (bool): If True, uses a Chat API payload; otherwise, uses a Completions API payload.
        max_tokens (int): Maximum tokens to generate for each judge response.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        do_sample (bool): Whether to use sampling (True) or greedy decoding (False).
        batch_size (int): Number of judge prompts to process concurrently.
        max_retries (int): Maximum number of retry attempts for API calls.
        timeout (Optional[float]): Timeout in seconds for HTTP requests.
        **kwargs: Additional parameters for the API (e.g., frequency_penalty, presence_penalty).
    """
    def __init__(
        self,
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
        timeout: Optional[float] = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not model_name:
            raise ValueError("model_name is required for OpenAIJudge.")
        if not api_base:
            raise ValueError("api_base is required for OpenAIJudge.")
        
        # Judge API endpoint is specified via api_base; API key is not required.
        self.api_base = api_base
        self.model_name = model_name
        self.system_message = system_message
        self.use_chat_api = use_chat_api
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.default_params = kwargs

    def _create_payload(
        self,
        prompt: str,
        until: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Constructs the API payload for the judge call.
        Supports both Chat and Completions API payload formats.
        If the payload is for the Chat API and a system message is provided, it is prepended.
        If 'until' is provided, it is added as a stop sequence.
        """
        params = deepcopy(self.default_params)
        params.update(kwargs)
        payload = {"model": self.model_name}
        if until is not None:
            if isinstance(until, str):
                until = [until]
            payload["stop"] = until

        if self.use_chat_api:
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            messages.append({"role": "user", "content": prompt})
            payload["messages"] = messages
        else:
            payload["prompt"] = prompt
            if params.get("logprobs") is not None:
                payload["logprobs"] = params["logprobs"]

        for param in ["max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if param in params:
                payload[param] = params[param]
        return {k: v for k, v in payload.items() if v is not None}

    async def _send_single_request(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        until: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sends a single HTTP POST request to the judge API endpoint using httpx.
        Implements retry logic with exponential backoff.
        """
        effective_retries = self.max_retries
        payload = self._create_payload(prompt, until=until, **kwargs)
        attempt = 0
        while attempt <= effective_retries:
            try:
                response = await client.post(self.api_base, json=payload, timeout=self.timeout)
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code} Error: {response.text}")
                resp_data = response.json()
                # Parse response: expected format similar to OpenAI ChatCompletion
                try:
                    message = resp_data["choices"][0]["message"]
                    result = {
                        "prediction": message.get("content", json.dumps(resp_data, indent=2)),
                        "finish_reason": message.get("finish_reason", ""),
                    }
                except (KeyError, IndexError):
                    result = {
                        "prediction": json.dumps(resp_data, indent=2),
                        "finish_reason": "",
                    }
                return result
            except Exception as e:
                logger.error(f"HTTP attempt {attempt + 1}/{effective_retries} failed: {e}")
                attempt += 1
                await asyncio.sleep(min(2 ** attempt, 32))
        raise RuntimeError(f"Failed after {effective_retries} retries via httpx.")

    async def _generate_batch_async(
        self,
        inputs: List[Dict[str, Any]],
        until: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously processes a batch of judge prompts using httpx.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = []
            for item in inputs:
                prompt = item["input"]
                tasks.append(self._send_single_request(client, prompt, until=until, **kwargs))
            results = await asyncio.gather(*tasks, return_exceptions=False)
        merged_results = []
        for orig, res in zip(inputs, results):
            merged = deepcopy(orig)
            merged.update(res)
            merged_results.append(merged)
        return merged_results

    def judge_batch(
        self,
        inputs: List[Dict[str, Any]],
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Processes a batch of judge prompts concurrently using httpx.
        Each input dictionary must have an "input" field containing the judge prompt.
        
        Args:
            inputs (List[Dict[str, Any]]): List of judge prompt dictionaries.
            until (Optional[Union[str, List[str]]]): Stop sequence(s) for generation.
            show_progress (bool): If True, displays a progress bar.
            **kwargs: Additional parameters for API calls.
        
        Returns:
            List[Dict[str, Any]]: The list of input dictionaries updated with a "prediction" field
                                  containing the judge model's output.
        """
        try:
            return asyncio.run(self._generate_batch_async(inputs, until, **kwargs))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(self._generate_batch_async(inputs, until, **kwargs))
            else:
                raise
