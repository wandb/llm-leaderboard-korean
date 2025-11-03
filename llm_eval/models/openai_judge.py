import os
import weave
import logging
import time
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from tqdm import tqdm
from .base import BaseJudge
from . import register_model

# Initialize module-level logger
logger = logging.getLogger(__name__)

@register_model("openai_judge")
class OpenAIJudge(BaseJudge):
    """
    A threaded judge backend for OpenAI-compatible APIs.
    
    Uses a pool of worker threads to send concurrent requests to the API,
    applies simple retry logic with exponential backoff, and returns
    structured results including predictions and finish reasons.
    """
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        batch_size: int = 10,
        **kwargs,
    ):
        """
        Args:
            model_name: Identifier for the model (e.g., "gpt-4o").
            api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var).
            api_base: Base URL for API calls (defaults to https://api.openai.com/v1).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate per request.
            top_p: Nucleus sampling probability.
            frequency_penalty: Penalize repeated tokens.
            presence_penalty: Penalize new topic introduction.
            batch_size: Number of threads to use for parallel calls.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        # Load API credentials and endpoint
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Provide it or set OPENAI_API_KEY."
            )

        # Shared payload parameters for each request
        self.params = {
            "model": self.model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        # Number of worker threads for concurrent evaluation
        self.batch_size = batch_size

    def _build_url(self) -> str:
        """
        Construct the full API URL for chat completions.

        Returns:
            A URL ending in '/chat/completions' or '/completions'.
        """
        base = self.api_base.rstrip('/')
        # If user already provided the full path, use it directly
        if base.endswith("/chat/completions") or base.endswith("/completions"):
            return base
        # Otherwise default to the chat endpoint
        return f"{base}/chat/completions"

    @weave.op(name="openai_judge_single_request")
    def _send_single_request(
        self,
        prompt: str,
        until: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Send one synchronous request to the judge API with basic retries.

        Args:
            prompt: The user prompt to evaluate.
            until: Optional stop sequence(s) for generation.

        Returns:
            A dict containing:
              - 'prediction': The model's response content.
              - 'finish_reason': Why the model stopped (e.g., 'stop', 'length').
        """
        # HTTP headers including auth
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # Merge common params and user message
        payload = {**self.params}
        payload["messages"] = [{"role": "user", "content": prompt}]
        if until:
            payload["stop"] = until

        url = self._build_url()
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                # Each thread uses its own client for simplicity
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()

                # Extract the first choice (chat or completion)
                choice = data.get("choices", [])[0]
                content = (
                    choice.get("message", {}).get("content") or choice.get("text")
                )
                finish = choice.get("finish_reason")
                return {"prediction": content, "finish_reason": finish}

            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                # Log warning and retry with exponential backoff
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == retries:
                    logger.error("Max retries reached.")
                    return {"prediction": f"Error: {e}", "finish_reason": "error"}
                time.sleep(2 ** attempt)

    def judge_batch(
        self,
        inputs: List[Dict[str, Any]],
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a list of prompts in parallel using a thread pool.

        Args:
            inputs: A list of dicts each containing an 'input' key with the prompt.
            until: Optional stop sequence(s) for all prompts.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            The same list of inputs, each updated with 'prediction' and 'finish_reason'.
        """
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            # Submit all requests and map futures to their indices
            future_to_index = {
                executor.submit(
                    self._send_single_request, item["input"], until
                ): i
                for i, item in enumerate(inputs)
            }
            # Wrap as_completed in tqdm for optional progress display
            progress = tqdm(
                as_completed(future_to_index),
                total=len(inputs),
                desc="Judging Batches",
                disable=not show_progress,
            )
            # As each thread finishes, store its result
            for future in progress:
                idx = future_to_index[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Error in thread {idx}: {e}")
                    result = {"prediction": f"Error: {e}", "finish_reason": "error"}
                inputs[idx].update(result)
        return inputs
