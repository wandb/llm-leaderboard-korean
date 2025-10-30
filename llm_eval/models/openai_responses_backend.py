"""
OpenAI Responses API Backend

This module provides a backend for OpenAI's Responses API, which is used for
reasoning models like o1, o4-mini, and GPT-5. The Responses API differs from
the Chat Completions API in how it handles reasoning chains and output tokens.

Key differences:
- Uses `max_output_tokens` instead of `max_tokens`
- Supports `reasoning` parameter for controlling reasoning effort
- Returns reasoning content separately from the main response
- Input format uses messages array directly (not wrapped in "messages" key)
"""

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai
import weave
from tqdm import tqdm

from llm_eval.utils.logging import get_logger
from . import register_model
from .base import BaseModel

logger = get_logger(name="openai_responses_backend", level=logging.INFO)


@register_model("openai_responses")
class OpenAIResponsesModel(BaseModel):
    """
    OpenAI Responses API backend for reasoning models.

    This backend supports:
    - Reasoning models (o1, o4-mini, GPT-5)
    - Extended reasoning chains with `reasoning` parameter
    - Separate reasoning content and response content
    - Tool/function calling

    Args:
        api_key (Optional[str]): OpenAI API key (or uses OPENAI_API_KEY env var)
        api_base (str): Base URL for the API (default: https://api.openai.com/v1)
        model_name (str): Model identifier (e.g., "o4-mini-2025-04-16", "gpt-5-2025-08-07")
        batch_size (int): Number of concurrent requests
        max_retries (int): Maximum retry attempts
        timeout (float): HTTP timeout in seconds
        reasoning (Optional[Dict]): Reasoning configuration
            - effort: "low", "medium", "high" (default: "medium")
            - summary: "auto", "full", "none" (default: "auto")
        **kwargs: Additional API parameters (temperature, top_p, etc.)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        model_name: str = None,
        batch_size: int = 8,
        max_retries: int = 3,
        timeout: float = 600.0,  # 10 minutes default for reasoning models
        reasoning: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not model_name:
            raise ValueError("model_name is required")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.reasoning = reasoning or {"effort": "medium", "summary": "auto"}
        self.default_params = kwargs

        # Initialize async OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=api_base,
            timeout=self.timeout,
            max_retries=0,  # We handle retries manually
        )

        logger.info(
            f"Initialized OpenAI Responses API backend for model: {self.model_name}"
        )

    def _prepare_params(
        self, max_tokens: Optional[int] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare parameters for Responses API call.

        Args:
            max_tokens: Maximum output tokens
            **kwargs: Additional parameters

        Returns:
            Dict with filtered and mapped parameters
        """
        # Merge default params with call-specific params
        all_params = {**self.default_params, **kwargs}

        # Responses API allowed parameters
        allowed_params = {
            "instructions",
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "parallel_tool_calls",
            "previous_response_id",
            "reasoning",
            "service_tier",
            "store",
            "stream",
            "text",
            "text_format",
            "tool_choice",
            "tools",
            "top_logprobs",
            "truncation",
            "user",
        }

        # Map common parameter names to Responses API names
        param_mapping = {
            "max_tokens": "max_output_tokens",
            "top_k": "top_logprobs",
        }

        # Build filtered params
        filtered_params = {}
        for key, value in all_params.items():
            # Map parameter name if needed
            api_key = param_mapping.get(key, key)

            # Only include if allowed
            if api_key in allowed_params:
                filtered_params[api_key] = value

        # Override max_output_tokens if provided
        # Apply model-specific limits to prevent API errors
        if max_tokens:
            # Define model-specific max_output_tokens limits
            model_limits = {
                "gpt-5-mini": 32768,
                "gpt-5": 65536,
                "o4-mini": 65536,
                "o1": 100000,
            }

            # Find applicable limit based on model name
            max_limit = None
            for model_prefix, limit in model_limits.items():
                if model_prefix in self.model_name.lower():
                    max_limit = limit
                    break

            # Apply limit: max(16, min(requested, model_limit))
            if max_limit:
                capped_tokens = min(max_tokens, max_limit)
                if capped_tokens < max_tokens:
                    logger.info(f"Capping max_output_tokens from {max_tokens} to {capped_tokens} "
                              f"(model limit for {self.model_name})")
                filtered_params["max_output_tokens"] = max(capped_tokens, 16)
            else:
                filtered_params["max_output_tokens"] = max(max_tokens, 16)

        # Add reasoning config if not already present
        if "reasoning" not in filtered_params and self.reasoning:
            filtered_params["reasoning"] = self.reasoning

        return filtered_params

    @weave.op()
    async def _call_api_async(
        self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Make an async call to the Responses API.

        Args:
            messages: List of message dicts
            max_tokens: Maximum output tokens
            **kwargs: Additional parameters

        Returns:
            Dict with content, reasoning_content, usage, etc.
        """
        params = self._prepare_params(max_tokens=max_tokens, **kwargs)

        # Build request
        request_params = {
            "model": self.model_name,
            "input": messages,  # Responses API uses "input" not "messages"
            **params,
        }

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = await self.client.responses.create(**request_params)

                # Parse response
                content = ""
                reasoning_content = ""

                for output in response.output:
                    if output.type == "message":
                        # Extract text from message
                        try:
                            if output.content and len(output.content) > 0:
                                content = output.content[0].text
                        except Exception:
                            pass
                    elif output.type == "reasoning":
                        # Extract reasoning summary
                        try:
                            if hasattr(output, "summary") and len(output.summary) > 0:
                                reasoning_content = output.summary[0].text
                        except Exception:
                            pass

                # Extract usage metadata for Weave tracking
                usage = getattr(response, "usage", None)

                return {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "usage": usage,
                    "raw_response": response,
                }

            except openai.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Rate limit error, retrying in {wait_time}s... (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except openai.APITimeoutError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Timeout error, retrying... (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(1)
                else:
                    raise

            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise

        # Should not reach here
        return {"content": "", "reasoning_content": ""}

    @weave.op()
    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        cot: bool = False,
        batch_size: Optional[Union[int, str]] = "auto",
        until: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of inputs.

        Args:
            inputs: List of input dicts with "input" key
            return_logits: Not supported for Responses API
            cot: Not needed (reasoning is built-in)
            batch_size: Batch size for concurrent requests
            until: Stop sequences (not typically used with Responses API)

        Returns:
            List of input dicts with "prediction", "chain_of_thought", and "usage" added
        """
        if return_logits:
            logger.warning(
                "return_logits=True is not supported for OpenAI Responses API"
            )

        # Determine batch size
        if batch_size == "auto" or batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = int(batch_size)

        # Prepare all async tasks
        async def process_all():
            tasks = []
            for sample in inputs:
                input_text = sample.get("input", "")

                # Format as messages (simple user message)
                messages = [{"role": "user", "content": input_text}]

                # Get max_tokens from sample or use default
                max_tokens = sample.get("max_tokens")

                # Create task
                task = self._call_api_async(messages, max_tokens=max_tokens)
                tasks.append((sample, task))

            # Execute in batches
            results = []
            for i in tqdm(
                range(0, len(tasks), batch_size),
                desc="Generating with OpenAI Responses API",
            ):
                batch_tasks = [task for _, task in tasks[i : i + batch_size]]
                batch_samples = [sample for sample, _ in tasks[i : i + batch_size]]

                # Run batch concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # Process results
                for sample, result in zip(batch_samples, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing sample: {result}")
                        sample["prediction"] = ""
                        sample["chain_of_thought"] = ""
                    else:
                        sample["prediction"] = result.get("content", "")
                        sample["chain_of_thought"] = result.get("reasoning_content", "")
                        # Store usage metadata for Weave tracking
                        if result.get("usage"):
                            sample["usage"] = result.get("usage")

                    results.append(sample)

            return results

        # Run async event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(process_all())
