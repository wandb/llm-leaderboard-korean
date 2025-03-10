import openai
import asyncio
import time
import base64
import logging
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import httpx
from tqdm import tqdm

from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger

logger = get_logger(name="openai_backend", level=logging.INFO)


@register_model("openai")
class OpenAIModel(BaseModel):
    """
    A production-grade backend model that supports both the official OpenAI API (used for vision models)
    and an httpx-based asynchronous call (used for plain text generation via vLLM OpenAI-compatible servers).

    When 'is_vision_model' is True, the official OpenAI SDK client is used; otherwise, asynchronous
    httpx calls are used for text generation. This design allows the API key to be optional for text
    generation via vLLM-compatible servers.

    Key Features:
      - Constructs payloads for both Chat and Completions API calls.
      - Processes image inputs by converting URLs or base64-encoded images.
      - Implements robust retry logic with exponential backoff.
      - Executes API calls concurrently using multi-threading (httpx) or ThreadPoolExecutor (SDK).
      - Supports chain-of-thought (CoT) prompting and parsing.

    Args:
        api_key (Optional[str]): OpenAI API key (optional if using an OpenAI-compatible server).
        api_base (str): Base URL for the API.
        model_name (str): Model identifier (e.g., "gpt-4", "gpt-3.5-turbo").
        system_message (Optional[str]): System message for chat completions.
        use_chat_api (bool): Whether to use the Chat API; if False, uses the Completions API.
        is_vision_model (bool): Flag indicating if the model supports vision inputs.
        cot_trigger (Optional[str]): A trigger phrase for chain-of-thought prompting.
        cot_parser (Optional[Callable[[str], Tuple[str, str]]]): Function to parse generated text into (chain_of_thought, final_answer).
        batch_size (int): Number of concurrent API calls for batch processing.
        max_retries (int): Maximum number of retry attempts for API calls.
        timeout (Optional[float]): Timeout (in seconds) for httpx requests.
        cot (bool): Flag to enable chain-of-thought prompting.
        **kwargs: Additional API parameters (e.g., temperature, max_tokens, top_p, etc.).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = None,
        model_name: str = None,
        system_message: Optional[str] = None,
        use_chat_api: bool = True,
        is_vision_model: bool = False,
        cot_trigger: Optional[str] = "Let's think step by step.",
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = None,
        batch_size: int = 8,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        cot: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not model_name or not api_base:
            raise ValueError("model_name and api_base are required")
        
        self.is_vision_model = is_vision_model
        self.use_chat_api = use_chat_api
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.cot = cot
        
        self.model_name = model_name
        self.system_message = system_message
        self.cot_trigger = cot_trigger
        self.cot_parser = cot_parser
        self.default_params = kwargs

        # For vision models, use the OpenAI SDK client
        if self.is_vision_model:
            if api_key:
                openai.api_key = api_key
            openai.api_base = api_base
            self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
            logger.info("Using OpenAI SDK client for vision model.")
        else:
            # For plain text generation, use httpx-based calls via the vLLM-compatible server.
            self.api_base = api_base
            logger.info("Using httpx-based asynchronous calls for text generation.")

    def _process_image_content(self, content: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        Processes image content into the format expected by the OpenAI Vision API.

        Supports URLs, base64 strings, or dictionaries with detailed specifications.
        """
        VALID_DETAILS = {"high", "low", "auto"}
        def validate_detail(detail: str) -> str:
            detail = detail.lower() if detail else "auto"
            return detail if detail in VALID_DETAILS else "auto"
        def process_base64(b64_str: str, mime_type: str = "image/jpeg") -> str:
            try:
                b64_bytes = base64.b64decode(b64_str)
                if len(b64_bytes) > 20 * 1024 * 1024:
                    raise ValueError("Image size exceeds 20MB limit")
                return f"data:{mime_type};base64,{b64_str}"
            except Exception as e:
                raise ValueError(f"Invalid base64 image: {str(e)}")
        if isinstance(content, list):
            max_images = self.default_params.get("max_images", float("inf"))
            if len(content) > max_images:
                raise ValueError(f"Number of images exceeds limit ({max_images})")
            return [self._process_image_content(item) for item in content]
        if isinstance(content, str):
            if content.startswith(("http://", "https://")):
                return {"type": "image_url", "image_url": {"url": content, "detail": "auto"}}
            try:
                return {"type": "image_url", "image_url": {"url": process_base64(content), "detail": "auto"}}
            except:
                return {"type": "text", "text": content}
        elif isinstance(content, dict):
            detail = validate_detail(content.get("detail", "auto"))
            if "image_url" in content:
                if isinstance(content["image_url"], str):
                    return {"type": "image_url", "image_url": {"url": content["image_url"], "detail": detail}}
                return {"type": "image_url", "image_url": {**content["image_url"], "detail": detail}}
            elif "base64" in content:
                mime_type = content.get("mime_type", "image/jpeg")
                return {"type": "image_url", "image_url": {"url": process_base64(content["base64"], mime_type), "detail": detail}}
        return {"type": "text", "text": str(content)}
    
    def _create_payload(
        self,
        inputs: Union[str, List[Dict], Dict],
        cot: bool = False,
        until: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Constructs the payload for an API call.

        Supports both Chat and Completions API calls. If chain-of-thought (CoT) is enabled
        (via the 'cot' parameter), the CoT trigger is appended to the prompt.
        Additionally, if 'until' is provided, it is added as a stop sequence.
        """
        params = deepcopy(self.default_params)
        params.update(kwargs)

        payload = {}
        if self.use_chat_api:
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            if isinstance(inputs, str):
                prompt_text = inputs
                if cot and self.cot_trigger:
                    prompt_text += f"\n{self.cot_trigger}\n"
                messages.append({"role": "user", "content": prompt_text})
            elif isinstance(inputs, list):
                messages.extend(inputs)
            else:
                messages.append({"role": "user", "content": str(inputs)})
            payload = {"model": self.model_name, "messages": messages}
            if until is not None:
                if isinstance(until, str):
                    until = [until]
                payload["stop"] = until
        else:
            prompt_text = inputs if not (cot and self.cot_trigger) else f"{inputs}\n{self.cot_trigger}\n"
            payload = {"model": self.model_name, "prompt": prompt_text}
            if params.get("logprobs") is not None:
                payload["logprobs"] = params["logprobs"]
            if until is not None:
                if isinstance(until, str):
                    until = [until]
                payload["stop"] = until

        for param in ["max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if param in params:
                payload[param] = params[param]
        
        return {k: v for k, v in payload.items() if v is not None}

    def _execute_tool_calls(self, tool_calls: List[dict]) -> str:
        """
        Executes tool calls if present in the response.
        Placeholder: In production, implement actual tool function invocation.
        """
        return "\n".join([f"Executed tool: {tc.get('function', {}).get('name', 'unknown')}" for tc in tool_calls])

    def _parse_normal_response(self, resp_data: dict) -> str:
        """
        Parses a non-streaming response.
        Expected OpenAI ChatCompletion format: choices[0]["message"]["content"].
        If tool_calls are present, executes them.
        """
        try:
            message = resp_data["choices"][0]["message"]
            if "tool_calls" in message and message["tool_calls"]:
                return self._execute_tool_calls(message["tool_calls"])
            return message.get("content", json.dumps(resp_data, indent=2))
        except (KeyError, IndexError):
            return json.dumps(resp_data, indent=2)

    async def _send_single_request_httpx(
        self,
        client: httpx.AsyncClient,
        item: Dict[str, Any],
        return_logits: bool,
        until: Optional[Union[str, List[str]]],
        cot: bool = False,
        max_retries: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sends a single HTTP POST request to the vLLM-compatible server using httpx.
        Implements retry logic with exponential backoff.
        """
        effective_retries = max_retries if max_retries is not None else self.max_retries
        payload = self._create_payload(item["input"], cot=cot, until=until, **kwargs)
        attempt = 0
        logger.info(f"Starting HTTP request for input: {item['input']}")
        while attempt <= effective_retries:
            try:
                response = await client.post(self.api_base, json=payload, timeout=self.timeout)
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code} Error: {response.text}")
                resp_data = response.json()
                result = {"prediction": self._parse_normal_response(resp_data)}
                if cot and self.cot_parser:
                    generated_text = result["prediction"]
                    cot_text, final_answer = self.cot_parser(generated_text)
                    result["chain_of_thought"] = cot_text
                    result["prediction"] = final_answer
                logger.info("HTTP request succeeded.")
                return result
            except Exception as e:
                logger.error(f"HTTP attempt {attempt + 1}/{effective_retries} failed: {e}")
                attempt += 1
                await asyncio.sleep(min(2 ** attempt, 32))
        raise RuntimeError(f"Failed after {effective_retries} retries via httpx.")

    async def _generate_batch_httpx(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool,
        until: Optional[Union[str, List[str]]],
        cot: bool,
        max_retries: Optional[int],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously generates outputs for a batch of input items using httpx.
        """
        logger.info(f"Starting asynchronous HTTP batch generation for {len(inputs)} items.")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._send_single_request_httpx(
                    client, item, return_logits, until, cot=cot, max_retries=max_retries, **kwargs
                )
                for item in inputs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        merged_results = []
        for orig, res in zip(inputs, results):
            merged = deepcopy(orig)
            merged.update(res)
            merged_results.append(merged)
        logger.info("Asynchronous HTTP batch generation completed.")
        return merged_results

    def _generate_single_sdk(
        self,
        item: Dict[str, Any],
        return_logits: bool,
        until: Optional[Union[str, List[str]]],
        cot: bool = False,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates text for a single input item using the OpenAI SDK client.
        Implements retry logic with exponential backoff.
        """
        effective_retries = max_retries if max_retries is not None else self.max_retries
        logger.info(f"Starting SDK request for input: {item['input']}")
        for attempt in range(effective_retries):
            try:
                payload = self._create_payload(item["input"], cot=cot, until=until, **kwargs)
                if not self.use_chat_api:
                    response = self.client.completions.create(**payload)
                    result = {
                        "prediction": response.choices[0].text,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                    if return_logits:
                        result.update({
                            "logprobs": response.choices[0].logprobs.token_logprobs,
                            "tokens": response.choices[0].logprobs.tokens,
                        })
                else:
                    response = self.client.chat.completions.create(**payload)
                    result = {
                        "prediction": response.choices[0].message.content,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                    if return_logits and hasattr(response.choices[0], "logprobs"):
                        result["logprobs"] = response.choices[0].logprobs
                if cot and self.cot_parser:
                    generated_text = result["prediction"]
                    cot_text, final_answer = self.cot_parser(generated_text)
                    result["chain_of_thought"] = cot_text
                    result["prediction"] = final_answer
                logger.info("SDK request succeeded.")
                return result
            except Exception as e:
                logger.error(f"SDK attempt {attempt + 1}/{effective_retries} failed: {e}")
                time.sleep(min(2 ** attempt, 32))
        raise RuntimeError(f"Failed after {effective_retries} SDK retries.")

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        until: Optional[Union[str, List[str]]] = None,
        cot: bool = False,
        max_retries: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generates text for a batch of input items.

        If the model is not a vision model, uses asynchronous httpx calls.
        Otherwise, uses the OpenAI SDK client via a ThreadPoolExecutor.
        """
        logger.info(f"Starting batch generation for {len(inputs)} items.")
        if not self.is_vision_model:
            # Use httpx-based asynchronous generation for text models.
            try:
                results = asyncio.run(
                    self._generate_batch_httpx(inputs, return_logits, until, cot, max_retries, **kwargs)
                )
            except RuntimeError as e:
                if "asyncio.run() cannot be called from a running event loop" in str(e):
                    import nest_asyncio
                    nest_asyncio.apply()
                    results = asyncio.run(
                        self._generate_batch_httpx(inputs, return_logits, until, cot, max_retries, **kwargs)
                    )
                else:
                    raise
        else:
            # Use the OpenAI SDK client (synchronous) for vision models.
            results = []
            max_workers = self.batch_size
            future_to_item = {}
            def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
                input_copy = deepcopy(item)
                return self._generate_single_sdk(input_copy, return_logits, until, cot=cot, max_retries=max_retries, **kwargs)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for item in inputs:
                    future = executor.submit(process_item, item)
                    future_to_item[future] = deepcopy(item)
                for future in tqdm(as_completed(future_to_item), total=len(inputs), desc="Generating SDK outputs", disable=not show_progress):
                    orig_item = future_to_item[future]
                    try:
                        res = future.result()
                        merged = deepcopy(orig_item)
                        merged.update(res)
                        results.append(merged)
                    except Exception as e:
                        logger.error(f"SDK error: {str(e)}")
                        error_item = deepcopy(orig_item)
                        error_item.update({
                            "error": str(e),
                            "prediction": None,
                            "finish_reason": "error"
                        })
                        results.append(error_item)
        logger.info("Batch generation completed.")
        return results
