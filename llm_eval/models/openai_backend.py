import openai
import time
import base64
import logging
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger

logger = get_logger(name="openai_backend", level=logging.INFO)

@register_model("openai")
class OpenAIModel(BaseModel):
    """
    A production-grade OpenAI API backend model that supports multimodal input,
    chain-of-thought (CoT) prompting, and concurrent batch processing using multi-threading.
    
    This implementation handles both Chat and Completions API calls based on the use_chat_api flag.
    It appends a CoT trigger if enabled and uses a CoT parser to extract a chain-of-thought and final answer.
    
    Key Features:
      - Constructs payloads for both Chat and traditional Completion API calls.
      - Processes image inputs by converting URLs or base64-encoded images to the required format.
      - Implements robust retry logic with exponential backoff.
      - Executes API calls concurrently using a ThreadPoolExecutor; the number of worker threads is set based on the batch_size.
      - Rich error handling and detailed logging for production monitoring.
    
    Args:
        api_key (str): OpenAI API key.
        api_base (str): OpenAI API base URL.
        model_name (str): Model identifier (e.g., "gpt-4", "gpt-4-vision-preview").
        system_message (Optional[str]): System message for chat completions.
        use_chat_api (bool): Whether to use the Chat API; if False, uses the Completions API.
        is_vision_model (bool): Flag indicating if the model supports vision inputs.
        cot_trigger (Optional[str]): A trigger phrase to induce chain-of-thought; if provided, appended to the prompt.
        cot_parser (Optional[Callable[[str], Tuple[str, str]]]): Function that parses generated text into (chain_of_thought, final_answer).
        batch_size (int): Number of concurrent API calls (worker threads) for batch processing.
        max_retries (int): Maximum number of retry attempts for API calls.
        **kwargs: Additional API parameters (e.g., temperature, top_p, max_tokens, etc.).
    """
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model_name: str,
        system_message: Optional[str] = None,
        use_chat_api: bool = True,
        is_vision_model: bool = False,
        cot_trigger: Optional[str] = "Let's think step by step.",
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = None,
        batch_size: int = 8,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not model_name:
            raise ValueError("model_name is required")
        if not api_key:
            raise ValueError("api_key is required")
        if not api_base:
            raise ValueError("api_base is required")
        
        # Set up OpenAI API credentials
        openai.api_key = api_key
        openai.api_base = api_base
        
        self.model_name = model_name
        self.system_message = system_message
        self.use_chat_api = use_chat_api
        self.is_vision_model = is_vision_model
        
        # Chain-of-thought settings
        self.cot_trigger = cot_trigger
        self.cot_parser = cot_parser
        
        # Batch and retry settings
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Default API parameters (e.g., temperature, max_tokens, top_p, etc.)
        self.default_params = kwargs

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
        return_logits: bool = False,
        use_chat_api: Optional[bool] = None,
        until: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Constructs the payload for an API call.
        
        Supports both Chat and Completions APIs. If chain-of-thought (CoT) is enabled,
        the CoT trigger is appended to the prompt.
        """
        params = self.default_params.copy()
        params.update(kwargs)
        use_chat_api = self.use_chat_api if use_chat_api is None else use_chat_api
        
        payload = {"model": self.model_name}
        
        # Add stop sequences if provided
        if until is not None:
            if isinstance(until, str):
                until = [until]
            payload["stop"] = until
        
        if use_chat_api:
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            # Process input: append CoT trigger if enabled
            if isinstance(inputs, str):
                prompt_text = inputs
                if self.cot and self.cot_trigger:
                    prompt_text = f"{inputs}\n{self.cot_trigger}\n"
                messages.append({"role": "user", "content": prompt_text})
            elif isinstance(inputs, list):
                for msg in inputs:
                    if isinstance(msg, dict):
                        if "role" not in msg:
                            msg = {"role": "user", **msg}
                        messages.append(msg)
                    else:
                        messages.append({"role": "user", "content": str(msg)})
            elif isinstance(inputs, dict):
                if self.is_vision_model:
                    content = inputs.get("content", [])
                    processed_content = []
                    for item in content:
                        if isinstance(item, dict) and ("image_url" in item or "base64" in item):
                            processed_content.append(self._process_image_content(item))
                        else:
                            processed_content.append({"type": "text", "text": str(item)})
                    messages.append({"role": "user", "content": json.dumps(processed_content)})
                else:
                    messages.append({"role": "user", "content": str(inputs)})
            else:
                messages.append({"role": "user", "content": str(inputs)})
            payload["messages"] = messages
        else:
            # For the Completions API, use the prompt field.
            prompt_text = inputs if not (self.cot and self.cot_trigger) else f"{inputs}\n{self.cot_trigger}\n"
            payload["prompt"] = prompt_text
            payload["logprobs"] = params.get("logprobs") if return_logits else None
        
        # Set additional API parameters (e.g., max_tokens, temperature, top_p, etc.)
        for param in ["max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if param in params:
                payload[param] = params[param]
        
        # Remove keys with None values
        return {k: v for k, v in payload.items() if v is not None}
    
    def _generate_single(
        self,
        input_item: Dict[str, Any],
        return_logits: bool,
        until: Optional[Union[str, List[str]]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates text for a single input item with retry logic and exponential backoff.
        
        Args:
            input_item (Dict[str, Any]): An input item containing at least "input".
            return_logits (bool): Whether to return logits.
            until (Optional[Union[str, List[str]]]): Optional stopping criteria.
            **kwargs: Additional parameters to pass to the API.
        
        Returns:
            Dict[str, Any]: A dictionary with keys "prediction", "finish_reason", and optionally "logprobs".
        """
        result = None
        for attempt in range(self.max_retries):
            try:
                payload = self._create_payload(
                    input_item["input"],
                    return_logits=return_logits,
                    until=until,
                    cot=kwargs.get("cot", False),
                    **kwargs,
                )
                if not self.use_chat_api:
                    # Call the traditional completions API
                    response = openai.Completion.create(**payload)
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
                    # Call the Chat API
                    response = openai.ChatCompletion.create(**payload)
                    result = {
                        "prediction": response.choices[0].message.content,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                    if return_logits and hasattr(response.choices[0], "logprobs"):
                        result["logprobs"] = response.choices[0].logprobs
                break  # Exit retry loop if successful
            except Exception as e:
                if attempt == self.max_retries - 1:
                    error_msg = f"Error after {self.max_retries} attempts: {str(e)}"
                    raise RuntimeError(error_msg) from e
                else:
                    # Exponential backoff
                    time.sleep(min(2 ** attempt, 32))
        return result
    
    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        use_chat_api: Optional[bool] = None,
        until: Optional[Union[str, List[str]]] = None,
        cot: bool = False,
        max_retries: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generates text for a batch of input items using multi-threading.
        
        This method processes each input item concurrently using a ThreadPoolExecutor.
        It supports chain-of-thought prompting and parsing. In case of API failures,
        each call is retried with exponential backoff.
        
        Args:
            inputs (List[Dict[str, Any]]): List of input items. Each item must include "input" and optionally "reference".
            return_logits (bool): If True, include logits in the output (if supported).
            use_chat_api (Optional[bool]): Overrides the instance's use_chat_api flag if provided.
            until (Optional[Union[str, List[str]]]): Stop sequence(s) for generation.
            cot (bool): If True, enable chain-of-thought processing.
            max_retries (int | None): Maximum retry attempts for each API call (defaults to self.max_retries).
            show_progress (bool): If True, display a progress bar.
            **kwargs: Additional API parameters.
        
        Returns:
            List[Dict[str, Any]]: The list of input items updated with generation results.
                Each item will have "prediction" and "finish_reason" keys, and if CoT is enabled,
                "chain_of_thought" will be added.
        """
        if max_retries is None:
            max_retries = self.max_retries

        results = []
        # Use the batch_size as the number of worker threads.
        max_workers = self.batch_size

        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            # Create a deep copy to preserve original input
            input_copy = deepcopy(item)
            result = self._generate_single(input_copy, return_logits, until, cot=cot, **kwargs)
            # If chain-of-thought is enabled and a parser is provided, parse the generated text.
            if result and result.get("prediction") is not None and cot and self.cot_parser:
                generated_text = result["prediction"]
                cot_text, final_answer = self.cot_parser(generated_text)
                result["chain_of_thought"] = cot_text
                result["prediction"] = final_answer
            return result or {
                "error": "Failed to generate",
                "prediction": None,
                "finish_reason": "error",
            }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in inputs}
            if show_progress:
                for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Generating OpenAI outputs"):
                    orig_item = future_to_item[future]
                    try:
                        res = future.result()
                        merged = deepcopy(orig_item)
                        merged.update(res)
                        results.append(merged)
                    except Exception as e:
                        logger.error(f"Error in API call: {str(e)}")
                        merged = deepcopy(orig_item)
                        merged.update({
                            "error": str(e),
                            "prediction": None,
                            "finish_reason": "error"
                        })
                        results.append(merged)
            else:
                for future in as_completed(future_to_item):
                    orig_item = future_to_item[future]
                    try:
                        res = future.result()
                        merged = deepcopy(orig_item)
                        merged.update(res)
                        results.append(merged)
                    except Exception as e:
                        logger.error(f"Error in API call: {str(e)}")
                        merged = deepcopy(orig_item)
                        merged.update({
                            "error": str(e),
                            "prediction": None,
                            "finish_reason": "error"
                        })
                        results.append(merged)
        return results
