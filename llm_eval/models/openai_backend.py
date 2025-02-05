import openai
import time
import base64
import logging
from typing import List, Dict, Any, Optional, Union
from copy import deepcopy
from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger
import logging
from tqdm import tqdm

logger = get_logger(name="runner", level=logging.INFO)

@register_model("openai")
class OpenAIModel(BaseModel):
    """
    An implementation of a Vision Language Model based on the OpenAI API, which processes inputs that include text and images.

    For example, it can process a sample like:
        {
            "input": {
                "content": [
                    {"type": "text", "text": "What can you see in this image?"},
                    {"type": "image_url", "image_url": "https://example.com/image.jpg"}
                ]
            }
        }

    Or a base64 encoded image:
        {
            "input": {
                "content": [
                    {"type": "text", "text": "Please describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,...", "detail": "high"}
                    }
                ]
            }
        }

    Args:
        api_key (str): OpenAI API key
        api_base (str): API base URL (default: OpenAI endpoint)
        model_name (str): Model identifier (e.g., gpt-4-vision-preview)
        system_message (Optional[str]): System message for chat completion
        use_chat_api (bool): Whether to use the chat API (default: True)
        is_vision_model (bool): Whether the model is a vision model (default: False)
        limit_mm_per_prompt (Optional[Dict[str, int]]): Multimedia limits per prompt
        **kwargs: Additional API parameters (temperature, max_tokens, etc.)
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model_name: str = str,  # e.g., gpt-4o-mini, o1, o1-mini, etc.
        system_message: Optional[str] = None,
        use_chat_api: bool = True,
        is_vision_model: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__()
        if not model_name:
            raise ValueError("model_name is required")

        # Set up an independent API client for each instance
        self._client = openai.Client(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.system_message = system_message
        self.limit_mm_per_prompt = limit_mm_per_prompt or {}

        # Set API type and vision model flags based on initialization parameters
        self.use_chat_api = use_chat_api
        self.is_vision_model = is_vision_model

        # Set default API parameters
        self.default_params = kwargs

    def _process_image_content(self, content: Union[str, Dict, List]) -> Dict:
        """Converts image content into the OpenAI Vision API format."""
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
            max_images = self.limit_mm_per_prompt.get("image", float("inf"))
            if len(content) > max_images:
                raise ValueError(
                    f"Number of images ({len(content)}) exceeds limit ({max_images})"
                )
            return [
                self._process_image_content(item) for item in content
            ]  # Recursive processing

        if isinstance(content, str):
            if content.startswith(("http://", "https://")):
                return {
                    "type": "image_url",
                    "image_url": {"url": content, "detail": "auto"},
                }
            try:
                return {
                    "type": "image_url",
                    "image_url": {"url": process_base64(content), "detail": "auto"},
                }
            except:
                return {"type": "text", "text": content}

        elif isinstance(content, dict):
            detail = validate_detail(content.get("detail", "auto"))

            if "image_url" in content:
                if isinstance(content["image_url"], str):
                    return {
                        "type": "image_url",
                        "image_url": {"url": content["image_url"], "detail": detail},
                    }
                return {
                    "type": "image_url",
                    "image_url": {**content["image_url"], "detail": detail},
                }
            elif "base64" in content:
                mime_type = content.get("mime_type", "image/jpeg")
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": process_base64(content["base64"], mime_type),
                        "detail": detail,
                    },
                }

        return {"type": "text", "text": str(content)}

    def _create_payload(
        self,
        inputs: Union[str, List[Dict], Dict],
        return_logits: bool = False,
        use_chat_api: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        params = self.default_params.copy()
        params.update(kwargs)
        use_chat_api = self.use_chat_api if use_chat_api is None else use_chat_api

        payload = {"model": self.model_name}

        if use_chat_api:
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})

            # Process vision or general inputs
            if self.is_vision_model and isinstance(inputs, dict):
                content = inputs.get("content", [])
                if isinstance(content, list):
                    # Check image count limit
                    image_count = sum(
                        1
                        for item in content
                        if isinstance(item, dict)
                        and ("image_url" in item or "base64" in item)
                    )
                    max_images = self.limit_mm_per_prompt.get("image", float("inf"))
                    if image_count > max_images:
                        raise ValueError(
                            f"Number of images ({image_count}) exceeds limit ({max_images})"
                        )

                    processed_content = [
                        (
                            self._process_image_content(item)
                            if isinstance(item, dict)
                            and ("image_url" in item or "base64" in item)
                            else {"type": "text", "text": str(item)}
                        )
                        for item in content
                    ]
                    messages.append({"role": "user", "content": processed_content})
                else:
                    messages.append({"role": "user", "content": content})
            elif isinstance(inputs, str):
                messages.append({"role": "user", "content": inputs})
            elif isinstance(inputs, list):
                for msg in inputs:
                    if isinstance(msg, dict):
                        if "role" not in msg:
                            msg = {"role": "user", **msg}
                        messages.append(msg)
                    else:
                        messages.append({"role": "user", "content": str(msg)})
            else:
                messages.append({"role": "user", "content": str(inputs)})

            payload["messages"] = messages
        else:
            payload.update(
                {
                    "prompt": inputs,
                    "logprobs": params.get("logprobs") if return_logits else None,
                }
            )

        # Set additional parameters
        for param in [
            "max_tokens",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]:
            if param in params:
                payload[param] = params[param]

        return {k: v for k, v in payload.items() if v is not None}

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        use_chat_api: Optional[bool] = None,
        raise_error: bool = True,
        max_retries: int = 3,
        cot: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Performs batch text generation.

        Args:
            inputs: [{"input": str|dict, "reference": str, ...}, ...]
                    If input is a dict, it can include multimodal data.
                    For example: {
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "..."}}
                        ]
                    }
        """
        outputs = []
        logger.info("generating output")
        for input_item in inputs:
            # Create a copy of the input to preserve the original
            item = deepcopy(input_item)
            result = None

            for attempt in range(max_retries):
                try:
                    payload = self._create_payload(
                        item["input"],
                        return_logits=return_logits,
                        use_chat_api=use_chat_api,
                        **kwargs,
                    )

                    if not use_chat_api:
                        response = self._client.completions.create(**payload)
                        result = {
                            "prediction": response.choices[0].text,
                            "finish_reason": response.choices[0].finish_reason,
                        }
                        if return_logits:
                            result.update(
                                {
                                    "logprobs": response.choices[0].logprobs.token_logprobs,
                                    "tokens": response.choices[0].logprobs.tokens,
                                }
                            )
                    else:
                        response = self._client.chat.completions.create(**payload)
                        result = {
                            "prediction": response.choices[0].message.content,
                            "finish_reason": response.choices[0].finish_reason,
                        }
                        if return_logits and hasattr(response.choices[0], "logprobs"):
                            result["logprobs"] = response.choices[0].logprobs

                    break

                except Exception as e:
                    if attempt == max_retries - 1:
                        error_msg = f"Error after {max_retries} attempts: {str(e)}"
                        if raise_error:
                            raise RuntimeError(error_msg)
                        result = {
                            "error": error_msg,
                            "prediction": None,
                            "finish_reason": "error",
                        }
                    time.sleep(min(2**attempt, 32))  # Exponential backoff

            # Copy the result into a new dictionary
            output_item = deepcopy(item)
            output_item.update(
                result
                or {
                    "error": "Failed to generate",
                    "prediction": None,
                    "finish_reason": "error",
                }
            )
            outputs.append(output_item)

        return outputs
