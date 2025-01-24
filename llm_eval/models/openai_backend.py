import openai
import time
from typing import List, Dict, Any, Optional, Union
from .base import BaseModel
from . import register_model

@register_model("openai")
class OpenAIModel(BaseModel):
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",  # gpt-4o-mini, o1, o1-mini 등
        system_message: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        if not api_key:
            raise ValueError("API key is required")

        self._client = openai.Client(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.system_message = system_message
        self.default_params = kwargs

    def _create_payload(
        self,
        inputs: Union[str, List[Dict]],
        return_logits: bool = False,
        use_chat_api: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        params = self.default_params.copy()
        params.update(kwargs)

        payload = {"model": self.model_name}

        if not use_chat_api:
            payload = {"model": self.model_name, "prompt": inputs, **params}
            if return_logits:
                payload["logprobs"] = 5

        else:
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            if isinstance(inputs, str):
                messages.append({"role": "user", "content": inputs})
            else:
                messages.extend(inputs)
            payload["messages"] = messages

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
        use_chat_api: bool = True,
        return_logits: bool = False,
        raise_error: bool = False,
        max_retries: int = 3,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        outputs = []

        for input_item in inputs:
            item = input_item.copy()
            result = None

            for attempt in range(max_retries):
                try:
                    payload = self._create_payload(
                        item["input"],
                        return_logits=return_logits,
                        **kwargs,
                    )

                    if not use_chat_api:
                        response = self._client.completions.create(**payload)
                        result = {
                            "prediction": response.choices[0].text,
                            "model_name": "openai"
                        }
                        if return_logits:
                            result.update({
                                "logprobs": response.choices[0].logprobs.token_logprobs,
                                "tokens": response.choices[0].logprobs.tokens,
                            })
                    else:
                        response = self._client.chat.completions.create(**payload)
                        result = {
                            "prediction": response.choices[0].message.content,
                            "model_name": "openai"
                        }
                    if return_logits and hasattr(response.choices[0], "logprobs"):
                        result["logprobs"] = response.choices[0].logprobs

                    break

                except Exception as e:
                    if attempt == max_retries - 1:
                        if raise_error:
                            raise
                        result = {"error": str(e), "prediction": None, "model_name": "openai"}
                    else:
                        time.sleep(1 * (attempt + 1))

            outputs.append(
                item | (result or {"error": "Failed to generate", "prediction": "", "model_name": "openai"})
            )

        return outputs
