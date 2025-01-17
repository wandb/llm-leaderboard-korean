import logging
from typing import List, Dict, Any, Optional
import litellm

from .base import BaseModel

logger = logging.getLogger(__name__)

class LiteLLMBackend(BaseModel):
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
        **kwargs
    ):
        super().__init__(
            backend_type=provider,
            model_name=model_name,
        )
        
        # 모든 설정값을 클래스 속성으로 저장
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.anthropic_api_key = anthropic_api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.extra_kwargs = kwargs

        # API 키 설정
        if provider == "openai":
            litellm.openai_key = api_key
            if api_base:
                litellm.api_base = api_base
        elif provider == "anthropic":
            litellm.anthropic_key = anthropic_api_key
        elif provider == "bedrock":
            litellm.aws_access_key_id = aws_access_key_id
            litellm.aws_secret_access_key = aws_secret_access_key
        elif provider == "azure":
            litellm.azure_key = api_key
            litellm.azure_endpoint = api_base

    def _prepare_completion_kwargs(self, prompt: str) -> Dict[str, Any]:
        """LiteLLM completion 함수에 전달할 인자 준비"""
        completion_kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            **self.extra_kwargs
        }
        
        if self.provider == "huggingface":
            completion_kwargs["model"] = f"{self.model_name}"
        elif self.provider == "azure":
            completion_kwargs.update({
                "model": self.model_name,
                "engine": self.model_name,
            })
        elif self.provider == "bedrock":
            completion_kwargs["model"] = f"bedrock/{self.model_name}"
        else:
            completion_kwargs["model"] = self.model_name
            
        logger.debug(f"Prepared completion kwargs: {completion_kwargs}")
        return completion_kwargs

    def generate_batch(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        
        for idx, item in enumerate(input_list):
            prompt = item["input"]
            reference = item.get("reference", "")

            logger.info(
                f"[LiteLLMBackend] Generating text for item {idx+1}/{len(input_list)}: {prompt[:50]}..."
            )

            try:
                completion_kwargs = self._prepare_completion_kwargs(prompt)
                response = litellm.completion(**completion_kwargs)
                prediction = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error generating completion: {str(e)}")
                prediction = f"Error: {str(e)}"

            results.append({
                "input": prompt,
                "reference": reference,
                "prediction": prediction
            })

        return results