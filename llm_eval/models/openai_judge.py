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

import re
import random

import nest_asyncio

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
        api_key (Optional[str]): API key for authentication.
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
        
    Examples:
        ```python
        # 1. OpenAI API 사용 예시
        judge = OpenAIJudge(
            api_base="https://api.openai.com/v1",  # 기본 URL만 지정 (엔드포인트는 자동 추가됨)
            model_name="gpt-4",
            api_key="sk-...",  # OpenAI API 키 지정
            system_message="You are a helpful judge that evaluates model outputs."
        )
        
        # 2. 자체 호스팅 API 서버 사용 예시
        judge = OpenAIJudge(
            api_base="http://localhost:8000",  # vLLM 또는 다른 OpenAI 호환 서버
            model_name="llama-3-8b",
            use_chat_api=True
        )
        
        # 3. 평가 실행
        results = judge.judge_batch([
            {"input": "Judge the following response: ..."},
            {"input": "Evaluate this answer: ..."}
        ])
        ```
    """
    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        use_chat_api: bool = True,
        max_tokens: int = 64,
        temperature: float = 0.1,
        top_p: float = 0.95,
        do_sample: bool = True,
        batch_size: int = 32,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not model_name:
            raise ValueError("model_name is required for OpenAIJudge.")
        if not api_base:
            raise ValueError("api_base is required for OpenAIJudge.")
        
        # Judge API endpoint is specified via api_base; API key is optional.
        self.api_base = api_base
        self.api_key = api_key
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
        
        # Determine the appropriate endpoint URL based on the API type
        api_url = self.api_base
        if not (api_url.endswith('/chat/completions') or api_url.endswith('/completions')):
            # Add the appropriate endpoint if not already included
            if self.use_chat_api:
                if not api_url.endswith('/'):
                    api_url += '/'
                if not api_url.endswith('v1/'):
                    api_url += 'v1/' if 'v1/' not in api_url else ''
                api_url += 'chat/completions'
            else:
                if not api_url.endswith('/'):
                    api_url += '/'
                if not api_url.endswith('v1/'):
                    api_url += 'v1/' if 'v1/' not in api_url else ''
                api_url += 'completions'
            
        # Prepare headers
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        while attempt <= effective_retries:
            try:
                response = await client.post(api_url, json=payload, headers=headers, timeout=self.timeout)
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code} Error: {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                resp_data = response.json()
                
                # Parse response: expected format similar to OpenAI ChatCompletion
                if self.use_chat_api:
                    try:
                        content = resp_data["choices"][0]["message"]["content"]
                        finish_reason = resp_data["choices"][0].get("finish_reason", "")
                    except (KeyError, IndexError):
                        content = json.dumps(resp_data, indent=2)
                        finish_reason = ""
                else:
                    try:
                        content = resp_data["choices"][0]["text"]
                        finish_reason = resp_data["choices"][0].get("finish_reason", "")
                    except (KeyError, IndexError):
                        content = json.dumps(resp_data, indent=2)
                        finish_reason = ""
                
                # Parse score from content if available
                score_pattern = r"\[RESULT\]\s*(\d+(?:\.\d+)?)"
                score_match = None
                try:
                    score_match = re.search(score_pattern, content)
                except:
                    pass
                
                score = float(score_match.group(1)) if score_match else None
                
                result = {
                    "prediction": content,
                    "finish_reason": finish_reason,
                }
                
                # score가 있을 때만 judge_score 필드 추가
                if score is not None:
                    result["judge_score"] = score
                return result
            except Exception as e:
                logger.error(f"HTTP attempt {attempt + 1}/{effective_retries} failed: {e}")
                attempt += 1
                # Add random jitter to backoff time to prevent thundering herd problem
                
                jitter = random.uniform(0, 1)
                backoff_time = min(2 ** attempt + jitter, 32)
                logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                await asyncio.sleep(backoff_time)
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
            # 배치 크기 증가 - 더 많은 요청을 병렬로 처리
            batch_size = min(self.batch_size, 32)  # 16에서 32로 증가
            logger.info(f"Processing {len(inputs)} items in batches of {batch_size}")
            
            all_results = []
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(inputs) + batch_size - 1)//batch_size}")
                
                # Process current batch
                tasks = []
                for item in batch:
                    prompt = item["input"]
                    tasks.append(self._send_single_request(client, prompt, until=until, **kwargs))
                
                # Wait for all requests in this batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for orig, res in zip(batch, batch_results):
                    merged = deepcopy(orig)
                    if isinstance(res, Exception):
                        logger.error(f"Error processing judge prompt: {str(res)}")
                        merged.update({
                            "prediction": f"Error: {str(res)}",
                            "finish_reason": "error"
                        })
                    else:
                        merged.update(res)
                    all_results.append(merged)
                
                # 배치 간 대기 시간 최소화
                if i + batch_size < len(inputs):
                    await asyncio.sleep(0.2)  # 0.5초에서 0.2초로 감소
            
            return all_results

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
        # Apply nest_asyncio preemptively to avoid nested event loop issues
        try:
            nest_asyncio.apply()
        except ImportError:
            pass
            
        # Now run the async function with the configured event loop
        return asyncio.run(self._generate_batch_async(inputs, until, **kwargs))