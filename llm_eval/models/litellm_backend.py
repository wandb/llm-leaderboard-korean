import logging
from typing import List, Dict, Any, Optional, Union
import time
import litellm
from tqdm import tqdm

from . import register_model
from .base import BaseModel

logger = logging.getLogger(__name__)

@register_model("litellm")
class LiteLLMBackend(BaseModel):
    """
    Backend implementation for various LLM Providers using LiteLLM

    Attributes:
        provider (str): LLM provider (e.g., openai, anthropic, bedrock, azure, etc.)
        model_name (str): Name of the model to use
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Degree of randomness in generated text (0.0 ~ 1.0)
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
        **kwargs
    ):
        """
        Initialize the LiteLLM backend

        Args:
            provider: Name of the LLM provider
            model_name: Name of the model to use
            api_key: API key (for OpenAI, Azure)
            api_base: Base URL for the API (for OpenAI, Azure)
            aws_access_key_id: AWS access key ID (for Bedrock)
            aws_secret_access_key: AWS secret key (for Bedrock)
            anthropic_api_key: Anthropic API key
            max_new_tokens: Maximum number of tokens to generate
            temperature: Degree of randomness in the generated text
            **kwargs: Additional configuration parameters
        """

        super().__init__(**kwargs)
        
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.extra_kwargs = kwargs

        # Configure completion_kwargs for each LiteLLM instance
        self.completion_kwargs = {
            "api_key": api_key,
            "api_base": api_base,
        }
        
        if provider == "bedrock":
            self.completion_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
            })
        elif provider == "anthropic":
            self.completion_kwargs["api_key"] = anthropic_api_key

    def _prepare_completion_kwargs(self, prompt: str, until: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Prepare the arguments to pass to the LiteLLM completion function

        Args:
            prompt: Input prompt text

        Returns:
            Dict[str, Any]: Dictionary of arguments for the LiteLLM completion function
        """
        # Basic completion settings
        completion_kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            **self.completion_kwargs,  # Add instance-specific API settings
            **self.extra_kwargs
        }
        
        # Add stop sequences if provided
        if until is not None:
            completion_kwargs["stop"] = until if isinstance(until, list) else [until]

        # Set model name format based on provider
        if self.provider == "azure":
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

    def _generate_with_retry(
        self, 
        completion_kwargs: Dict[str, Any], 
        max_attempts: int = 3,
        initial_wait: int = 4
    ) -> str:
        """
        Generate text with retry logic

        Args:
            completion_kwargs: Arguments to pass to the LiteLLM completion function
            max_attempts: Maximum number of attempts
            initial_wait: Wait time before the first retry (in seconds)

        Returns:
            str: Generated text

        Raises:
            Exception: If all retry attempts fail
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
                    # Exponential backoff: 4s, 8s, 16s...
                    wait_time = initial_wait * (2 ** (attempt - 1))
                    logger.warning(
                        f"Attempt {attempt} failed with error: {str(e)}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
        
        error_msg = f"All {max_attempts} attempts failed. Last error: {str(last_exception)}"
        logger.error(error_msg)
        raise last_exception or Exception(error_msg)

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        cot: bool = False,
        batch_size: Optional[Union[int, str]] = 1,
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,  # 진행바 표시 여부를 결정하는 인자 추가
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform batch text generation

        Args:
            inputs (List[Dict[str, Any]]): List of input data. Each item should include:
                - input (str): Input text/prompt
                - reference (str): Reference text (optional)
            return_logits (bool, optional): Whether to return log probabilities. Defaults to False.
                If True, each item will include a "logits" field.
            cot (bool, optional): Whether to use Chain-of-Thought. Defaults to False.
                If True, a CoT trigger will be added to the input prompt.
            batch_size (Optional[Union[int, str]], optional): Batch size. Defaults to 1.
                - "auto": Automatically adjust the batch size
                - int: Use a fixed batch size
                - None: Use the length of inputs as the batch size

        Returns:
            List[Dict[str, Any]]: List of generation results. Each item includes:
                - input (str): Original input text
                - reference (str): Original reference text
                - prediction (str): Generated text
                - logits (dict, optional): Log probability information if return_logits=True
                - chain_of_thought (str, optional): Inference process if cot=True
        
        Raises:
            NotImplementedError: If return_logits=True is specified, as it is not supported yet
        """

        if return_logits:
            raise NotImplementedError("LiteLLM backend does not support logits calculation yet")

        results = []
        batch_size = len(inputs) if batch_size is None else batch_size
        
        # Convert string batch_size to int if "auto"
        if isinstance(batch_size, str) and batch_size.lower() == "auto":
            batch_size = len(inputs)
            logger.info(f"[LiteLLMBackend] Using auto batch size: {batch_size}")
        
            logger.info(f"[LiteLLMBackend] Generating text...")
        # Process in batches
        for start_idx in tqdm(range(0, len(inputs), batch_size), disable = not show_progress):
            batch_items = inputs[start_idx:start_idx + batch_size]
            
            for idx, item in enumerate(batch_items):
                prompt = item["input"]
                reference = item.get("reference", "")
                
                # Add CoT trigger if enabled
                if cot and self.cot_trigger:
                    prompt = f"{prompt}\n{self.cot_trigger}"

                

                try:
                    completion_kwargs = self._prepare_completion_kwargs(prompt, until=until)
                    
                    # Generate response
                    prediction = self._generate_with_retry(completion_kwargs)
                    
                    # Prepare result item
                    result_item = {
                        "input": item["input"],
                        "reference": reference,
                        "prediction": prediction
                    }
                    
                    # Parse CoT if enabled and parser is available
                    if cot and self.cot_parser:
                        chain_of_thought, final_answer = self.cot_parser(prediction)
                        result_item.update({
                            "chain_of_thought": chain_of_thought,
                            "prediction": final_answer
                        })
                    
                    results.append(result_item)

                except Exception as e:
                    logger.error(f"Error generating completion after retries: {str(e)}")
                    results.append({
                        "input": prompt,
                        "reference": reference,
                        "prediction": f"Error: {str(e)}"
                    })

        return results
