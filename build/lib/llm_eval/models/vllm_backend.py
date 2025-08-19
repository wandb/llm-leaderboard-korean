import logging
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import copy

# Attempt to import vllm, set a flag, and raise ImportError later if not available.
try:
    import vllm
    # Import the necessary parameter classes from vllm
    from vllm.sampling_params import SamplingParams, GuidedDecodingParams
    VLLM_AVAILABLE = True
except ImportError:
    vllm = None
    SamplingParams = None
    GuidedDecodingParams = None
    VLLM_AVAILABLE = False

from tqdm import tqdm

from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger
from llm_eval.utils.prompt_template import default_cot_parser

logger = get_logger(name="vllm_backend", level=logging.INFO)

@register_model("vllm")
class VLLMModel(BaseModel):
    # This class uses vLLM for model inference.
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        cot: bool = False,
        cot_trigger: Optional[str] = "Let's think step by step.",
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = default_cot_parser,
        **kwargs,
    ):
        super().__init__(cot=cot, cot_trigger=cot_trigger, cot_parser=cot_parser, **kwargs)
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed.")

        self.model_name = f"vllm:{model_name_or_path}"
        logger.info(f"Initializing vLLM engine for model: {model_name_or_path}")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop if stop is not None else []
        self.sampling_kwargs = kwargs
        self.cot = cot

        try:
            llm_init_valid_args = [
                "tokenizer", "tokenizer_mode", "skip_tokenizer_init", "tokenizer_revision", "trust_remote_code",
                "revision", "code_revision", "rope_scaling", "rope_theta", "seed", "quantization", "enforce_eager",
                "max_model_len", "swap_space", "kv_cache_dtype", "block_size", "worker_use_ray", "pipeline_parallel_size",
                "enable_prefix_caching", "disable_custom_all_reduce", "max_num_batched_tokens", "max_num_seqs",
                "max_paddings", "num_gpu_blocks_override", "load_format", "engine_use_ray", "disable_log_stats",
                "disable_log_requests"
            ]
            llm_init_kwargs = {k: v for k, v in kwargs.items() if k in llm_init_valid_args}
            self.llm = vllm.LLM(
                model=model_name_or_path, dtype=dtype, tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization, **llm_init_kwargs
            )
            self.tokenizer = self.llm.get_tokenizer()
            logger.info("vLLM engine and tokenizer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}", exc_info=True)
            raise e

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        batch_size: Optional[int] = None,
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not self.llm:
            raise RuntimeError("vLLM engine is not initialized.")
            
        mcqa_items = []
        mcqa_indices = []
        normal_gen_items = []
        normal_gen_indices = []

        # 1. Separate inputs into MCQA and normal generation tasks.
        for i, item in enumerate(inputs):
            is_mcqa = "options" in item and isinstance(item["options"], list) and item["options"]
            if is_mcqa:
                mcqa_items.append(item)
                mcqa_indices.append(i)
            else:
                normal_gen_items.append(item)
                normal_gen_indices.append(i)
        
        all_processed_items = {}

        # 2. Process MCQA items using Guided Decoding.
        if mcqa_items:
            for i, item in enumerate(tqdm(mcqa_items, desc="Generating MCQA choices")):
                original_index = mcqa_indices[i]
                prompt = [item['input']]
                options = item['options']
                
                # --- CORRECT IMPLEMENTATION BASED ON USER'S EXAMPLE ---
                # 1. Create a GuidedDecodingParams object with the choices.
                guided_params = GuidedDecodingParams(choice=options)
                
                # 2. Pass this object to SamplingParams using the 'guided_decoding' keyword.
                sampling_params = SamplingParams(
                    temperature=0,
                    max_tokens=10,  # A small buffer for the choice string length.
                    guided_decoding=guided_params
                )
                # --- END OF FIX ---
                
                updated_item = copy.deepcopy(item)
                try:
                    outputs = self.llm.generate(prompt, sampling_params, use_tqdm=False)
                    if outputs:
                        prediction = outputs[0].outputs[0].text.strip()
                        matched_choice = next((opt for opt in options if opt in prediction), None)
                        updated_item["prediction"] = matched_choice if matched_choice else prediction
                    else:
                        updated_item["prediction"] = "Error: Generation failed."
                except Exception as e:
                    logger.error(f"Error during guided generation for item {original_index}: {e}")
                    updated_item["prediction"] = "Error: Guided generation failed."

                all_processed_items[original_index] = updated_item

        # 3. Process normal generation items.
        if normal_gen_items:
            prompts = [item.get("input", "") for item in normal_gen_items]
            sampling_params = SamplingParams(
                temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens,
                stop=self.stop, **{k:v for k,v in self.sampling_kwargs.items() if k not in ['temperature', 'top_p', 'max_tokens', 'stop']}
            )
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=show_progress)

            for i, output in enumerate(outputs):
                original_index = normal_gen_indices[i]
                result_item = copy.deepcopy(normal_gen_items[i])
                if output.outputs:
                    result_item["prediction"] = output.outputs[0].text.strip()
                    result_item["finish_reason"] = output.outputs[0].finish_reason
                else:
                    result_item["prediction"] = "vLLM Error: No output sequence generated"
                    result_item["finish_reason"] = "error"
                all_processed_items[original_index] = result_item

        # 4. Assemble final results in their original order.
        return [all_processed_items[i] for i in range(len(inputs))]