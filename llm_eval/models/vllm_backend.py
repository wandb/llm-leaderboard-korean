import logging
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import copy

# Attempt to import vllm, set a flag, and raise ImportError later if not available.
try:
    import vllm
    VLLM_AVAILABLE = True
except ImportError:
    vllm = None
    VLLM_AVAILABLE = False

from tqdm import tqdm

from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger
from llm_eval.utils.prompt_template import default_cot_parser

# Initialize a logger for this module
logger = get_logger(name="vllm_backend", level=logging.INFO)

@register_model("vllm")
class VLLMModel(BaseModel):
    """
    A backend that uses the vLLM library for offline inference.
    It initializes and uses the vLLM engine within the HRET process without a separate API server.

    Requires the `vllm` package to be installed (`pip install vllm`).
    Installation may vary based on your CUDA setup; please refer to the official vLLM documentation.

    Args:
        model_name_or_path (str): Path to the model or its Hugging Face ID.
        temperature (float): Sampling temperature. Default is 0.0 (greedy).
        top_p (float): Top-p sampling probability. Default is 1.0.
        max_tokens (int): Maximum number of tokens to generate. Default is 512.
        stop (Optional[List[str]]): List of stop sequences. Default is None.
        dtype (str): Data type for model weights ('auto', 'half', 'bfloat16', etc.). Default is 'auto'.
        tensor_parallel_size (int): Number of GPUs for tensor parallelism. Default is 1.
        gpu_memory_utilization (float): The fraction of GPU memory to reserve for the model. Default is 0.9.
        max_num_seqs (int): Maximum number of sequences the vLLM engine can process at once.
        max_num_batched_tokens (int): Maximum number of tokens the vLLM engine can process in a batch.
        cot (bool): Whether to enable Chain-of-Thought prompting. Default is False.
        cot_trigger (Optional[str]): The phrase added to the prompt to trigger CoT.
        cot_parser (Optional[Callable]): A function to parse CoT output into (chain_of_thought, final_answer).
        **kwargs: Additional arguments to be passed to `vllm.LLM` initialization or `vllm.SamplingParams`.
                  Only valid arguments for each class will be used.
    """
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
            raise ImportError(
                "vLLM is not installed. Please install it to use VLLMModel: "
                "`pip install vllm`. Note: Installation might require specific steps "
                "depending on your CUDA version. Please refer to vLLM documentation."
            )

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
                "tokenizer", "tokenizer_mode", "skip_tokenizer_init", "tokenizer_revision",
                "trust_remote_code", "revision", "code_revision", "rope_scaling", "rope_theta",
                "seed", "quantization", "enforce_eager", "max_model_len", "swap_space",
                "kv_cache_dtype", "block_size", "worker_use_ray", "pipeline_parallel_size",
                "enable_prefix_caching", "disable_custom_all_reduce", "max_num_batched_tokens",
                "max_num_seqs", "max_paddings", "num_gpu_blocks_override", "load_format",
                "engine_use_ray", "disable_log_stats", "disable_log_requests"
            ]
            llm_init_kwargs = {k: v for k, v in kwargs.items() if k in llm_init_valid_args}


            self.llm = vllm.LLM(
                model=model_name_or_path,
                dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                **llm_init_kwargs
            )
            # Fetch the tokenizer from the LLM instance for log probability calculations.
            self.tokenizer = self.llm.get_tokenizer()
            logger.info("vLLM engine and tokenizer initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}", exc_info=True)
            raise e

    def _score_options_vllm(self, item: Dict[str, Any]) -> List[float]:
        """
        Calculates log probability scores for each option in an MCQA sample using vLLM.
        """
        question = item.get("input", "")
        options = item.get("options", [])
        if not options:
            return []

        # Create full prompts for each option
        prompts = [question + opt for opt in options]
        
        # Configure sampling parameters for log probability calculation.
        # Setting `prompt_logprobs` makes vLLM return log-likelihoods for each token in the prompt.
        # `max_tokens=1` speeds up the process by minimizing unnecessary generation.
        sampling_params = vllm.SamplingParams(
            temperature=0, 
            max_tokens=1, 
            prompt_logprobs=1
        )
        
        # Generate with vLLM (in this case, primarily for scoring)
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        log_likelihoods = []
        
        for output, choice in zip(outputs, options):
            # As of vLLM 0.4.1, the first element of prompt_logprobs is None.
            if output.prompt_logprobs is None or len(output.prompt_logprobs) <= 1:
                log_likelihoods.append(float('-inf')) # Assign a very low value on error
                logger.warning(f"Could not get prompt_logprobs for prompt: {output.prompt[:50]}...")
                continue
            
            prompt_logprobs = output.prompt_logprobs

            # Calculate the token length of the choice part
            choice_token_len = len(self.tokenizer(choice)['input_ids'])
            
            # Extract and sum the log probabilities corresponding to the choice part.
            # We slice the last `choice_token_len` logprobs from the list.
            choice_logprobs = prompt_logprobs[-choice_token_len:]
            
            total_logprob = sum(
                list(logprob_dict.values())[0]
                for logprob_dict in choice_logprobs if logprob_dict
            )
            log_likelihoods.append(total_logprob)
            
        return log_likelihoods


    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        batch_size: Optional[int] = None,
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generates text for a batch of inputs using the initialized vLLM engine.
        It distinguishes between MCQA (if 'options' field is present) and standard generation.
        """
        if not self.llm:
            raise RuntimeError("vLLM engine is not initialized.")
            
        mcqa_items = []
        mcqa_indices = []
        normal_gen_items = []
        normal_gen_indices = []

        # 1. Separate inputs into MCQA and normal generation tasks
        for i, item in enumerate(inputs):
            is_mcqa = "options" in item and isinstance(item["options"], list) and item["options"] and return_logits
            if is_mcqa:
                mcqa_items.append(item)
                mcqa_indices.append(i)
            else:
                normal_gen_items.append(item)
                normal_gen_indices.append(i)
        
        # Dictionary to hold all processed results, keyed by original index
        all_processed_items = {}

        # 2. Process MCQA items individually
        for i, item in enumerate(mcqa_items):
            original_index = mcqa_indices[i]
            updated_item = copy.deepcopy(item)
            option_scores = self._score_options_vllm(updated_item)
            
            # Store the results in the format expected by LogProbEvaluator
            updated_item.setdefault("logits", {})["option_log_probs"] = option_scores
            
            if option_scores:
                best_idx = option_scores.index(max(option_scores))
                updated_item["prediction"] = updated_item["options"][best_idx]
            else:
                updated_item["prediction"] = "Error: Failed to score options."
            
            all_processed_items[original_index] = updated_item

        # 3. Process normal generation items in a single batch
        if normal_gen_items:
            prompts = []
            for item in normal_gen_items:
                prompt = item.get("input", "")
                if self.cot and self.cot_trigger:
                    prompt += f"\n{self.cot_trigger}"
                prompts.append(prompt)

            current_sampling_kwargs = copy.deepcopy(self.sampling_kwargs)
            current_sampling_kwargs.update(kwargs)

            temperature = current_sampling_kwargs.get("temperature", self.temperature)
            if temperature <= 1e-6:
                temperature = 0.0
                current_sampling_kwargs["top_p"] = 1.0

            stop_sequences = until if until is not None else self.stop
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]

            logprobs_setting = current_sampling_kwargs.get("logprobs", None)
            include_logprobs = return_logits and isinstance(logprobs_setting, int) and logprobs_setting > 0

            valid_sampling_param_keys = [
                "n", "best_of", "presence_penalty", "frequency_penalty", "temperature",
                "top_p", "top_k", "min_p", "seed", "use_beam_search", "length_penalty",
                "early_stopping", "stop", "stop_token_ids", "include_stop_str_in_output",
                "ignore_eos", "max_tokens", "logprobs", "prompt_logprobs", "skip_special_tokens",
                "spaces_between_special_tokens"
            ]
            filtered_sampling_kwargs = {k: v for k, v in current_sampling_kwargs.items() if k in valid_sampling_param_keys}
            
            sampling_params = vllm.SamplingParams(
                temperature=temperature,
                top_p=filtered_sampling_kwargs.get("top_p", self.top_p),
                max_tokens=filtered_sampling_kwargs.get("max_tokens", self.max_tokens),
                stop=stop_sequences,
                logprobs=logprobs_setting if include_logprobs else None,
                **{k: v for k, v in filtered_sampling_kwargs.items() if k not in ["temperature", "top_p", "max_tokens", "stop", "logprobs"]}
            )

            try:
                outputs = self.llm.generate(prompts, sampling_params, use_tqdm=show_progress)
            except Exception as e:
                logger.error(f"Error during vLLM generation: {e}", exc_info=True)
                # Populate results with error messages if generation fails
                for i, item in enumerate(normal_gen_items):
                    original_index = normal_gen_indices[i]
                    result_item = copy.deepcopy(item)
                    result_item["prediction"] = f"vLLM Generation Error: {e}"
                    result_item["error"] = str(e)
                    result_item["finish_reason"] = "error"
                    all_processed_items[original_index] = result_item
                # Re-sort and return
                final_results_sorted = [all_processed_items[i] for i in range(len(inputs))]
                return final_results_sorted


            for i, output in enumerate(outputs):
                original_index = normal_gen_indices[i]
                result_item = copy.deepcopy(normal_gen_items[i])
                
                if output.outputs:
                    first_output_seq = output.outputs[0]
                    generated_text = first_output_seq.text.strip()
                    result_item["finish_reason"] = first_output_seq.finish_reason

                    if self.cot and self.cot_parser:
                        try:
                            chain_of_thought, final_answer = self.cot_parser(generated_text)
                            result_item["chain_of_thought"] = chain_of_thought
                            result_item["prediction"] = final_answer
                        except Exception as parse_err:
                            logger.warning(f"CoT parsing failed for output: '{generated_text[:100]}...'. Error: {parse_err}. Using raw output.")
                            result_item["prediction"] = generated_text
                    else:
                        result_item["prediction"] = generated_text
                    
                    if include_logprobs and first_output_seq.logprobs:
                        token_log_probs_list = [list(logprob_dict.values())[0] for logprob_dict in first_output_seq.logprobs if logprob_dict]
                        sum_log_prob = sum(token_log_probs_list)
                        result_item["logits"] = {
                            "sum_log_prob": sum_log_prob,
                            "token_log_probs": token_log_probs_list,
                        }
                else:
                    result_item["prediction"] = "vLLM Error: No output sequence generated"
                    result_item["error"] = "No output sequence from vLLM"
                    result_item["finish_reason"] = "error"
                
                all_processed_items[original_index] = result_item

        # 4. Assemble final results in their original order
        final_results_sorted = [all_processed_items[i] for i in range(len(inputs))]

        return final_results_sorted