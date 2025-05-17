import logging
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import copy

# Try importing vllm, raise ImportError later if not available
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

# Initialize logger for this module
logger = get_logger(name="vllm_backend", level=logging.INFO)

@register_model("vllm") 
class VLLMModel(BaseModel):
    """
    Backend using the vLLM library for offline inference.
    Initializes and uses the vLLM engine within the HRET process, without needing a separate API server.

    Requires the `vllm` package to be installed (`pip install vllm`).
    Installation might depend on your specific CUDA setup. Check the vLLM documentation.

    Args:
        model_name_or_path (str): Path or Hugging Face identifier for the model.
        temperature (float): Sampling temperature. Defaults to 0.0 (greedy).
        top_p (float): Top-p sampling probability. Defaults to 1.0.
        max_tokens (int): Maximum number of tokens to generate. Defaults to 512.
        stop (Optional[List[str]]): List of stop sequences. Defaults to None.
        dtype (str): Data type for model weights (e.g., 'auto', 'half', 'bfloat16'). Defaults to 'auto'.
        tensor_parallel_size (int): Number of GPUs for tensor parallelism. Defaults to 1.
        gpu_memory_utilization (float): Fraction of GPU memory to reserve for the model. Defaults to 0.9.
        cot (bool): Whether to enable Chain-of-Thought prompting. Defaults to False.
        cot_trigger (Optional[str]): Trigger phrase appended to the prompt for CoT. Defaults to "Let's think step by step.".
        cot_parser (Optional[Callable]): Function to parse CoT output into (chain_of_thought, final_answer). Defaults to `default_cot_parser`.
        **kwargs: Additional arguments passed to `vllm.LLM` during initialization or `vllm.SamplingParams` during generation.
                  Only valid arguments for each respective class will be used.
    """
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        cot: bool = False,
        cot_trigger: Optional[str] = "Let's think step by step.",
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = default_cot_parser,
        **kwargs,
    ):
        # Initialize BaseModel with CoT related parameters
        super().__init__(cot=cot, cot_trigger=cot_trigger, cot_parser=cot_parser, **kwargs)

        # Check if vLLM library is available
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Please install it to use VLLMOfflineModel: "
                "`pip install vllm`. Note: Installation might require specific steps "
                "depending on your CUDA version. Please refer to vLLM documentation."
            )

        # Set model name identifier
        self.model_name = f"vllm_offline:{model_name_or_path}"
        logger.info(f"Initializing vLLM engine for model: {model_name_or_path}")

        # Store default generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop if stop is not None else []
        # Store remaining kwargs to be potentially used for SamplingParams later
        self.sampling_kwargs = kwargs

        # Initialize the vLLM engine
        try:
            # Filter kwargs to only include valid arguments for vllm.LLM constructor
            # This prevents passing unsupported arguments like 'temperature' to LLM init
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
                **llm_init_kwargs # Pass filtered kwargs
            )
            # Optionally get the tokenizer if needed later (e.g., for logit processing)
            # Requires vLLM >= 0.2.7
            # try:
            #     self.tokenizer = self.llm.get_tokenizer()
            # except AttributeError:
            #     logger.warning("Could not get tokenizer from vLLM engine. Logit token display might be limited.")
            #     self.tokenizer = None

        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}", exc_info=True)
            raise e

        logger.info(" vLLM engine initialized successfully.")


    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        batch_size: Optional[int] = None, # Note: vLLM handles batching internally, this arg is ignored.
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generates text for a batch of inputs using the initialized vLLM engine.

        Args:
            inputs (List[Dict[str, Any]]): List of dictionaries, each must have an "input" key containing the prompt text.
            return_logits (bool): If True, attempts to return log probabilities for the generated tokens.
                                  Requires `logprobs` (int > 0) to be set in `kwargs` or `self.sampling_kwargs`. Defaults to False.
            batch_size (Optional[int]): This argument is ignored as vLLM handles batching internally.
            until (Optional[Union[str, List[str]]]): Optional stop sequence(s). Overrides the `stop` parameter set during initialization if provided.
            show_progress (bool): Whether to display a tqdm progress bar during generation (passed to vLLM's `use_tqdm`). Defaults to True.
            **kwargs: Overrides for `vllm.SamplingParams` (e.g., temperature, top_p, max_tokens, logprobs, n, best_of, etc.).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries. Each dictionary corresponds to an input item,
                                  updated with "prediction" (the generated text), and optionally
                                  "chain_of_thought" (if CoT is enabled and parsing succeeds) and
                                  "logits" (if `return_logits` is True and logprobs are available).
                                  Includes "error" field if generation failed for an item.
        """
        if not self.llm:
            raise RuntimeError("vLLM engine is not initialized.")

        # --- Prepare Prompts ---
        prompts = []
        for item in inputs:
            prompt = item.get("input", "") # Get prompt from input dict
            # Append CoT trigger if CoT is enabled
            if self.cot and self.cot_trigger:
                prompt += f"\n{self.cot_trigger}"
            prompts.append(prompt)

        # --- Prepare SamplingParams ---
        # Start with default sampling kwargs stored during init
        current_sampling_kwargs = copy.deepcopy(self.sampling_kwargs)
        # Override with any kwargs passed directly to this method call
        current_sampling_kwargs.update(kwargs)

        # Handle temperature: vLLM uses 0.0 for greedy.
        temperature = current_sampling_kwargs.get("temperature", self.temperature)
        if temperature <= 1e-6:
            temperature = 0.0
            # Ensure top_p is 1.0 for greedy decoding
            current_sampling_kwargs["top_p"] = 1.0

        # Handle stop sequences: override init `stop` if `until` is provided
        stop_sequences = until if until is not None else self.stop
        if isinstance(stop_sequences, str): # Ensure it's a list
            stop_sequences = [stop_sequences]

        # Determine if logprobs should be requested from vLLM
        logprobs_setting = current_sampling_kwargs.get("logprobs", None) # User can specify number of logprobs
        include_logprobs = return_logits and isinstance(logprobs_setting, int) and logprobs_setting > 0

        # Filter kwargs to only include valid arguments for vllm.SamplingParams
        valid_sampling_param_keys = [
            "n", "best_of", "presence_penalty", "frequency_penalty", "temperature",
            "top_p", "top_k", "min_p", "seed", "use_beam_search", "length_penalty",
            "early_stopping", "stop", "stop_token_ids", "include_stop_str_in_output",
            "ignore_eos", "max_tokens", "logprobs", "prompt_logprobs", "skip_special_tokens",
            "spaces_between_special_tokens"
        ]
        filtered_sampling_kwargs = {k: v for k, v in current_sampling_kwargs.items() if k in valid_sampling_param_keys}

        # Create SamplingParams object
        sampling_params = vllm.SamplingParams(
            temperature=temperature,
            top_p=filtered_sampling_kwargs.get("top_p", self.top_p),
            max_tokens=filtered_sampling_kwargs.get("max_tokens", self.max_tokens),
            stop=stop_sequences,
            logprobs=logprobs_setting if include_logprobs else None,
            # Pass remaining valid filtered kwargs
            **{k: v for k, v in filtered_sampling_kwargs.items() if k not in ["temperature", "top_p", "max_tokens", "stop", "logprobs"]}
        )

        # --- Generate ---
        logger.info(f"Generating {len(prompts)} prompts with vLLM using SamplingParams: {sampling_params}")
        try:
            # vLLM's generate method handles batching internally.
            # use_tqdm delegates progress bar display to vLLM.
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=show_progress)
        except Exception as e:
            logger.error(f"Error during vLLM generation: {e}", exc_info=True)
            # If generation fails for the whole batch, return inputs with error info
            results = []
            for item in inputs:
                result_item = copy.deepcopy(item)
                result_item["prediction"] = f"vLLM Generation Error: {e}"
                result_item["error"] = str(e)
                result_item["finish_reason"] = "error"
                results.append(result_item)
            return results

        # --- Process Outputs ---
        results = []
        # Check if the number of outputs matches inputs, log warning if not
        if len(outputs) != len(inputs):
             logger.warning(f"Input count ({len(inputs)}) and vLLM output count ({len(outputs)}) mismatch. "
                            f"This might indicate dropped requests. Will attempt to align results.")

        # Iterate through original inputs to maintain order and structure
        for i, item in enumerate(inputs):
            result_item = copy.deepcopy(item) # Start with a copy of the input item

            if i < len(outputs):
                output = outputs[i] # Get the corresponding vLLM output object
                # vLLM can return multiple sequences per prompt (e.g., with n>1 or best_of>1)
                # Here, we assume we only need the first one (index 0).
                if output.outputs:
                    first_output_seq = output.outputs[0]
                    generated_text = first_output_seq.text.strip()
                    finish_reason = first_output_seq.finish_reason

                    # Apply CoT parser if CoT is enabled
                    if self.cot and self.cot_parser:
                        try:
                            # Parse the raw generated text
                            chain_of_thought, final_answer = self.cot_parser(generated_text)
                            result_item["chain_of_thought"] = chain_of_thought
                            result_item["prediction"] = final_answer
                        except Exception as parse_err:
                            # If parsing fails, log a warning and use the raw text as prediction
                            logger.warning(f"CoT parsing failed for output: '{generated_text[:100]}...'. Error: {parse_err}. Using raw output.", exc_info=False)
                            result_item["prediction"] = generated_text # Fallback to raw output
                    else:
                        # If CoT is not enabled, the prediction is the raw generated text
                        result_item["prediction"] = generated_text

                    result_item["finish_reason"] = finish_reason

                    # Handle logits/logprobs if requested and returned by vLLM
                    if include_logprobs and first_output_seq.logprobs:
                        # vLLM returns logprobs as List[Dict[int, float]] where each dict maps token_id to logprob for that step
                        token_log_probs_list = []
                        sum_log_prob = 0.0
                        generated_token_ids = first_output_seq.token_ids
                        step_logprobs = first_output_seq.logprobs # This is the List[Dict[int, float]]

                        # Ensure lengths match before processing
                        if len(generated_token_ids) == len(step_logprobs):
                            for token_id, logprob_dict in zip(generated_token_ids, step_logprobs):
                                 # The logprob dict for the step should contain the logprob for the generated token_id
                                 if token_id in logprob_dict:
                                     token_log_prob = logprob_dict[token_id]
                                     token_log_probs_list.append(token_log_prob)
                                     sum_log_prob += token_log_prob
                                 else:
                                     # This case might occur if the top logprobs didn't include the sampled token
                                     logger.warning(f"Generated token ID {token_id} not found in logprobs dict {list(logprob_dict.keys())} for step. Appending None.")
                                     token_log_probs_list.append(None) # Indicate missing logprob

                            result_item["logits"] = {
                                "sum_log_prob": sum_log_prob,
                                "token_log_probs": token_log_probs_list,
                                # Optionally add tokens if tokenizer is available
                                # "tokens": self.tokenizer.convert_ids_to_tokens(generated_token_ids) if self.tokenizer else None,
                            }
                        else:
                             # Log warning if lengths don't match, indicating an issue
                             logger.warning(f"Length mismatch between generated tokens ({len(generated_token_ids)}) "
                                            f"and logprobs steps ({len(step_logprobs)}). Cannot reliably extract logprobs.")
                             result_item["logits"] = {"error": "Logprob length mismatch"}

                    elif return_logits:
                         # Logprobs were requested but not returned by vLLM
                         result_item["logits"] = {"error": "Logprobs requested but not returned by vLLM (check sampling_params 'logprobs' setting)"}
                else:
                    # Handle case where vLLM returns a RequestOutput but with an empty 'outputs' list
                    logger.error(f"vLLM returned no output sequences for input index {i}. Prompt: '{output.prompt[:100]}...'. Finish reason: {output.outputs[0].finish_reason if output.outputs else 'N/A'}")
                    result_item["prediction"] = "vLLM Error: No output sequence generated"
                    result_item["error"] = "No output sequence from vLLM"
                    result_item["finish_reason"] = output.outputs[0].finish_reason if output.outputs else "error"

            else:
                 # Handle case where vLLM output list is shorter than input list
                 logger.error(f"Missing vLLM output for input index {i}. Input prompt: '{item.get('input', '')[:100]}...'")
                 result_item["prediction"] = "vLLM Error: Missing output in batch result"
                 result_item["error"] = "Missing output from vLLM batch"
                 result_item["finish_reason"] = "error"

            # Append the processed item (or item with error info) to the results
            results.append(result_item)

        return results

