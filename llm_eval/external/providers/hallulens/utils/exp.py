# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, List, Optional
import os
from tqdm.contrib.concurrent import thread_map

from llm_eval.external.providers.hallulens.utils import lm
from llm_eval.models import load_model

def run_exp(
    task: str,
    model_path: str,
    all_prompts,
    generations_file_path=None,
    base_path="output",
    inference_method="together",
    max_workers=64,
    max_tokens=512,
    return_gen = False,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    evaluation_logger: Optional[Any] = None,
):  
    if not generations_file_path:
        base_path = Path(base_path)
        model_name = model_path.split("/")[-1]
        output_folder = base_path / task / model_name
        output_folder.mkdir(exist_ok=True, parents=True)
        generations_file_path = output_folder / "generation.jsonl"

    generations_file_path = str(generations_file_path)
    print('generations_file_path', generations_file_path)

    prompts =  all_prompts.prompt.to_list()

    # get the response from the model
    # if inference_method == 'openai':
    #     all_prompts["generation"] = thread_map(
    #         lambda p: lm.openai_generate(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens),
    #         prompts,
    #         max_workers=max_workers,
    #         desc="Predict openai",
    #     )
    # elif inference_method == "vllm":
    #     port = None
    #     all_prompts["generation"] = thread_map(
    #         lambda p: lm.call_vllm_api(p, model=model_path, temperature=0.0, top_p=1.0,  max_tokens=max_tokens, port=port),
    #         prompts,
    #         max_workers=max_workers,
    #         desc="Predict on vllm",
    #     )
    # elif inference_method == "custom":
    #     all_prompts["generation"] = thread_map(
    #         lambda p: lm.generate(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens ),
    #         prompts,
    #         max_workers=max_workers,
    #         desc="Predict on custom API",
    #     )
    # elif inference_method == "anthropic":
    #     all_prompts["generation"] = thread_map(
    #         lambda p: lm.claude_generate(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens),
    #         prompts,
    #         max_workers=max_workers,
    #         desc="Predict on Claude API",
    #     )
    # elif inference_method == "together":
    #     all_prompts["generation"] = thread_map(
    #         lambda p: lm.call_together_api(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens),
    #         prompts,
    #         max_workers=max_workers,
    #         desc="Predict on together API",
    #     )
    # elif inference_method in ("openai_backend", "litellm_backend", "huggingface_backend", "vllm_backend"):
        # Use llm_eval/models backends via registry
    backend_kwargs = backend_kwargs or {}

    # Build inputs for batch generation
    inputs: List[Dict[str, Any]] = [{"input": p} for p in prompts]

    import weave
    from tqdm import tqdm as tqdm_progress

    # Wrap inference in Weave op for automatic token/latency tracking
    @weave.op()
    def hallulens_inference_single(model_obj, input_data):
        """Process a single input with token/latency tracking."""
        return model_obj.generate_batch([input_data], show_progress=False)[0]

    @weave.op()
    def hallulens_inference_batch(model_obj, inputs_data):
        """Process batch of inputs (fallback when no evaluation_logger)."""
        return model_obj.generate_batch(inputs_data, show_progress=True)

    if inference_method == "openai":
        # Registry name is "openai"
        api_base = backend_kwargs.get(
            "api_base",
        )
        batch_size = int(backend_kwargs.get("batch_size", max_workers))
        model = load_model(
            name="openai",
            use_chat_api=True,
            **backend_kwargs,
        )
        # If evaluation_logger is provided, process each input individually with log_prediction
        if evaluation_logger is not None:
            generations = []
            for idx, inp in enumerate(tqdm_progress(inputs, desc="HalluLens inference with tracking")):
                prompt_text = inp.get("input", "")
                with evaluation_logger.log_prediction(
                    inputs={"input": str(prompt_text), "index": idx},
                    output=""
                ) as pred_logger:
                    result = hallulens_inference_single(model, inp)
                    prediction_text = result.get("prediction") or ""
                    generations.append(prediction_text)
                    # Update output within the context
                    if pred_logger is not None:
                        try:
                            pred_logger.output = prediction_text
                        except (AttributeError, TypeError):
                            if hasattr(pred_logger, '_output'):
                                pred_logger._output = prediction_text
            all_prompts["generation"] = generations
        else:
            results = hallulens_inference_batch(model, inputs)
            generations = [(r.get("prediction") or "") for r in results]
            all_prompts["generation"] = generations
    elif inference_method == "litellm":
        # Registry name is "litellm"
        # provider = backend_kwargs.get("provider", "openai")
        # api_key = backend_kwargs.get("api_key", os.environ.get("OPENAI_API_KEY"))
        batch_size = int(backend_kwargs.get("batch_size", max_workers))
        model = load_model(name="litellm", **backend_kwargs)
        #     name="litellm",
        #     provider=provider,
        #     model_name=model_path,
        #     api_key=api_key,
        #     batch_size=batch_size,
        #     max_new_tokens=max_tokens,
        # )
        # If evaluation_logger is provided, process each input individually with log_prediction
        if evaluation_logger is not None:
            generations = []
            for idx, inp in enumerate(tqdm_progress(inputs, desc="HalluLens inference with tracking")):
                prompt_text = inp.get("input", "")
                with evaluation_logger.log_prediction(
                    inputs={"input": str(prompt_text), "index": idx},
                    output=""
                ) as pred_logger:
                    result = hallulens_inference_single(model, inp)
                    prediction_text = result.get("prediction") or ""
                    generations.append(prediction_text)
                    # Update output within the context
                    if pred_logger is not None:
                        try:
                            pred_logger.output = prediction_text
                        except (AttributeError, TypeError):
                            if hasattr(pred_logger, '_output'):
                                pred_logger._output = prediction_text
            all_prompts["generation"] = generations
        else:
            results = hallulens_inference_batch(model, inputs)
            generations = [(r.get("prediction") or "") for r in results]
            all_prompts["generation"] = generations
    elif inference_method == "huggingface":
        # Registry name is "huggingface"
        batch_size = int(backend_kwargs.get("batch_size", 1))
        device = backend_kwargs.get("device")
        dtype = backend_kwargs.get("dtype", "auto")
        model = load_model(
            name="huggingface",
            model_name_or_path=model_path,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            device=device,
            dtype=dtype,
        )
        # If evaluation_logger is provided, process each input individually with log_prediction
        if evaluation_logger is not None:
            generations = []
            for idx, inp in enumerate(tqdm_progress(inputs, desc="HalluLens inference with tracking")):
                prompt_text = inp.get("input", "")
                with evaluation_logger.log_prediction(
                    inputs={"input": str(prompt_text), "index": idx},
                    output=""
                ) as pred_logger:
                    result = hallulens_inference_single(model, inp)
                    prediction_text = result.get("prediction") or ""
                    generations.append(prediction_text)
                    # Update output within the context
                    if pred_logger is not None:
                        try:
                            pred_logger.output = prediction_text
                        except (AttributeError, TypeError):
                            if hasattr(pred_logger, '_output'):
                                pred_logger._output = prediction_text
            all_prompts["generation"] = generations
        else:
            results = hallulens_inference_batch(model, inputs)
            generations = [(r.get("prediction") or "") for r in results]
            all_prompts["generation"] = generations
    elif inference_method == "vllm":
        # Registry name is "vllm" (requires vllm installed and GPU)
        temperature = backend_kwargs.get("temperature", 0.0)
        top_p = backend_kwargs.get("top_p", 1.0)
        stop = backend_kwargs.get("stop")  # Optional[List[str]]
        dtype = backend_kwargs.get("dtype", "auto")
        tensor_parallel_size = int(backend_kwargs.get("tensor_parallel_size", 1))
        gpu_memory_utilization = float(backend_kwargs.get("gpu_memory_utilization", 0.9))
        model = load_model(
            name="vllm",
            model_name_or_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        # If evaluation_logger is provided, process each input individually with log_prediction
        if evaluation_logger is not None:
            generations = []
            for idx, inp in enumerate(tqdm_progress(inputs, desc="HalluLens inference with tracking")):
                prompt_text = inp.get("input", "")
                with evaluation_logger.log_prediction(
                    inputs={"input": str(prompt_text), "index": idx},
                    output=""
                ) as pred_logger:
                    result = hallulens_inference_single(model, inp)
                    prediction_text = result.get("prediction") or ""
                    generations.append(prediction_text)
                    # Update output within the context
                    if pred_logger is not None:
                        try:
                            pred_logger.output = prediction_text
                        except (AttributeError, TypeError):
                            if hasattr(pred_logger, '_output'):
                                pred_logger._output = prediction_text
            all_prompts["generation"] = generations
        else:
            results = hallulens_inference_batch(model, inputs)
            generations = [(r.get("prediction") or "") for r in results]
            all_prompts["generation"] = generations
    elif inference_method == "openai_responses":
        # Registry name is "openai_responses" (OpenAI Responses API for reasoning models)
        model = load_model(name="openai_responses", **backend_kwargs)
        # If evaluation_logger is provided, process each input individually with log_prediction
        if evaluation_logger is not None:
            generations = []
            for idx, inp in enumerate(tqdm_progress(inputs, desc="HalluLens inference with tracking")):
                prompt_text = inp.get("input", "")
                with evaluation_logger.log_prediction(
                    inputs={"input": str(prompt_text), "index": idx},
                    output=""
                ) as pred_logger:
                    result = hallulens_inference_single(model, inp)
                    prediction_text = result.get("prediction") or ""
                    generations.append(prediction_text)
                    # Update output within the context
                    if pred_logger is not None:
                        try:
                            pred_logger.output = prediction_text
                        except (AttributeError, TypeError):
                            if hasattr(pred_logger, '_output'):
                                pred_logger._output = prediction_text
            all_prompts["generation"] = generations
        else:
            results = hallulens_inference_batch(model, inputs)
            generations = [(r.get("prediction") or "") for r in results]
            all_prompts["generation"] = generations
    else:
        raise NotImplementedError(f"Unsupported backend method: {inference_method}")
    # else:
    #     raise NotImplementedError(f"No method {inference_method}")

    # save the results
    all_prompts.to_json(generations_file_path, lines=True, orient="records")

    if return_gen:
        return all_prompts