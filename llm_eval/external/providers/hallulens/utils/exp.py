# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from tqdm.contrib.concurrent import thread_map

from llm_eval.external.providers.hallulens.utils import lm

def run_exp(
    task: str,
    model_path: str,
    all_prompts,
    generations_file_path=None,
    base_path="output",
    inference_method="together",
    max_workers=64,
    max_tokens=512,
    return_gen = False
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
    if inference_method == 'openai':
        all_prompts["generation"] = thread_map(
            lambda p: lm.openai_generate(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens),
            prompts,
            max_workers=max_workers,
            desc="Predict openai",
        )
    elif inference_method == "vllm":
        port = None
        all_prompts["generation"] = thread_map(
            lambda p: lm.call_vllm_api(p, model=model_path, temperature=0.0, top_p=1.0,  max_tokens=max_tokens, port=port),
            prompts,
            max_workers=max_workers,
            desc="Predict on vllm",
        )
    elif inference_method == "custom":
        all_prompts["generation"] = thread_map(
            lambda p: lm.generate(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens ),
            prompts,
            max_workers=max_workers,
            desc="Predict on custom API",
        )
    elif inference_method == "anthropic":
        all_prompts["generation"] = thread_map(
            lambda p: lm.claude_generate(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens),
            prompts,
            max_workers=max_workers,
            desc="Predict on Claude API",
        )
    elif inference_method == "together":
        all_prompts["generation"] = thread_map(
            lambda p: lm.call_together_api(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens),
            prompts,
            max_workers=max_workers,
            desc="Predict on together API",
        )
    else:
        raise NotImplementedError(f"No method {inference_method}")

    # save the results
    all_prompts.to_json(generations_file_path, lines=True, orient="records")

    if return_gen:
        return all_prompts