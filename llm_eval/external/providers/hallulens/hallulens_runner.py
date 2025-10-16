# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jsonlines
import pandas as pd
import os
import json
from pathlib import Path
import wandb

from llm_eval.external.providers.hallulens.tasks.longwiki.longwiki_main import run_eval

from llm_eval.external.providers.hallulens.utils import exp
from llm_eval.external.providers.hallulens.tasks.shortform.precise_wikiqa import PreciseQAEval
from llm_eval.external.providers.hallulens.tasks.refusal_test.nonsense_mixed_entities import NonsenseMixedEval, \
    NonsenseMixedInference
from llm_eval.external.providers.hallulens.tasks.refusal_test.round_robin_nonsense_name import NonsenseNameEval, \
    NonsenseNameInference


HALLULENS_DIR = os.path.dirname(os.path.abspath(__file__))

def precise_wikiqa_runner(
        qa_dataset_path: str,
        model: str = 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        inference_method: str = 'together',
        max_inference_tokens: int = 256,
        inf_batch_size: int = 64,
        N: int = 5000,
        limit: int = None,
        evaluator_abstention_model: str | None = None,
        evaluator_halu_model: str | None = None,
):
    """
    Args:
        model: model to use for generation
        inference_method: check and customize util/lm.py
        max_inference_tokens: maximum number of tokens for inference
        inf_batch_size: inference batch size
        N: number of samples
    """
    TASKNAME = f'precise_wikiqa'

    model_name = model.split("/")[-1]
    print(f"Running {TASKNAME} with model {model_name}")

    QAs_df = None

    import os
    seed = int(os.environ.get('RANDOM_SEED', '42'))
    # load all then apply limit if provided; otherwise keep legacy N behavior
    QAs = [line for line in jsonlines.open(qa_dataset_path, 'r')]
    QAs_df = pd.DataFrame(QAs)
    if limit is not None and len(QAs_df) > limit:
        QAs_df = QAs_df.sample(n=limit, random_state=seed)
    elif N is not None and len(QAs_df) > N:
        QAs_df = QAs_df.head(N)

    print(f"Starting Inference for [{model}], Testset_N: {QAs_df.shape}")
    exp.run_exp(
        task=f"{TASKNAME}",
        model_path=model,
        all_prompts=QAs_df,
        inference_method=inference_method, \
        max_tokens=max_inference_tokens,
        max_workers=inf_batch_size)
    print('Inference completed')

    print(f"Starting Evaluation for {model}")
    # evaluator 모델을 동적으로 지정: PreciseQAEval 내부 속성 재설정
    pq = PreciseQAEval(model_path=model, TASKNAME=TASKNAME)
    if evaluator_abstention_model:
        pq.abtention_evaluator = evaluator_abstention_model
    if evaluator_halu_model:
        pq.halu_evaluator = evaluator_halu_model
    eval_result = pq.run_eval()
    print(f'{TASKNAME} Evaluation completed')

    _log_to_wandb(pd.DataFrame([eval_result]), TASKNAME)



def longwiki_runner(
        benchmark_dataset_path: str = None,
        do_extract_only: bool = False,
        model: str = 'meta-llama/Llama-3.1-405B-Instruct-FP8',
        claim_extractor: str = 'meta-llama/Llama-3.1-405B-Instruct-FP8',
        abstain_evaluator: str = "meta-llama/Llama-3.1-70B-Instruct",
        verifier: str = 'meta-llama/Llama-3.1-405B-Instruct-FP8',
        inference_method: str = 'together',
        eval_cache_path: str = None,
        db_path: str = os.path.join(HALLULENS_DIR, 'data', 'wiki_data', '.cache', 'enwiki-20230401.db'),
        N: int = 250,
        k: int = 32,
        max_tokens: int = 1024,
        max_workers: int = 64,
        limit: int = None,
):
    TASKNAME = "longwiki"

    if benchmark_dataset_path is None:
        raise Exception("No long wiki benchmark dataset provided")

    # save all args details in
    base_path = os.path.dirname(os.path.abspath(__name__))
    model_name = model.split("/")[-1]

    # RUN INFERENCE
    seed = int(os.environ.get('RANDOM_SEED', '42'))
    all_prompts = pd.read_json(benchmark_dataset_path, lines=True)
    # apply limit if provided, else keep legacy N semantics
    if limit is not None and len(all_prompts) > limit:
        all_prompts = all_prompts.sample(n=limit, random_state=seed)
    elif N is not None and len(all_prompts) > N:
        all_prompts = all_prompts.head(N)

    print(f"Start Inference for {model} ", "dynamic", N)

    exp.run_exp(task=f"{TASKNAME}-dynamic",
                model_path=model,
                all_prompts=all_prompts,
                inference_method=inference_method,
                max_tokens=max_tokens,
                max_workers=max_workers,
                )

    print('\n***Inference completed')

    # RUN EVALUATION:
    print("============= [[ {} ]] =================".format("dynamic"))
    print(f"Running evaluation for {model_name};")
    print(f"** Refusal Evaluator: {abstain_evaluator}")
    print(f"** Claim Extractor: {claim_extractor}")
    print(f"** Verifier: {verifier}")
    print("=========================================")
    eval_result, overall_result_df = run_eval(
        do_extract_only=do_extract_only,
        model=model,
        claim_extractor=claim_extractor,
        abstain_evaluator=abstain_evaluator,
        verifier=verifier,
        eval_cache_path=eval_cache_path,
        db_path=db_path,
        k=k,
    )

    print('\n***Evaluation completed')

    _log_to_wandb(pd.DataFrame([eval_result]), TASKNAME)
    _log_to_wandb(overall_result_df, TASKNAME + "_detailed" + model_name)


def non_mixed_entity_runner(
        exp='nonsense_all',
        infer_overwrite=False,
        eval_overwrite=False,
        output_base_dir="output",
        prompt_path=None,
        tested_model='meta-llama/Llama-3.1-405B-Instruct-FP8',
        N=2000,
        seed=1,
        inference_method='together',
        limit: int = None,
):
    # set variables
    EXP = exp  # nonsense_medicine

    if not prompt_path:
        raise Exception("No prompt path provided")
    TASKNAME = f"{EXP}_{seed}_{N}"

    # run inference
    inference = NonsenseMixedInference(TASKNAME, output_base_dir, tested_model, prompt_path, seed,
                                       inference_method, limit=limit)
    if infer_overwrite:
        inference.remove_existing_files()
    inference.run_inference()

    # run evaluation
    if 'gemma' in tested_model:
        med_safety_filtered_model = True
        eval = NonsenseMixedEval(TASKNAME, output_base_dir, tested_model, prompt_path,
                                 med_safety_filtered_model)
    else:
        eval = NonsenseMixedEval(TASKNAME, output_base_dir, tested_model, prompt_path)
    res, task_path = eval.run_eval(eval_overwrite)

    # Log to evaluation result to wandb
    _log_non_refusal_result_to_wandb(task_path, inference.TASKNAME)

    return res


def non_generated_entity_runner(
        infer_overwrite=False,
        eval_overwrite=False,
        output_base_dir="output",
        prompt_path=None,
        generate_model='meta-llama/Llama-3.1-8B-Instruct',
        inference_method='together',
        seed=0,
        limit: int = None,
):
    if prompt_path == None:
        raise Exception("No prompt path provided")

    inference = NonsenseNameInference(output_base_dir, generate_model, prompt_path, seed, inference_method, limit=limit)
    if infer_overwrite:
        inference.remove_existing_files()
    inference.run_inference()

    eval = NonsenseNameEval(output_base_dir, generate_model, prompt_path)
    res, task_path = eval.run_eval(eval_overwrite)
    N = len(res['refusal_eval_raw'])
    refusal_rate = sum(res['refusal_eval_raw']) / N * 100
    print(f"[{res['model']}] || Refusal rate: {refusal_rate} || N = {N}")

    # Log to evaluation result to wandb
    _log_non_refusal_result_to_wandb(task_path, inference.TASKNAME)

    return res


def _log_non_refusal_result_to_wandb(task_path: str, task_name: str):
    # Upload results to wandb
    generation = []

    with open(f"{task_path}/eval_results.json", "r") as f:  # Evaluation information
        eval_result = json.load(f)

    with open(f"{task_path}/generation.jsonl", "r") as f:  # Prompt information
        for line in f:
            generation.append(json.loads(line))
    eval_result['prompt_information'] = generation

    eval_col = ['model', 'false_acceptance_rate', 'refusal_rate']

    eval_result_with_prompt_info = pd.DataFrame(eval_result)
    score = {key: eval_result[key] for key in eval_col}
    score = pd.DataFrame([score])

    _log_to_wandb(score, task_name)
    # TODO: This evaluation detail log is too large to upload to wandb. If it needs, consider a way to upload it.
    # _log_to_wandb(eval_result_with_prompt_info, f"{score['model']}_{task_name}_detailed")


def _log_to_wandb(result_df: pd.DataFrame, model_task_name: str) -> None:
    """Log evaluation summary to Weights & Biases if configured."""
    from llm_eval.wandb_singleton import WandbConfigSingleton
    leaderboard_table = wandb.Table(dataframe=result_df)
    run = WandbConfigSingleton.get_instance().run
    run.log({model_task_name+'_leaderboard_table': leaderboard_table})
