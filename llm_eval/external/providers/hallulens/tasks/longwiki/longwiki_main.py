# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pandas as pd
import os
import json
from pathlib import Path

from llm_eval.external.providers.hallulens.tasks.longwiki.facthalu import FactHalu
from llm_eval.external.providers.hallulens.utils import exp
from llm_eval.external.providers.hallulens.utils import generate_question as qa

TASKNAME = "longwiki"


def run_eval(
        do_extract_only: bool = False,
        model: str = 'meta-llama/Llama-3.1-405B-Instruct-FP8',
        claim_extractor: str = 'meta-llama/Llama-3.1-405B-Instruct-FP8',
        abstain_evaluator: str = "meta-llama/Llama-3.1-70B-Instruct",
        verifier: str = 'meta-llama/Llama-3.1-405B-Instruct-FP8',
        eval_cache_path: str = None,
        db_path: str = "data/wiki_data/.cache/enwiki-20230401.db",
        k: int = 32,
):
    TASKNAME = "longwiki"

    model_name = model.split("/")[-1]
    output_folder = Path(f'output/{TASKNAME}-dynamic/{model_name}')
    output_csv = output_folder / "output.csv"
    generations_file_path = output_folder / "generation.jsonl"
    base_path = os.path.dirname(os.path.abspath(__name__))
    eval_cache_path = f"{base_path}/data/longwiki/.cache" if eval_cache_path is None else eval_cache_path

    facthalu = FactHalu(generations_file_path,
                        output_csv,
                        abstain_evaluator=abstain_evaluator,
                        claim_extractor=claim_extractor,
                        verifier=verifier,
                        k=k,
                        eval_cache_path=eval_cache_path,
                        db_path=db_path,
                        do_extract_only=do_extract_only
                        )

    # save all evalaution details
    eval_details = {
        "output_csv": str(output_csv),
        "abstain_evaluator": abstain_evaluator,
        "claim_extractor": claim_extractor,
        "verifier": verifier,
        "k": k,
        "evalauted_model": model_name,
        "exp_mode": "dynamic",
        "eval_time": str(pd.Timestamp.now())
    }

    with open(output_folder / "eval_details.json", 'w') as f:
        json.dump(eval_details, f)

    score, overall_result_df = facthalu.run()
    eval_result_dict = {"model": model_name, **score, 'meta_data': eval_details}

    overall_result_df['model'] = model_name
    overall_result_df = overall_result_df[['model',
                                           'overall_recall', 'overall_precision', 'overall_f1',
                                           'prompt', 'is_supported', 'claim',
                                           'sentence', 'title',
                                           'precision', 'recall', 'f1', 'k', 'n_claims', ]]
    return eval_result_dict, overall_result_df
