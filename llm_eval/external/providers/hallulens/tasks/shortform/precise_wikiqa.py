# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import pandas as pd
from tqdm.contrib.concurrent import thread_map
import os

from llm_eval.external.providers.hallulens.utils import exp, lm, eval_utils


class PreciseQAEval:
    def __init__(self, model_path, TASKNAME, language: str = 'kor'):
        self.model_name = model_path.split("/")[-1]
        self.task_name = TASKNAME
        generations_file_path = f'output/{self.task_name}/{self.model_name}/generation.jsonl'

        self.output_path = f'output/{self.task_name}/{self.model_name}'
        self.test_df = pd.read_json(generations_file_path, lines=True)

        self.abtention_evaluator = 'meta-llama/Llama-3.1-70B-Instruct'
        self.halu_evaluator = 'meta-llama/Llama-3.1-70B-Instruct'

        if language == 'kor':
            from llm_eval.external.providers.hallulens.tasks.shortform.ko_prompt import KO_IS_HALLUCINATION_RESPONSE, \
                KO_ABSTAIN_PROMPT_UPDATED
            self.is_hallucination_response = KO_IS_HALLUCINATION_RESPONSE
            self.abstain_prompt_updated = KO_ABSTAIN_PROMPT_UPDATED
        elif language == 'eng':
            from llm_eval.external.providers.hallulens.tasks.shortform.en_prompt import IS_HALLUCINATION_RESPONSE, \
                ABSTAIN_PROMPT_UPDATED
            self.is_hallucination_response = IS_HALLUCINATION_RESPONSE
            self.abstain_prompt_updated = ABSTAIN_PROMPT_UPDATED
        else:
            raise ValueError(f"Invalid language: {language}")

    def eval_abstention(self, evaluator):
        print("Start abstantion evaluation")
        abs_path = f'{self.output_path}/abstain_eval_raw.jsonl'
        abstain_prompts = [
            self.abstain_prompt_updated.format(
                prompt=g.prompt, generation=g.generation
            )
            for _, g in self.test_df.iterrows()
        ]

        if os.path.exists(abs_path):
            # read from jsonl abspath
            with open(abs_path, "r") as f:
                abstains_eval_raw = [json.loads(line)["eval_res"] for line in f]
        else:
            abstains_eval_raw = thread_map(
                lambda p: lm.generate(p, evaluator),
                abstain_prompts,
                max_workers=32,
                desc=f"using {evaluator}")

            eval_utils.save_eval_raw(abstains_eval_raw, abs_path)

        ABSTAIN_JSON_KEY = 'is_abstaining'
        abstains_eval = eval_utils.jsonify_ans(raw_responses=abstains_eval_raw, \
                                               eval_prompts=abstain_prompts, \
                                               evaluator_model=evaluator, \
                                               key=ABSTAIN_JSON_KEY)
        refusal_res = []
        for o in abstains_eval:
            if ABSTAIN_JSON_KEY in o:
                refusal_res.append(o[ABSTAIN_JSON_KEY])
            else:
                refusal_res.append(False)
        self.test_df['refusal'] = refusal_res

        return refusal_res, abstains_eval_raw

    def judge_hallucination(self, evaluator):
        print("Starting Hallucination Evaluation")

        halu_prompts = [
            self.is_hallucination_response.format(
                prompt=g.prompt, generation=g.generation, gold_answer=g.answer
            ) for _, g in self.test_df.iterrows()
        ]

        if evaluator == "meta-llama/Llama-3.1-70B-Instruct":
            halu_eval_raw = thread_map(
                lambda p: lm.generate(p, evaluator),
                halu_prompts,
                max_workers=8,
                desc=f"using {evaluator}"
            )
        else:
            raise ValueError(f"Invalid evaluator: {evaluator}")

        return halu_eval_raw

    def process_res(self, abstantion_res_raw, halu_eval_raw):
        abstantion_res = [json.loads(x)['is_abstaining'] for x in abstantion_res_raw]
        halu_test_res = []
        for txt in halu_eval_raw:
            if txt.lower() not in ['correct', 'incorrect', 'unverifiable']: print(txt)
            hallucinated_judge = False if txt.lower() == 'correct' or txt.lower() == 'yes' else True
            halu_test_res.append(hallucinated_judge)
        return abstantion_res, halu_test_res

    def run_eval(self):
        abstantion_res, abstantion_raw_gen = self.eval_abstention(self.abtention_evaluator)
        halu_test_raw_gen = self.judge_hallucination(self.halu_evaluator)
        abstantion_res, halu_test_res = self.process_res(abstantion_raw_gen, halu_test_raw_gen)

        not_abstained = sum([1 for x in abstantion_res if x == False])
        if not_abstained == 0:
            hallu_rate_not_abstain = 0
        else:
            hallu_rate_not_abstain = sum([1 for is_abstaining, is_hallucinated in zip(abstantion_res, halu_test_res) \
                                          if is_abstaining == False and is_hallucinated == True]) / not_abstained
        refusal_rate = sum([1 for is_abstaining in abstantion_res if is_abstaining == True]) / len(abstantion_res)
        correct_rate = sum([1 for is_hallucinated in halu_test_res if is_hallucinated == False]) / len(halu_test_res)

        res = {
            'model': self.model_name,
            'halu_Rate': hallu_rate_not_abstain,
            'refusal_rate': refusal_rate,
            'correct_rate': correct_rate,

            'evaluator_abstantion': self.abtention_evaluator,
            'evaluator_hallucination': self.halu_evaluator,

            'abstantion': abstantion_res,
            'halu_test_res': halu_test_res,
            'abstantion_raw_generation': abstantion_raw_gen,
            'is_hallucinated_raw_generation': halu_test_raw_gen,
        }

        # save the results
        res_path = f'output/{self.task_name}/{self.model_name}/eval_results.json'
        with open(res_path, 'w') as f:
            json.dump(res, f, indent=4)

            # Print the results
        print("=" * 80)
        print(f" Evaluation Results for: <<{self.model_name}>>")
        print("=" * 80)
        print(f"  >> Results saved to: {res_path}")
        print("-" * 80)
        print(f"  Evaluator for Abstention: {self.abtention_evaluator}")
        print(f"  Evaluator for Hallucination: {self.halu_evaluator}")
        print("-" * 80)
        print(f"  Total Number of Samples: {len(abstantion_res)}")
        print(f"  Hallucination Rate (not abstained): {hallu_rate_not_abstain:.3f} %")
        print(f"  False Refusal Rate: {refusal_rate:.3f} %")
        print(f"  Correct Rate: {correct_rate:.3f} %")
        print("-" * 80)
        return res
