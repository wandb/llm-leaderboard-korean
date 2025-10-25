# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse

from tqdm.contrib.concurrent import thread_map
from llm_eval.external.providers.hallulens.utils import lm, eval_utils

from llm_eval.external.providers.hallulens.tasks.refusal_test.nonsense_name import NonsenseNameInference, NonsenseNameEval
from llm_eval.external.providers.hallulens.tasks.refusal_test.entities_generation import NonsenseMixedGeneration


class NonsenseMixedInference(NonsenseNameInference):
    def __init__(self, taskname, output_base_dir, generate_model, prompt_path, seed, method='vllm', limit=None, backend_kwargs=None):
        super().__init__(output_base_dir, generate_model, prompt_path, seed, method, limit=limit, backend_kwargs=backend_kwargs)

        self.output_base_dir = output_base_dir
        self.generate_model = generate_model
        self.inference_method = method
        self.prompt_path = prompt_path
        self.seed = seed
        self.TASKNAME = taskname #prompt_path.split('/')[-1].replace('.csv', '') #  f"{seed}_{N}.csv"
        print('INFER TASKNAME', self.TASKNAME)

class NonsenseMixedEval(NonsenseNameEval):
    def __init__(self, taskname, output_base_dir, model_path, prompt_path, med_safety_filtered_model=False, language='kor', evaluator: str | None = None):
        super().__init__(output_base_dir, model_path, prompt_path)

        self.prompt_path = prompt_path
        self.TASKNAME = taskname #prompt_path.split('/')[-1].replace('.csv', '') #  f"{seed}_{N}.csv"
        self.model_name = model_path.split("/")[-1]

        self.task_output_dir = f"{output_base_dir}/{self.TASKNAME}/{self.model_name}"
        self.generations_file_path = f'{self.task_output_dir}/generation.jsonl'
        self.res_path = f'{self.task_output_dir}/eval_results.json'
        self.eval_raw_path = f'{self.task_output_dir}/raw_eval_res.jsonl'

        self.med_safety_filtered_model = med_safety_filtered_model
        self.evaluator = evaluator or "meta-llama/Llama-3.1-8B-Instruct"

        print('EVAL TASKNAME', self.TASKNAME)

        if language == 'kor':
            import llm_eval.external.providers.hallulens.tasks.refusal_test.ko_prompt as prompt_templates
        elif language == 'eng':
            import llm_eval.external.providers.hallulens.tasks.refusal_test.prompt as prompt_templates
        else:
            raise NotImplementedError
        self.prompt_templates = prompt_templates

    def automatic_abstention(self, generations, evaluator_model="meta-llama/Llama-3.1-8B-Instruct"):
        JSON_KEY = "does_believe"

        eval_prompts = {
            'medicine' : self.prompt_templates.ABSTAIN_PROMPT_NONSENSE_MEDICINE,
            'animal' : self.prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
            'plant' : self.prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
            'bacteria' : self.prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
        }

        abstain_prompts = [
                eval_prompts.get(gen_obj['type']).format(
                    prompt=gen_obj['prompt'],
                    name=gen_obj['name'],
                    generation=gen_obj['generation'],
                )
                for gen_obj in generations
            ]
        
        # Route by evaluator name: llama* -> together, gpt-5* -> openai
        eval_name = str(self.evaluator).lower()
        if "gpt" in eval_name:
            gen_fn = lambda p: lm.openai_generate(p, self.evaluator)
        else:
            gen_fn = lambda p: lm.generate(p, self.evaluator)
        abstains_eval_raw = thread_map(
            gen_fn,
            abstain_prompts,
            max_workers=50,
            desc=f"using {self.evaluator}")
        
        if self.med_safety_filtered_model:
            for i, gen_obj in enumerate(generations):
                if gen_obj['type'] == 'medicine':
                    abstains_eval_raw[i] = "{\"does_believe\": false}"

        eval_utils.save_eval_raw(abstains_eval_raw, self.eval_raw_path)

        abstains_eval = eval_utils.jsonify_ans(raw_responses=abstains_eval_raw, \
                                                eval_prompts=abstain_prompts, \
                                                evaluator_model=evaluator_model,\
                                                key=JSON_KEY)
        abstains_eval_res = []
        for o in abstains_eval:
            abstains_eval_res.append(not o[JSON_KEY])
        
        return abstains_eval_res
