# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm.contrib.concurrent import thread_map
from llm_eval.external.providers.hallulens.utils import exp, lm, eval_utils
import os
import json
import pandas as pd
import re

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


class NonsenseNameInference:
    """method = together, openai, vllm, custom"""
    def __init__(self, output_base_dir, generate_model, prompt_path, seed, method='together', limit=None):
        self.output_base_dir = output_base_dir
        self.generate_model = generate_model
        self.inference_method = method
        self.prompt_path = prompt_path
        self.seed = seed
        self.limit = limit
        self.TASKNAME = prompt_path.split('/')[-1].replace('_all_not_exist.csv', '') #  f"{seed}_{BUSINESS_N}_{EVENT_N}_{PRODUCT_N}"
        print('INFER TASKNAME', self.TASKNAME)
    
    def run_inference(self):
        generate_model = self.generate_model
        print('generate_model', generate_model)
        TASKNAME = self.TASKNAME
        # prompt_path = f"{self.root_path}/save/{self.seed}_{self.BUSINESS_N}_{self.EVENT_N}_{self.PRODUCT_N}_all_not_exist.csv"
        all_prompts = pd.read_csv(self.prompt_path)
        # apply random subsampling if limit is provided
        import os
        env_seed = int(os.environ.get('RANDOM_SEED', str(self.seed)))
        if self.limit is not None and len(all_prompts) > self.limit:
            all_prompts = all_prompts.sample(n=self.limit, random_state=env_seed)

        # This assertion is to check if generated prompts are Korean.
        assert any(re.search('[가-힣]', str(value)) for _, row in all_prompts.iterrows() for value in row), (
            "This prompt file does not seem to be Korean.\n"
            f"{all_prompts} should contain Korean prompts."
        )

        exp.run_exp(task=TASKNAME, 
                    model_path=generate_model, 
                    all_prompts=all_prompts, 
                    inference_method=self.inference_method, 
                    max_tokens=256, 
                    base_path=self.output_base_dir)
        print(TASKNAME, 'Inference completed')

    def remove_existing_files(self):
        model_name = self.generate_model.split("/")[-1]
        generations_file_path = f"{self.output_base_dir}/{self.TASKNAME}/{model_name}/generation.jsonl"
        results_file_path = f"{self.output_base_dir}/{self.TASKNAME}/{model_name}/eval_results.json"
        remove_file(generations_file_path)
        remove_file(results_file_path)

class NonsenseNameEval:
    def __init__(self, output_base_dir, model_path, prompt_path, language='kor'):
        self.prompt_path = prompt_path
        self.language = language
        self.TASKNAME = prompt_path.split('/')[-1].replace('_all_not_exist.csv', '') #  f"{seed}_{BUSINESS_N}_{EVENT_N}_{PRODUCT_N}"
        print('EVAL TASKNAME', self.TASKNAME)
        self.model_name = model_path.split("/")[-1]
        self.task_output_dir = f"{output_base_dir}/{self.TASKNAME}/{self.model_name}"
        self.generations_file_path = f'{self.task_output_dir}/generation.jsonl'
        self.res_path = f'{self.task_output_dir}/eval_results.json'
        self.eval_raw_path = f'{self.task_output_dir}/raw_eval_res.jsonl'
        self.evaluator = "meta-llama/Llama-3.1-70B-Instruct"

        if self.language == 'kor':
            import llm_eval.external.providers.hallulens.tasks.refusal_test.ko_prompt as prompt_templates
        elif self.language == 'eng':
            import llm_eval.external.providers.hallulens.tasks.refusal_test.prompt as prompt_templates
        else:
            raise ValueError("language should be 'kor' or 'eng'")
        self.prompt_templates = prompt_templates

    def automatic_abstention(self, generations, evaluator_model="meta-llama/Meta-Llama-3.1-70B-Instruct"):
        abstain_prompts = [
                self.prompt_templates.ABSTAIN_PROMPT_PLACE_NONSENSE.format(
                    name=generation['name'], 
                    TYPE=generation['type_'],
                    PLACE=" in " + generation['place'] if generation['place'] else "",
                    generation=generation['generation'],
                )
                for generation in generations
            ]
        abstains_eval_raw = thread_map(lambda p: lm.generate(p, self.evaluator),
                                        abstain_prompts,
                                        max_workers=50,
                                        desc=f"using {self.evaluator}")
                        
        eval_utils.save_eval_raw(abstains_eval_raw, self.eval_raw_path)

        abstains_eval = eval_utils.jsonify_ans(raw_responses=abstains_eval_raw, \
                                    eval_prompts=abstain_prompts, \
                                        evaluator_model=evaluator_model,\
                                            key="does_believe")
        abstains_eval_res = []
        for o in abstains_eval:
            try:
                abstains_eval_res.append(not o['does_believe'])
            except:
                print(f"Error in eval_answer: {o}")
                exit()

        return abstains_eval_res

    def run_eval(self, overwrite=False):
        if os.path.exists(self.res_path) and not overwrite:
            print(f'{self.TASKNAME} Evaluation already completed')
            res = json.load(open(self.res_path, "r"))
            return res
        
        generations = [json.loads(line) for line in open(self.generations_file_path, "r")]
        eval_results = self.automatic_abstention(generations)
        refusal_rate = sum(eval_results) / len(eval_results)

        res = {
            'model': self.model_name,
            'false_acceptance_rate': 1 - refusal_rate,
            'refusal_rate': refusal_rate,
            'refusal_eval_raw': eval_results,
        }
        # save the results
        with open(self.res_path, 'w') as f:
            json.dump(res, f, indent=4)

        print()
        print(f'*** {self.TASKNAME} Evaluation completed')
        # Print the results 
        print("=" * 80)
        print(f" Evaluation Results for: <<{self.model_name}>>")
        print("=" * 80)
        print(f"  >> Results saved to: {self.res_path}")
        print("-" * 80)
        print(f"  Evaluator for Abstention: {self.evaluator}")
        print("-" * 80)
        print(f"  Total Number of Samples: {len(generations)}")
        print(f"  False Acceptance Rate: {1 - refusal_rate:.3f} %")
        print("-" * 80)

        return res, self.task_output_dir
