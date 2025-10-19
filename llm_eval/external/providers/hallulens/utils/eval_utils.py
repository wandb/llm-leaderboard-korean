# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import List

from llm_eval.external.providers.hallulens.utils import lm

def save_eval_raw(
        raw_eval_list: List[str], 
        output_file):
    # write raw evals to jsonl
    with open(output_file, "w") as f:
        for r in raw_eval_list:
            f.write(json.dumps({"eval_res": r}) + "\n")

def jsonify_ans(
        raw_responses: List[str],
        eval_prompts: List[str], 
        evaluator_model: str, 
        key: str):

    def check_validity(gen):
        gen_nospace = gen.replace(" ", "")
        if '{{"{}":false}}'.format(key) in gen_nospace:
            return '{{"{}":false}}'.format(key)
        elif '{{"{}":true}}'.format(key) in gen_nospace:
            return '{{"{}":true}}'.format(key)
        else:
            return -1
        
    jsonifyed_res  = []
    for r, p in zip(raw_responses, eval_prompts):
        
        if check_validity(r) != -1:
            jsonifyed_res.append(json.loads(check_validity(r)))
            continue
        else:
            r = r.split("\n")[0]
            try:
                jsonifyed_res.append(json.loads(r))
            except:
                print(f"Error in eval_answer: {r}")
                error = True
                error_count = 0
                
                while error:
                    try:
                        # Route by evaluator name: llama* -> together, gpt-5* -> openai
                        eval_name = str(evaluator_model).lower()
                        if "gpt" in eval_name:
                            re_eval = lm.openai_generate(p, evaluator_model)
                        else:
                            re_eval = lm.generate(p, evaluator_model)
                    except:
                        raise ValueError(f"Invalid evaluator: {evaluator_model}")
                        
                    try: 
                        print("\n** RETRY:", re_eval)
                        if check_validity(re_eval) != -1:
                            json_res = json.loads(check_validity(re_eval))
                        else:
                            json_res = json.loads(re_eval.split("\n")[0])
                        error = False
                        
                    except:
                        print("*** trying again** \n")
                        error = True
                    error_count += 1

                    if error_count > 3:
                        print("Error count exceeded 3. Skipping this prompt.")
                        jsonifyed_res.append({"error": "Error count exceeded 3. Skipping this prompt."})
                        break
                jsonifyed_res.append(json_res)
                print("<<< PASS >>>")
    return jsonifyed_res
