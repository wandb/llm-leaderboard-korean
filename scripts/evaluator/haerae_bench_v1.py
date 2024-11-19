import json
import time
from pathlib import Path
import numpy as np

import pandas as pd
from toolz import pipe
from tqdm import tqdm
import wandb

from config_singleton import WandbConfigSingleton
from .evaluate_utils import (
    apply_chat_template,
    get_few_shot_messages,
    kaster_metrics_dict,
    controllability_dict,
    task_to_sub_category,
    kmmlu_dict,
    LLMAsyncProcessor,
    normalize,
    text_formatter,
    evaluate_robustness,
    commet_score
)
from .abstract import AbstractEvaluator


class HaeraeBenchV1Evaluator(AbstractEvaluator):
    def __init__(self, few_shots:bool):
        super().__init__(few_shots)
        self.dataset_name = "haerae_bench_v1"
        self.tasks = ['haerae_bench-HI',
            'haerae_bench-KGK',
            'haerae_bench-LW',
            'haerae_bench-RC',
            'haerae_bench-RW',
            'haerae_bench-SN']
    
    def get_few_shot_messages(self, target_dataset_path: str, num_few_shots: int):
        samples = json.loads(target_dataset_path.read_text(encoding="utf-8"))["samples"]

        few_shot_messages = []
        for i in range(num_few_shots):
            few_shot_messages.append({"role": "user", "content": samples[i]["input"]})
            few_shot_messages.append({"role": "assistant", "content": samples[i]["output"]})
        return few_shot_messages

    def evaluate(self):
        evaluation_results = []
        # execute evaluation
        artifact_dir = Path(self.cfg.haerae_bench_v1.artifacts_path)
        dataset_dir = Path(self.cfg.haerae_bench_v1.artifacts_path + "/" + self.cfg.haerae_bench_v1.dataset_dir)
        # collect test data
        for task in self.tasks:
            if self.cfg.testmode:
                self.test_max_num_samples = 1
                self.val_max_num_samples = 1
            else:
                self.test_max_num_samples = 100
                self.val_max_num_samples = 10
            for subset in ("test", "dev"):
                if subset == "test":
                    num_samples = self.test_max_num_samples
                elif subset == "dev":
                    num_samples = self.val_max_num_samples
                task_data, task_data_path = self.read_task_data(artifact_dir, dataset_dir, task)
                samples = task_data["samples"][:num_samples]
                eval_matainfo = {
                    "model_name": self.cfg.model.pretrained_model_name_or_path,
                    "dataset": self.dataset_name,
                    "task": task,
                    "subset": subset,
                    "num_few_shots": self.num_few_shots,
                }
                for idx, sample in enumerate(samples):
                    inputs = []
                    # compose messages
                    messages = []

                    # add fewshots samples
                    if self.few_shots:
                        few_shot_messages = self.get_few_shot_messages(
                            target_dataset_path=task_data_path,
                            num_few_shots=self.num_few_shots,
                        )
                        messages.extend(few_shot_messages)
                        # user input
                        messages.append({"role": "user", "content": sample["input"][self.num_few_shots:]})
                    else:
                        # user input
                        messages.append({"role": "user", "content": sample["input"]})

                    # instruction message
                    if "mmlu_en" in task:
                        message_intro = "The following text provides instructions for a certain task."
                    else:
                        message_intro = "다음은 작업을 설명하는 지침과 컨텍스트 입력의 조합입니다. 요구를 적절하게 만족시키는 응답을 적으십시오."
                    task_data_instruction = self.instructions[self.instructions['dataset'] == 'haerae_bench_v1'].iloc[0]['instruction']
                    
                    instruction = "\n".join(
                        [message_intro, task_data_instruction]
                    )

                    # Add instruction message at the beginning
                    first_content = messages[0]["content"]
                    messages[0]["content"] = f"{instruction}\n\n{first_content}"

                    # generate output
                    prompt = apply_chat_template(messages=messages)
                    y_pred = None
                    y_true: str = pipe(sample["output"], normalize)
                    metrics: str = self.instructions[self.instructions['dataset'] == 'haerae_bench_v1'].iloc[0]['metric']
                    metrics_func: callable = kaster_metrics_dict[metrics]
                    control_task = "mmlu_en" if "mmlu_en" in task else "kmmlu" if "kmmlu" in task else task
                    control_method: str = controllability_dict[control_task].__name__
                    control_func: callable = controllability_dict[control_task]

                    generator_config = {"max_tokens": task_data["output_length"]}
                    inputs.extend([messages, generator_config])

                    # collect data
                    evaluation_results.append(
                        {
                            **eval_matainfo,
                            "index": idx,
                            "input": sample["input"],
                            "raw_output": None,  # to be filled
                            "output": None,  # to be filled
                            "expected_output": y_true,
                            "prompt": prompt,
                            "metrics": metrics,
                            "metrics_func": metrics_func,
                            "control_method": control_method,
                            "control_func": control_func,
                            "score": None,  # to be filled
                            "inputs": inputs,
                        }
                    )

        all_inputs = [er["inputs"] for er in evaluation_results]
        llm_ap = LLMAsyncProcessor(
            llm=self.llm,
            inputs=all_inputs,
        )
        results = llm_ap.get_results()
        for response, evaluation_result in tqdm(zip(results, evaluation_results)):
            raw_output = response.content
            y_pred: str = pipe(
                raw_output,
                lambda x: text_formatter(x, evaluation_result["task"]),
                lambda x: x.split("\n\n")[0],
                lambda x: x.strip(),
                lambda x: x.strip("'").strip('"'),
                lambda x: x.strip(),
                normalize,
            )
            metrics_func = evaluation_result["metrics_func"]
            control_func = evaluation_result["control_func"]
            score = metrics_func(y_pred, evaluation_result["expected_output"])        
            control_score = control_func(y_pred)
            evaluation_result["raw_output"] = raw_output
            evaluation_result["output"] = y_pred
            evaluation_result["score"] = score
            evaluation_result["control_score"] = control_score
            del evaluation_result["metrics_func"], evaluation_result["control_func"], evaluation_result["inputs"]
            
        output_df = pd.DataFrame(evaluation_results)

        # group mmlu_en and kmmlu task
        output_df['sub_category'] = output_df['task'].map(task_to_sub_category)  
        dev_table = output_df.query("subset == 'dev'")
        test_table = output_df.query("subset == 'test'")

        # calculate later in kaster_translation.py
        leaderboard_table = pd.pivot_table(
        data=test_table,
        values="score",
        index="model_name",
        columns="task",
        aggfunc="mean",
        ).reset_index()

        leaderboard_table_control = pd.pivot_table(
            data=test_table,
            values="control_score",
            index="model_name",
            columns="task",
            aggfunc="mean",
        ).reset_index()

        # leaderboard_table['AVG'] = leaderboard_table.iloc[:, 2:].mean(axis=1) # calculate later in kaster_translation.py
        leaderboard_table_control.insert(0, 'AVG', leaderboard_table_control.iloc[:, 2:].mean(axis=1))
        leaderboard_table_control.drop(columns=["model_name"], inplace=True)
        leaderboard_table_control.insert(0, 'model_name', self.cfg.model.pretrained_model_name_or_path)
        leaderboard_table.insert(0, 'AVG', leaderboard_table.iloc[:, 2:].mean(axis=1))
        leaderboard_table.drop(columns=["model_name"], inplace=True)
        leaderboard_table.insert(0, 'model_name', self.cfg.model.pretrained_model_name_or_path)
        
        new_order=["model_name","task","index","input","raw_output","output","expected_output",
                "prompt","score","control_score","metrics","control_method",
                "dataset","num_few_shots","subset","sub_category"]
        dev_table = dev_table[new_order]
        test_table = test_table[new_order]

        self.run.log(
            {
                f"{self.dataset_name}_{self.num_few_shots}shot_output_table_dev": dev_table,
                f"{self.dataset_name}_{self.num_few_shots}shot_output_table": test_table,
                f"{self.dataset_name}_{self.num_few_shots}shot_leaderboard_table": leaderboard_table,  # log later in kaster_translation.py
                f"{self.dataset_name}_control_{self.num_few_shots}shot_leaderboard_table": leaderboard_table_control,
            }
        )