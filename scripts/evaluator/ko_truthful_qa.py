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
    kaster_metrics_dict,
    controllability_dict,
    task_to_sub_category,
    LLMAsyncProcessor,
    normalize,
    text_formatter,
    ko_truthful_qa_judge
)
from .abstract import AbstractEvaluator

class KoTruthfulQAEvaluator(AbstractEvaluator):
    def __init__(self, few_shots:bool):
        super().__init__(few_shots)
        self.dataset_name = "ko_truthful_qa"
        self.tasks = ["ko_truthful_qa-generation"]
        self.test_max_num_samples = 1 if self.cfg.testmode else 1000
        self.val_max_num_samples = 1 if self.cfg.testmode else 10

    def evaluate(self):
        evaluation_results = []
        # execute evaluation
        artifact_dir, dataset_dir = self.download_dataset(self.dataset_name)
        # collect test data
        for task in self.tasks:
            task_data, task_data_path = self.read_task_data(artifact_dir, dataset_dir, task)
            samples = task_data["samples"][:self.test_max_num_samples]
            eval_matainfo = {
                "model_name": self.cfg.model.pretrained_model_name_or_path,
                "dataset": self.dataset_name,
                "task": task,
                "subset": "test",
                "num_few_shots": self.num_few_shots,
            }
            for idx, sample in enumerate(samples):
                inputs = []
                # compose messages
                messages = []
                messages.append({"role": "user", "content": sample["question"]})

                # instruction message
                message_intro = "다음은 작업을 설명하는 지침과 컨텍스트 입력의 조합입니다. 요구를 적절하게 만족시키는 응답을 적으십시오."
                task_data_instruction = self.instructions[self.instructions['dataset'] == 'ko_truthful_qa-generation'].iloc[0]['instruction']
                    
                instruction = "\n".join(
                    [message_intro, task_data_instruction]
                )

                # Add instruction message at the beginning
                first_content = messages[0]["content"]
                messages[0]["content"] = f"{instruction}\n\n{first_content}"

                # generate output
                prompt = apply_chat_template(messages=messages)
                y_pred = None
                y_true: str = pipe(sample["best_answer"], normalize)
                metrics: str = self.instructions[self.instructions['dataset'] == 'haerae_bench_v1'].iloc[0]['metric']
                metrics_func: callable = ko_truthful_qa_judge

                generator_config = {"max_tokens": task_data["output_length"]}
                inputs.extend([messages, generator_config])

                # collect data
                evaluation_results.append(
                    {
                        **eval_matainfo,
                        "index": idx,
                        "input": sample["question"],
                        "raw_output": None,  # to be filled
                        "output": None,  # to be filled
                        "expected_output": y_true,
                        "prompt": prompt,
                        "metrics": metrics,
                        "metrics_func": metrics_func,
                        "score": None,  # to be filled
                        "inputs": inputs,
                        'question': sample["question"],
                        'best_answer': sample["best_answer"],
                        'correct_answer': sample["correct_answer"],
                    }
                )

        # Predict
        all_inputs = [er["inputs"] for er in evaluation_results]
        llm_ap = LLMAsyncProcessor(
            llm=self.llm,
            inputs=all_inputs,
        )
        result_data = []
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
            result_data.append(
                    {
                    'question': evaluation_result["question"],
                    'best_answer': evaluation_result["best_answer"],
                    'correct_answer': evaluation_result["correct_answer"],
                    'model_answer': y_pred,
                    }
                )
            metrics_func = evaluation_result["metrics_func"]
            evaluation_result["raw_output"] = raw_output
            evaluation_result["output"] = y_pred
        
        # Calculate score
        scores = metrics_func(result_data)
        for score, evaluation_result in zip(scores, evaluation_results):
            evaluation_result["score"] = score

        
        # Postprocess    
        output_df = pd.DataFrame(evaluation_results)
        output_df['sub_category'] = output_df['task'].map(task_to_sub_category)  
        dev_table = output_df.query("subset == 'dev'")
        test_table = output_df.query("subset == 'test'")

        leaderboard_table = pd.pivot_table(
            data=test_table,
            values="score",
            index="model_name",
            columns="task",
            aggfunc="mean",
            ).reset_index()

        leaderboard_table.insert(0, 'AVG', leaderboard_table.iloc[:, 1:].mean(axis=1))
        leaderboard_table.drop(columns=["model_name"], inplace=True)
        leaderboard_table.insert(0, 'model_name', self.cfg.model.pretrained_model_name_or_path)
        
        new_order=["model_name","task","index","input","raw_output","output","expected_output",
                "prompt","score","metrics","dataset","num_few_shots","subset","sub_category"]
        dev_table = dev_table[new_order]
        test_table = test_table[new_order]

        self.run.log(
            {
                f"{self.dataset_name}_{self.num_few_shots}shot_output_table_dev": dev_table,
                f"{self.dataset_name}_{self.num_few_shots}shot_output_table": test_table,
                f"{self.dataset_name}_{self.num_few_shots}shot_leaderboard_table": leaderboard_table,  # log later in kaster_translation.py
            }
        )
