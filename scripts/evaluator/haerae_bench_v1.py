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
)

def get_few_shot_messages(target_dataset_path: str, num_few_shots: int):
    samples = json.loads(target_dataset_path.read_text(encoding="utf-8"))["samples"]

    few_shot_messages = []
    for i in range(num_few_shots):
        few_shot_messages.append({"role": "user", "content": samples[i]["input"]})
        few_shot_messages.append({"role": "assistant", "content": samples[i]["output"]})
    return few_shot_messages

def evaluate_n_shot(few_shots: bool):
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    # download dataset
    dataset_name = "haerae_bench_v1"
    artifact = run.use_artifact(cfg[dataset_name].artifacts_path, type="dataset")
    artifact_dir = artifact.download()
    # artifact_dir = "artifacts/eval_data:latest"
    dataset_dir = Path(artifact_dir) / cfg[dataset_name].dataset_dir
    print(dataset_dir)
    if not dataset_dir.exists():
        print(f"skip {dataset_name} because it is not found in {artifact_dir}")
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    #HI, KGK, LW, RC, RW, SN
    tasks = ['haerae_bench-HI',
        'haerae_bench-KGK',
        'haerae_bench-LW',
        'haerae_bench-RC',
        'haerae_bench-RW',
        'haerae_bench-SN']

    if few_shots:
        num_few_shots = cfg.get("num_few_shots", None)
        if (num_few_shots is None) or (num_few_shots == 0):
            return
    else:
        num_few_shots = 0

    evaluation_results = []
    for task in tasks:
        # execute evaluation
        for subset in ["test"]:
            eval_matainfo = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "dataset": dataset_name,
                "task": task,
                "num_few_shots": num_few_shots,
                "subset": subset,
            }

            # read task data
            task_data_path = dataset_dir / f"{task}.json"
            if not task_data_path.exists():
                print(f"skip {task} because it is not found in {artifact_dir}")
                continue
            with task_data_path.open(encoding="utf-8") as f:
                task_data = json.load(f)

            # number of evaluation samples
            if cfg.testmode:
                test_max_num_samples = 1
                val_max_num_samples = 1
            else:
                test_max_num_samples = 100
                val_max_num_samples = 10

            num_samples = test_max_num_samples
            samples = task_data["samples"][:num_samples]
            for idx, sample in enumerate(samples):
                inputs = []
                # compose messages
                messages = []

                # add fewshots samples
                if few_shots:
                    few_shot_messages = get_few_shot_messages(
                        target_dataset_path=task_data_path,
                        num_few_shots=num_few_shots,
                    )
                    messages.extend(few_shot_messages)
                    # user input
                    messages.append({"role": "user", "content": sample["input"][num_few_shots:]})
                else:
                    # user input
                    messages.append({"role": "user", "content": sample["input"]})

                # instruction message
                if "mmlu_en" in task:
                    message_intro = "The following text provides instructions for a certain task."
                else:
                    message_intro = "다음은 작업을 설명하는 지침과 컨텍스트 입력의 조합입니다. 요구를 적절하게 만족시키는 응답을 적으십시오."
                
                instruction = "\n".join(
                    [message_intro, task_data["instruction"]]
                )

                # Add instruction message at the beginning
                first_content = messages[0]["content"]
                messages[0]["content"] = f"{instruction}\n\n{first_content}"

                # generate output
                prompt = apply_chat_template(messages=messages)
                y_pred = None
                y_true: str = pipe(sample["output"], normalize)
                metrics: str = task_data["metrics"][0]
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
        llm=llm,
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
    output_df.to_csv('./test.csv')

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
    leaderboard_table_control.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)
    leaderboard_table.insert(0, 'AVG', leaderboard_table.iloc[:, 2:].mean(axis=1))
    leaderboard_table.drop(columns=["model_name"], inplace=True)
    leaderboard_table.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)
    
    new_order=["model_name","task","index","input","raw_output","output","expected_output",
               "prompt","score","control_score","metrics","control_method",
               "dataset","num_few_shots","subset","sub_category"]
    dev_table = dev_table[new_order]
    test_table = test_table[new_order]

    run.log(
        {
            f"{dataset_name}_{num_few_shots}shot_output_table_dev": dev_table,
            f"{dataset_name}_{num_few_shots}shot_output_table": test_table,
            f"{dataset_name}_{num_few_shots}shot_leaderboard_table": leaderboard_table,  # log later in kaster_translation.py
            f"{dataset_name}_control_{num_few_shots}shot_leaderboard_table": leaderboard_table_control,
        }
    )
        
def evaluate():
    evaluate_n_shot(few_shots=True)
    evaluate_n_shot(few_shots=False)