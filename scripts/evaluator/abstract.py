import json
from pathlib import Path
import pandas as pd
from toolz import pipe
from tqdm import tqdm
from config_singleton import WandbConfigSingleton
from .evaluate_utils import (
    apply_chat_template,
    task_to_sub_category,
    LLMAsyncProcessor,
    normalize,
    text_formatter,
)
from abc import ABC

class AbstractEvaluator(ABC):
    def __init__(self, few_shots:bool):
        self.retrieve_wandb_instance()
        self.dataset_name = None
        self.tasks = None
        self.test_max_num_samples = None
        self.val_max_num_samples = None
        self.few_shots = few_shots
        self.num_few_shots = self.cfg.get("num_few_shots", None) if few_shots else 0
        self.instructions = pd.read_csv("scripts/evaluator/instructions.csv")

    def retrieve_wandb_instance(self):
        # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
        self.instance = WandbConfigSingleton.get_instance()
        self.run = self.instance.run
        self.cfg = self.instance.config
        self.llm = self.instance.llm

    def read_task_data(self, artifact_dir, dataset_dir, task, subset=""):
        # read task data
        dataset_dir = dataset_dir.joinpath(subset)
        task_data_path = dataset_dir / f"{task}.json"
        if not task_data_path.exists():
            print(f"skip {task} because it is not found in {artifact_dir}")
            # raise FileNotFoundError(f"skip {task} because it is not found in {artifact_dir}")
        with task_data_path.open(encoding="utf-8") as f:
            task_data = json.load(f)
        return task_data, task_data_path

    def evaluate(self):
        pass