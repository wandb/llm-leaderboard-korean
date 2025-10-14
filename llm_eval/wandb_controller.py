import wandb
import pandas as pd
from typing import Any, Dict, List, Tuple
from llm_eval.utils.util import EvaluationResult
import weave

class WandbController:
    def __init__(self, wandb_params: dict, dataset_name: str, model_name: str):
        self.wandb_params = wandb_params
        self.dataset_name = dataset_name
        self.model_name = model_name

    def log_to_wandb(self, result: EvaluationResult) -> None:
        """Log evaluation summary to Weights & Biases if configured."""
        table_name = self.dataset_name + "_leaderboard_table"
        data = {k: result.metrics.get(k) for k in {"model_name", "AVG", *result.metrics.keys()}}
        data["model_name"] = self.model_name
        df = pd.DataFrame([data])
        
        cols = ["model_name", "AVG"] + sorted([c for c in df.columns if c not in ["model_name", "AVG"]])
        df = df[cols]
        leaderboard_table = wandb.Table(dataframe=df)
        with wandb.init(
            entity=self.wandb_params.get("entity"),
            project=self.wandb_params.get("project"),
            name=self.model_name
            ) as run:
            run.log({table_name: leaderboard_table})


class WeaveSampleLogger:
    """Minimal helper to keep Weave Inputs and Outputs clean for per-sample traces.

    - Inputs: dataset_name, subset_name, input_text
    - Outputs: every field except 'input' (e.g., prediction, reference, evaluation columns)
    """

    _cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    @staticmethod
    def cache_sample(dataset_name: Any, item: Dict[str, Any]) -> None:
        input_text = str(item.get("input", ""))
        key = (str(dataset_name), input_text)
        WeaveSampleLogger._cache[key] = {
            'normalized_pred': item['evaluation']['normalized_pred'],
            'normalized_ref': item['evaluation']['normalized_ref'],
            'is_correct': item['evaluation']['is_correct'],
            }

    @staticmethod
    def make_op(op_name: str):
        @weave.op(name=op_name)
        def _op(dataset_name: str, subset_name: Any, input_text: str) -> Dict[str, Any]:
            key = (str(dataset_name), str(input_text))
            item = WeaveSampleLogger._cache.get(key, {})
            return {k: v for k, v in item.items() if k != "input"}

        return _op

    @staticmethod
    def log_samples(op_name: str, dataset_name: Any, samples: List[Dict[str, Any]]) -> None:
        op = WeaveSampleLogger.make_op(op_name=str(op_name))
        for s in samples or []:
            subset_name = s.get("_subset_name", None)
            WeaveSampleLogger.cache_sample(dataset_name, s)
            op(dataset_name, subset_name, str(s.get("input", "")))