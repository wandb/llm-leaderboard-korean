import wandb
import pandas as pd
from llm_eval.utils.util import EvaluationResult

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