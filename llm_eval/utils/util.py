import json
import pandas as pd
from typing import Any, Dict, List

class EvaluationResult:
    """
    Encapsulates the evaluation output:
    - metrics: Dict[str, float] or more complex structure
    - samples: List[Dict[str, Any]], each representing a data instance with 'prediction', 'reference', etc.
    - info: Dict containing pipeline metadata (model name, dataset, etc.)

    Provides helper methods for introspection, error analysis, 
    and dictionary-like convenience access.
    """
    def __init__(
        self,
        metrics: Dict[str, Any],
        samples: List[Dict[str, Any]],
        info: Dict[str, Any]
    ):
        self.metrics = metrics
        self.samples = samples
        self.info = info

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entire result to a dictionary (similar to the old JSON structure).
        """
        return {
            "metrics": self.metrics,
            "samples": self.samples,
            "info": self.info
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame where each row is a sample, with columns:
         - "input", "reference", "prediction"
         - Possibly flattened fields like "evaluation.is_correct"
         - Additional fields if they exist
        """
        df = pd.DataFrame(self.samples)
        if "evaluation" in df.columns:
            # Flatten 'evaluation' dict into separate columns 
            eval_df = df["evaluation"].apply(pd.Series)
            df = pd.concat([df.drop(columns=["evaluation"]), eval_df.add_prefix("eval_")], axis=1)
        return df

    def save_json(self, path: str):
        """
        Save the entire result (metrics, samples, info) to a JSON file.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def __repr__(self):
        return f"EvaluationResult(metrics={self.metrics}, info={self.info}, samples=[...])"

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to 'metrics', 'samples', 'info'.

        Example:
            result["metrics"] -> self.metrics
            result["samples"] -> self.samples
            result["info"]    -> self.info
        """
        if key == "metrics":
            return self.metrics
        elif key == "samples":
            return self.samples
        elif key == "info":
            return self.info
        else:
            raise KeyError(f"'{key}' is not a valid key. Use 'metrics', 'samples', or 'info'.")

    def __contains__(self, key: str) -> bool:
        """
        Check if a key is one of the valid fields: 'metrics', 'samples', 'info'.
        """
        return key in ["metrics", "samples", "info"]

    def get(self, key: str, default=None) -> Any:
        """
        Emulate dict.get() behavior. Returns the corresponding field if it exists,
        otherwise returns default.
        """
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
