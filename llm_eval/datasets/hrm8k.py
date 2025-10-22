from typing import List, Dict, Any, Optional, Union
import os
import json
from .base import BaseDataset
from . import register_dataset

@register_dataset("hrm8k")
class HRM8KDataset(BaseDataset):
    """
    HRM8K Dataset Class.

    - subset: 
         None -> Load the entire dataset
         list[str] -> Load and combine only the specified subsets
         str -> Load only the specified subset
    - Adds a '_subset_name' field to indicate from which subtask the data is loaded,
      which is used to differentiate scores by subset during evaluation.

    Example usage:
        ds = HRM8KDataset(
            dataset_name="hrm8k",
            subset=["GSM8K", "KSM"],  # multiple subsets
            split="test"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "...",
        #     "reference": "...",
        #     "_subset_name": "GSM8K"
        #   }, ...
        # ]
    """
    def __init__(
        self, 
        dataset_name: str = "hrm8k",
        subset: Optional[Union[str, list]] = ["GSM8K", "KSM", "MATH", "MMMLU", "OMNI_MATH"],
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        """
        Initializes the HRM8K dataset.

        Args:
            dataset_name (str): Unique identifier for the dataset.
            subset (str or list[str] or None): Subset(s) to load.
            split (str): The dataset split, e.g., "train", "test", or "validation".
            base_prompt_template (str, optional): A prompt template to format the input. 
                If not provided, a default template is used.
            **kwargs: Additional parameters for dataset loading (e.g., HF config, revision, use_auth_token).
        """
        super().__init__(dataset_name, split=split, subset=subset, base_prompt_template=base_prompt_template, **kwargs)
        
    def _normalize_split(self, split: str) -> str:
        s = (split or "").lower()
        if s in ("train", "training"):
            return "train"
        if s in ("dev", "validation", "valid", "val"):
            return "dev"
        return "test"

    def _download_and_load(self) -> Dict[str, Any]:
        from llm_eval.wandb_singleton import WandbConfigSingleton
        artifact_dir = WandbConfigSingleton.download_artifact(self.dataset_name)
        file_path = os.path.join(artifact_dir, "hrm8k.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"hrm8k.json not found in artifact: {artifact_dir}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid hrm8k.json format: expected an object keyed by splits")
        return data

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset from artifact and converts it into a standardized format.

        Returns:
            List[Dict[str, Any]]: A list of processed samples.
        """
        # Default subsets when not specified
        if self.subset is None:
            self.subset = ['GSM8K', 'KSM', 'MATH', 'MMMLU', 'OMNI_MATH']

        raw = self._download_and_load()
        split_key = self._normalize_split(self.split)
        split_data = raw.get(split_key, {})
        if not isinstance(split_data, dict):
            raise ValueError(
                f"Invalid '{split_key}' split format: expected an object keyed by subsets"
            )

        processed_list: List[Dict[str, Any]] = []
        for sub in self.subset:
            items = split_data.get(sub, [])
            if not isinstance(items, list):
                continue
            processed_list.extend(self._convert_to_list(items, subset_name=sub))
        return processed_list

    def _convert_to_list(self, items, subset_name: str) -> List[Dict[str, Any]]:
        """
        Converts the HuggingFace Dataset object into a standardized format.

        English Comments:
        - Iterates over each sample in the HuggingFace dataset and constructs a formatted input using a prompt template.
        - If `self.base_prompt_template` is provided, it is used to format the prompt; otherwise, a default template is applied.
        - The default template instructs the model to "put your final answer within \\boxed{}" and then shows the question.
        - Each processed sample includes the "input", "reference", and "_subset_name" fields.

        Returns:
            List[Dict[str, Any]]: A list of processed samples.
        """
        processed_list = []
        # Define a default prompt template if none is provided.
        # default_template = "put your final answer within \\boxed{{}}.\nQuestion: {question}"
        default_template = "Solve the following problem. Briefly show your reasoning, then end with a single line in the form 'Answer: X'.\n\n{question}"
        
        for item in items:
            # Extract and clean the question text.
            raw_question = item.get("question", "").strip()
            # Use the provided base_prompt_template or fall back to the default template.
            template = self.base_prompt_template if self.base_prompt_template else default_template
            formatted_query = template.format(question=raw_question)
            
            # Extract the answer and ensure it is a string.
            answer = str(item.get("answer", ""))
            processed_list.append({
                "input": formatted_query.strip(),
                "reference": answer.strip(),
                "_subset_name": subset_name,
            })
            if getattr(self, "dev_mode", False) and len(processed_list) >= 2:
                break
            if getattr(self, "limit", None) and len(processed_list) >= self.limit:
                break
        return processed_list

    def get_raw_samples(self) -> Any:
        return self._download_and_load()

    def info(self) -> Dict[str, Any]:
        """
        Returns metadata information about the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing the dataset name, subset, split, and a description.
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self._normalize_split(self.split),
            "description": (
                "HRM8K dataset. "
                "If subset is a list -> loads multiple subsets, "
                "if subset is a string -> loads a single subset."
            ),
            "evaluation_only": None
        }
