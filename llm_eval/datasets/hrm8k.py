from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("hrm8k")
class HRM8KDataset(BaseDataset):
    """
    HRM8K Dataset Class.

    - subset: None -> Load the entire dataset
              list[str] -> Load and combine only the specified subsets
              str -> Load only the specified subset
    - Adds a '_subset_name' field to indicate from which subtask the data is loaded
      (to differentiate scores by subset during evaluation)

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
        dataset_name: str = "HAERAE-HUB/HRM8K",
        subset: Optional[Union[str, list]] = None,
        split: str = "test",
        **kwargs
    ):
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)
        
    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and returns it in the format:
        [{"input":..., "reference":..., "_subset_name":...}, ...]
        """
        # 1) Split based on the type of subset provided
        if self.subset is None:
            self.subset = ['GSM8K', 
                           'KSM', 
                           'MATH', 
                           'MMMLU', 
                           'OMNI_MATH']

        if isinstance(self.subset, list):
            # Case with multiple subsets
            all_items = []
            for sub in self.subset:
                partial_data = load_dataset(
                    self.dataset_name,
                    sub,
                    split=self.split,
                    **self.kwargs
                )
                all_items.extend(self._convert_to_list(partial_data, subset_name=sub))
            return all_items

        else:  # subset is a single string
            raw_data = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **self.kwargs
            )
            return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        Iterates over the HuggingFace Dataset (hf_dataset) and converts each entry to:
        {"input":..., "reference":...,  "_subset_name": subset_name}
        """
        processed_list = []
        # Fixed options A~E

        for item in hf_dataset:
            # Extract the 'question' and 'answer' fields from the original data (use empty string if missing)
            query = "put your final answer within \\boxed{}." + item.get("question", "")
            answer = item.get("answer", "")
            processed_list.append({
                "input": query.strip(),
                "reference": answer.strip(),
                "_subset_name": subset_name,
            })
        return processed_list

    def get_raw_samples(self) -> Any:
        """
        Returns the raw dataset.
        If multiple subsets are specified, returns a list; if a single subset or None is specified,
        returns a single Dataset (simple example).
        """
        if self.subset is None:
            return load_dataset(self.dataset_name, split=self.split, **self.kwargs)
        elif isinstance(self.subset, list):
            result = []
            for s in self.subset:
                partial = load_dataset(
                    self.dataset_name, s, split=self.split, **self.kwargs
                )
                result.append(partial)
            return result
        else:
            return load_dataset(
                self.dataset_name, 
                self.subset,
                split=self.split,
                **self.kwargs
            )

    def info(self) -> Dict[str, Any]:
        """
        Returns meta information related to the dataset.
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "HRM8K dataset. "
                "If subset is a list -> loads partial subsets, "
                "if subset is a string -> loads a single subset."
            ),
            "evaluation_only": None
        }
