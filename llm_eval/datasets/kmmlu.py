from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("kmmlu")
class KMMLUDataset(BaseDataset):
    """
    KMMLU Dataset Class.

    - subset: None -> Load the entire dataset
              list[str] -> Load and merge multiple specified subsets
              str -> Load a single specified subset
    - Each sample includes an 'options' field (if the original data does not have it, use ["(A)", "(B)", "(C)", "(D)", "(E)"])
    - A '_subset_name' field is added to indicate which subtask the data came from
      (used to calculate separate scores per subset during evaluation)

    Usage example:
        ds = KMMLUDataset(
            dataset_name="kmmlu",
            subset=["Accounting", "Biology"],  # multiple subsets
            split="test"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "...",
        #     "reference": "...",
        #     "options": ["(A)","(B)","(C)","(D)"],  # necessary for log-prob calculation
        #     "_subset_name": "Accounting"
        #   }, ...
        # ]
    """
    def __init__(
        self, 
        dataset_name: str = "HAERAE-HUB/KMMLU",
        subset: Optional[Union[str, list]] = None,
        split: str = "test",
        **kwargs
    ):
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)
        
    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and returns it in the form of 
        [{"input":..., "reference":..., "options":..., "_subset_name":...}, ...].
        """
        # 1) Split based on the type of subset
        if self.subset is None:
            self.subset = [
                'Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance', 'Biology',
                'Chemical-Engineering', 'Chemistry', 'Civil-Engineering', 'Computer-Science', 'Construction',
                'Criminal-Law', 'Ecology', 'Economics', 'Education', 'Electrical-Engineering', 'Electronics-Engineering',
                'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing',
                'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer',
                'Information-Technology', 'Interior-Architecture-and-Design', 'Law', 'Machine-Design-and-Manufacturing',
                'Management', 'Maritime-Engineering', 'Marketing', 'Materials-Engineering', 'Mechanical-Engineering',
                'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety',
                'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare',
                'Taxation', 'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math'
            ]

        if isinstance(self.subset, list):
            # When multiple subsets are provided
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

        else:  # when subset is a single string
            raw_data = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **self.kwargs
            )
            return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        Iterates over the HuggingFace Dataset object (hf_dataset) and converts each sample into the format:
        {"input":..., "reference":..., "options":..., "_subset_name": subset_name}.
        """
        processed_list = []
        # Fixed options A~D
        options = ["(A)", "(B)", "(C)", "(D)"]

        for item in hf_dataset:
            # Extract 'question' and 'answer' fields from the original data (if not present, use empty string)
            query = item.get('question', '') + f"\n(A) {item.get('A', '')}\n(B) {item.get('B', '')}\n(C) {item.get('C', '')}\n(D) {item.get('D', '')}"
            answer = options[item.get("answer", "") - 1]
            processed_list.append({
                "input": query.strip(),
                "reference": answer.strip(),
                "options": options,
                "_subset_name": subset_name,
            })
        return processed_list

    def get_raw_samples(self) -> Any:
        """
        Returns the raw data.
        If multiple subsets are specified, returns a list; if a single subset or None is specified, returns a single Dataset (for simplicity).
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
        Meta information related to the dataset.
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "KMMLU Benchmark. https://arxiv.org/abs/2402.11548 "
                "subset=list -> load partial subsets, "
                "subset=str -> load single subset."
            ),
            "evaluation_only": None
        }
