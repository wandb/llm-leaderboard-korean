from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("click")
class ClickDataset(BaseDataset):
    """
    Click Dataset Class

    * Logic:
      - input: paragraph + "\nQuestion: " + question + "\nChoices: " + ", ".join(choices)
      - reference: answer
      - options: choices

    Example:
        ds = ClickDataset(
            dataset_name="EunsuKim/CLIcK",
            split="train"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "...\nQuestion: ...\nChoices: ...",
        #     "reference": "answer",
        #     "options": ["choice1", "choice2", ...]
        #   },
        #   ...
        # ]
    """

    def __init__(
        self, 
        dataset_name: str = "EunsuKim/CLIcK", 
        split: str = "train", 
        **kwargs
    ):
        super().__init__(dataset_name, split)

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and returns it in the following format:
        [{"input": "...", "reference": "...", "options": [...]}, ...]
        """
        raw_data = load_dataset(self.dataset_name, split=self.split)
        return self._convert_to_list(raw_data)
        
    def _convert_to_list(self, hf_dataset) -> List[Dict[str, Any]]:
        """
        Iterates over the HuggingFace Dataset object (hf_dataset)
        and converts each item into the format:
        {"input": ..., "reference": ..., "options": [...]}
        """
        processed_list = []
        for item in hf_dataset:
            input_text = f"{item['paragraph']}\nQuestion: {item['question']}\nChoices: {', '.join(item['choices'])}"
            processed_list.append({
                "input": input_text,
                "reference": item['answer'],
                "options": item['choices']
            })  

        return processed_list

    def get_raw_samples(self) -> Any:
        """
        Returns the raw data.
        """
        return load_dataset(
            self.dataset_name,
            split=self.split,
            **self.kwargs
        )

    def info(self) -> Dict[str, Any]:
        """
        Returns metadata information about the dataset.
        """
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "description": (
                "CLIcK Benchmark. https://arxiv.org/abs/2403.06412 "
                "Columns: paragraph, question, choices, answer."
            ),
            "evaluation_only": None   
        }
