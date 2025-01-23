from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("click")
class ClickDataset(BaseDataset):
    """
    Click 데이터셋 클래스

    * 로직:
      - input: paragraph + "\n문제: " question + "\n선택지: " + ", ".join(choices)
      - reference: answer
      - options: choices

    예시:
        ds = ClickDataset(
            dataset_name="EunsuKim/CLIcK",
            split="train"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "...\n질문: ...\n선택지: ...",
        #     "reference": "정답",
        #     "options": ["선택지1", "선택지2", ...]
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
        데이터를 로드하고
        [{"input":"...", "reference":"...", "options":[...]}, ...] 형태를 반환.
        """
        raw_data = load_dataset(self.dataset_name, split=self.split)
        return self._convert_to_list(raw_data)
        
    def _convert_to_list(self, hf_dataset) -> List[Dict[str, Any]]:
        """
        HuggingFace Dataset 객체(hf_dataset)를 순회하며,
        {"input":..., "reference":..., "options":[...]} 형태로 변환.
        """
        processed_list = []
        for item in hf_dataset:
            input_text = f"{item['paragraph']}\n질문: {item['question']}\n선택지: {', '.join(item['choices'])} \n답:"
            processed_list.append({
                "input": input_text,
                "reference": item['answer'],
                "options": item['choices']
            })  

        return processed_list

    def get_raw_samples(self) -> Any:
        """
        원본 데이터를 반환.
        """
        return load_dataset(
            self.dataset_name,
            split=self.split
            **self.kwargs
        )

    def info(self) -> Dict[str, Any]:
        """
        데이터셋 메타 정보를 반환.
        """
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "description": (
                "CLIcK Benchmark. https://arxiv.org/abs/2403.06412 "
                "Columns: paragraph, question, choices, answer. "
            ),
            "evaluation_only": None   
        }
        