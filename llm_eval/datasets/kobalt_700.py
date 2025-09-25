import re
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset

from . import register_dataset
from .base import BaseDataset


@register_dataset("kobalt_700")
class KoBALT700Dataset(BaseDataset):
    """
    KoBALT-700 Dataset Class for data sourced from 'HAERAE-HUB/KoSimpleEval'.

    This loader handles the 'KoBALT-700' subset, which features multiple-choice questions
    with lettered options up to 'J'.

    - The 'question' column, containing the full problem text with choices, is used as the 'input'.
    - The 'gold' column, a single letter (e.g., 'H'), is converted to a string like "(H)" for the 'reference'.
    - A fixed list of 'options' from "(A)" to "(J)" is provided for all samples to support log-probability-based evaluations.
    - The 'category' column (e.g., '의미론') is used as '_subset_name' for per-subject analysis.

    Usage example:
        ds = KoBALT700Dataset(split="test")
        data = ds.load()
        # data -> [
        #   {
        #     "input": "지문: 현진, 수빈아, 혹시 지금 시간 돼? 다음주 회의 관련해서 부탁할 게 있어서... ",
        #     "reference": "(H)",
        #     "options": ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"],
        #     "_subset_name": "의미론"
        #   }, ...
        # ]
    """
    # Define a fixed list of options from (A) to (J)
    DEFAULT_OPTIONS = [f"({chr(ord('A') + i)})" for i in range(10)]

    def __init__(
        self,
        dataset_name: str = "HAERAE-HUB/KoSimpleEval",
        subset: str = "KoBALT-700",
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        if base_prompt_template is None:
            base_prompt_template = (
                "다음은 언어학 관련 객관식 문제입니다. 제시된 지문과 질문, 그리고 선택지를 주의 깊게 읽고 가장 적절한 답 하나를 고르시오.\n\n"
                "{question}"
            )
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            subset=subset,
            base_prompt_template=base_prompt_template,
            **kwargs,
        )

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and returns it in the HRET standard list format.
        """
        raw_data = load_dataset(
            self.dataset_name, name=self.subset, split=self.split, **self.kwargs
        )
        return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(
        self, hf_dataset, subset_name: str
    ) -> List[Dict[str, Any]]:
        """
        Converts the HuggingFace Dataset object to a list in the HRET standard format.
        """
        processed_list = []
        for item in hf_dataset:
            question_text = item.get("question", "").strip()

            final_input = (
                self.base_prompt_template.format(question=question_text)
                if self.base_prompt_template
                else question_text
            )

            # Convert the gold letter (e.g., 'H') to the format "(H)".
            gold_letter = item.get("gold", "").strip()
            reference = f"({gold_letter})" if gold_letter else ""

            processed_list.append(
                {
                    "input": final_input,
                    "reference": reference,
                    "options": self.DEFAULT_OPTIONS,  # Use the fixed list of options
                    "_subset_name": item.get("category", "unknown"),
                    "metadata": {"original_gold": gold_letter},
                }
            )
        return processed_list

    def get_raw_samples(self) -> Any:
        """Returns the raw HuggingFace Dataset object for debugging."""
        return load_dataset(
            self.dataset_name, name=self.subset, split=self.split, **self.kwargs
        )

    def info(self) -> Dict[str, Any]:
        """Returns metadata about the dataset."""
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": "KoBALT-700 dataset from HAERAE-HUB/KoSimpleEval, formatted for HRET with fixed options A-J.",
            "evaluation_only": None,
        }