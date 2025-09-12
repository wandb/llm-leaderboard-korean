from typing import Any, Dict, List, Optional

from datasets import load_dataset

from . import register_dataset
from .base import BaseDataset


@register_dataset("kmmlu_hard")
class KMMLUHardDataset(BaseDataset):
    """
    KMMLU-HARD Dataset Class for data sourced from 'HAERAE-HUB/KoSimpleEval'.

    This loader is designed for the 'KMMLU-HARD' subset which contains questions
    with numbered choices (1, 2, 3, 4) in the question text.

    - The 'question' column, containing the full problem text with choices, is used as the 'input'.
    - The integer 'gold' column (1-4) is converted to a string like "(4)" for the 'reference'.
    - The 'category' column (e.g., 'Accounting') is used as '_subset_name' for per-subject analysis.
    - Default 'options' are provided as `["(1)", "(2)", "(3)", "(4)"]` to support log-probability-based evaluations.

    Usage example:
        ds = KMMLUHardDataset(split="test")
        data = ds.load()
        # data -> [
        #   {
        #     "input": "(주)성공의 11월 말 현재 장부상에 계상된 현금과부족계정 차변 잔액은 ₩58,000이었다...",
        #     "reference": "(4)",
        #     "options": ["(1)", "(2)", "(3)", "(4)"],
        #     "_subset_name": "Accounting"
        #   }, ...
        # ]
    """

    def __init__(
        self,
        dataset_name: str = "HAERAE-HUB/KoSimpleEval",
        subset: str = "KMMLU-HARD",
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        if base_prompt_template is None:
            base_prompt_template = (
                "다음은 전문가 수준의 객관식 문제입니다. 제시된 문제와 선택지를 읽고 가장 적절한 답 하나를 고르시오.\n\n"
                "문제:\n{question}"
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
        # As the choices are 1-4, we provide them as default options.
        options = ["(1)", "(2)", "(3)", "(4)"]

        for item in hf_dataset:
            question_text = item.get("question", "").strip()

            final_input = (
                self.base_prompt_template.format(question=question_text)
                if self.base_prompt_template
                else question_text
            )

            # Convert the 1-based integer 'gold' answer to a string format like "(4)".
            gold_answer_index = item.get("gold")
            reference = f"({gold_answer_index})" if gold_answer_index is not None else ""

            processed_list.append(
                {
                    "input": final_input,
                    "reference": reference,
                    "options": options,
                    "_subset_name": item.get("category", "unknown"),
                    "metadata": {"original_gold_index": gold_answer_index},
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
            "description": "KMMLU-HARD dataset from HAERAE-HUB/KoSimpleEval, formatted for HRET with 4 numbered choices.",
            "evaluation_only": None,
        }