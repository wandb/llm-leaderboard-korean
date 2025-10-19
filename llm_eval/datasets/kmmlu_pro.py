from typing import Any, Dict, List, Optional

from datasets import load_dataset

from . import register_dataset
from .base import BaseDataset


@register_dataset("kmmlu_pro")
class KMMLUProDataset(BaseDataset):
    """
    KMMLU-Pro Dataset Class for data sourced from 'HAERAE-HUB/KoSimpleEval'.

    This loader is specifically designed for the 'KMMLU-Pro' subset, which has a
    different structure from the original KMMLU dataset.

    - The 'question' column, containing the full problem text with choices, is used as the 'input'.
    - The integer 'gold' column (1-5) is converted to a string like "(1)" for the 'reference'.
    - The 'category' column (e.g., '민법') is used as '_subset_name' for detailed per-subject analysis.
    - Default 'options' are provided as `["(1)", "(2)", "(3)", "(4)", "(5)"]` to support log-probability-based evaluations.

    Usage example:
        ds = KMMLUProDataset(split="test")
        data = ds.load()
        # data -> [
        #   {
        #     "input": "민법의 법원(法源)에 관한 설명으로 옳지 않은 것은?...",
        #     "reference": "(4)",
        #     "options": ["(1)", "(2)", "(3)", "(4)", "(5)"],
        #     "_subset_name": "민법"
        #   }, ...
        # ]
    """

    def __init__(
        self,
        dataset_name: str = "HAERAE-HUB/KoSimpleEval",
        subset: str = "KMMLU-Pro",  # Fixed subset for this loader
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        # Set a default prompt template suitable for professional MCQA.
        if base_prompt_template is None:
            base_prompt_template = (
                "다음은 전문 분야의 객관식 문제입니다. 제시된 문제와 선택지를 읽고 가장 적절한 답 하나를 고르시오.\n\n"
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
        # For this specific dataset, the 'subset' from our class corresponds to the 'name' parameter in load_dataset.
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
        # As the choices are 1-5, we provide them as default options.
        options = ["(1)", "(2)", "(3)", "(4)", "(5)"]

        for item in hf_dataset:
            question_text = item.get("question", "").strip()

            # Apply the base prompt template if provided.
            final_input = (
                self.base_prompt_template.format(question=question_text)
                if self.base_prompt_template
                else question_text
            )

            # Convert the integer 'gold' answer to a string format like "(4)".
            gold_answer_index = item.get("gold")
            reference = f"({gold_answer_index})" if gold_answer_index else ""

            processed_list.append(
                {
                    "input": final_input,
                    "reference": reference,
                    "options": options,
                    "_subset_name": item.get("category", "unknown"),
                    "metadata": {"original_gold_index": gold_answer_index},
                }
            )
            if getattr(self, "dev_mode", False) and len(processed_list) >= 10:
                break
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
            "description": "KMMLU-Pro dataset from HAERAE-HUB/KoSimpleEval, formatted for HRET.",
            "evaluation_only": None,
        }