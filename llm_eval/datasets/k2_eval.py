from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("k2_eval")
class K2_EvalDataset(BaseDataset):
    """
    K2-Eval Dataset Class

    - subset: None -> set to ["knowledge"]
              list[str] -> load and merge multiple subsets
              str -> load only the specified subset

    * Logic:
      - input: question + "\nOptions: a, b, c, d"
      - reference: the text of the option corresponding to gold_answer (0~3)
      - options: [a, b, c, d]

    Example:
        ds = K2_EvalDataset(
            dataset_name="HAERAE-HUB/K2-Eval",
            subset="knowledge",
            split="test"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "question\nOptions: ...",
        #     "reference": "...",  # the option text corresponding to the gold_answer index
        #     "options": ["...", "...", "...", "..."],
        #     "_subset_name": "knowledge"
        #   }, ...
        # ]
    """

    def __init__(
        self,
        dataset_name: str = "HAERAE-HUB/K2-Eval",
        subset: Optional[Union[str, list]] = None,
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)

    def load(self) -> List[Dict[str, Any]]:
        """
        Load the data and return it in the following format:
        [{"input": "...", "reference": "...", "options": [a, b, c, d], "_subset_name": ...}, ...]
        """
        # 1) Handle default value for subset
        if self.subset is None:
            # For K2-Eval, only the 'knowledge' subset is currently available (generate is not supported yet)
            self.subset = ["knowledge"]

        # When multiple subsets are provided
        if isinstance(self.subset, list):
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

        else:
            # When subset is a single string
            raw_data = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **self.kwargs
            )
            return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        Iterate over the HuggingFace Dataset (hf_dataset) and convert each item into the format:
        {"input": ..., "reference": ..., "options": [...], "_subset_name": subset_name}
        """
        processed_list = []

        for item in hf_dataset:
            question = item.get("question", "")
            gold_idx = item.get("gold_answer", -1)

            # Convert the option texts into a list
            choices = [
                item.get("a", ""),  # index 0
                item.get("b", ""),  # index 1
                item.get("c", ""),  # index 2
                item.get("d", ""),  # index 3
            ]

            # If gold_idx is within the range 0~3, use the corresponding option as the reference
            if 0 <= gold_idx < len(choices):
                reference_text = choices[gold_idx]
            else:
                reference_text = ""

            # Format the input text to include the question and options
            # Example: "question\nOptions: optionA, optionB, optionC, optionD\nAnswer:"
            input_text = question.strip() + "\nOptions: " + ", ".join(choices) + "\nAnswer:"

            processed_list.append({
                "input": input_text,
                "reference": reference_text,
                "options": choices,
                "_subset_name": subset_name,
            })

        return processed_list

    def get_raw_samples(self) -> Any:
        """
        Return the raw data.
        If multiple subsets are specified, returns a list of datasets;
        if a single subset or None is specified, returns a single Dataset.
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
        """Return meta-information about the dataset."""
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "K2_Eval benchmark featuring multiple-choice knowledge tasks. "
                "Columns: question, gold_answer (0~3), a, b, c, d. "
                "subset=list -> loads multiple subsets, subset=str -> loads a single subset."
            ),
            "evaluation_only": None,
        }
