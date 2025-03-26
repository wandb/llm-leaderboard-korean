from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset

from .base import BaseDataset
from . import register_dataset

@register_dataset("k2_eval")
class K2_EvalDataset(BaseDataset):
    """
    K2-Eval Dataset Class for Generation Subset

    - subset:
         None -> defaults to ["generation"]
         list[str] -> load and merge multiple subsets
         str -> load only the specified subset
    - For generation, only the generation subset is needed.
      It is assumed that the raw dataset contains the following columns:
          "instruction", "subject", "ability"
      base_prompt_template (if provided) will be used to format the final prompt using the instruction.
      
    Usage example:
        ds = K2_EvalDataset(
            dataset_name="HAERAE-HUB/K2-Eval",
            subset="generation",
            split="test",
            base_prompt_template="{instruction}"  # Optional, defaults to just instruction.
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "instruction text formatted via base_prompt_template",
        #     "reference": "",  # No gold answer for generation subset
        #     "subject": "...",
        #     "ability": "...",
        #     "_subset_name": "generation"
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
        # Save base_prompt_template as an attribute so that it can be used in conversion
        self.base_prompt_template = base_prompt_template
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset and returns a list of samples in the following format:
          {
            "input": <final prompt>,
            "reference": "",          # no reference for generation subset
            "subject": <subject text>,
            "ability": <ability text>,
            "_subset_name": <subset name>
          }
        """
        # 1) If subset is not provided, default to "generation"
        if self.subset is None:
            self.subset = ["generation"]

        # 2) If multiple subsets are provided, load and merge them
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
            # subset is a single string
            raw_data = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **self.kwargs
            )
            return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        Iterates over the HuggingFace Dataset and converts each item into the standardized format:
          {
            "input": final prompt (using base_prompt_template if provided),
            "reference": "",  # generation subset has no reference
            "subject": value from "subject" column,
            "ability": value from "ability" column,
            "_subset_name": subset_name
          }
        """
        processed_list = []
        for item in hf_dataset:
            instruction = item.get("instruction", "").strip()
            subject = item.get("subject", "").strip()
            ability = item.get("ability", "").strip()

            # Use base_prompt_template if provided, otherwise default to instruction only.
            if self.base_prompt_template:
                # 예: base_prompt_template="{instruction}"
                prompt = self.base_prompt_template.format(instruction=instruction)
            else:
                prompt = instruction

            processed_list.append({
                "input": prompt,
                "reference": "",
                "subject": subject,
                "ability": ability,
                "_subset_name": subset_name,
            })
        return processed_list

    def get_raw_samples(self) -> Any:
        """
        Returns the raw HuggingFace Dataset(s).
        If multiple subsets are specified, returns a list; otherwise a single Dataset.
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
        Returns metadata information about the dataset.
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "K2-Eval generation subset for LLM-as-a-Judge. "
                "Expected columns: instruction, subject, ability. "
                "The 'input' is generated using the base_prompt_template (if provided) and the instruction."
            ),
            "evaluation_only": ["llm_judge"]  # llm_judge 평가 방법만 허용
        }
