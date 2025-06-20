import logging
from typing import List, Dict, Any, Optional, Union

from datasets import load_dataset
from huggingface_hub import get_dataset_config_names

from .base import BaseDataset
from . import register_dataset
from ..utils.logging import get_logger

logger = get_logger(name="kbl_dataset", level=logging.INFO)

# Default prompt template for the KBL dataset, as per user request.
DEFAULT_KBL_PROMPT_TEMPLATE = "Question: {question}\nChoices: {choices}"

@register_dataset("kbl")
class KBLDataset(BaseDataset):
    """
    KBL(Korean Benchmark for Legal Language Understanding) Dataset Class.

    - Loads the 'lbox/kbl' dataset from the Hugging Face Hub.
    - It can load all or specific subsets via the 'subset' argument.
        - None: Dynamically loads and combines all subsets.
        - str: Loads a single specified subset.
        - list: Loads and combines multiple specified subsets.
    - It formats the prompt using the dataset's 'question' column and choice columns.
    - The 'gt' (ground truth) column is used as the reference answer.
    """
    def __init__(
        self,
        dataset_name: str = "lbox/kbl",
        subset: Optional[Union[str, List[str]]] = None,
        split: str = "test",
        base_prompt_template: str = DEFAULT_KBL_PROMPT_TEMPLATE,
        **kwargs,
    ):
        """
        Initializes an instance of the KBLDataset.

        Args:
            dataset_name (str): The name of the dataset on the HuggingFace Hub.
            subset (Optional[Union[str, List[str]]]): The name of the subset or a list of subset names.
                                                      If None, all subsets are loaded.
            split (str): The data split to load (defaults to 'test').
            base_prompt_template (str): The prompt template string for formatting the input.
            **kwargs: Additional arguments to be passed to `load_dataset`.
        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            subset=subset,
            base_prompt_template=base_prompt_template,
            **kwargs
        )

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and returns it in the HRET standard format.
        """
        subsets_to_load = self.subset
        # If subset is None, dynamically fetch the list of all subsets from the Hub.
        if subsets_to_load is None:
            try:
                logger.info(f"Subset is None. Fetching all available subsets from '{self.dataset_name}'...")
                subsets_to_load = get_dataset_config_names(self.dataset_name)
                logger.info(f"Found {len(subsets_to_load)} subsets to load.")
            except Exception as e:
                logger.error(f"Failed to fetch subsets for '{self.dataset_name}': {e}")
                return []
        
        # If subset is a single string, convert it to a list for iteration.
        if isinstance(subsets_to_load, str):
            subsets_to_load = [subsets_to_load]

        all_items = []
        for sub in subsets_to_load:
            try:
                # Load each subset by passing its name to the 'name' argument.
                partial_data = load_dataset(
                    self.dataset_name,
                    name=sub,
                    split=self.split,
                    **self.kwargs
                )
                all_items.extend(self._convert_to_list(partial_data, subset_name=sub))
            except Exception as e:
                logger.warning(f"Could not load subset '{sub}' for split '{self.split}'. Skipping. Error: {e}")
                continue
        
        return all_items

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        Converts a HuggingFace Dataset object to a list in the HRET standard format.
        """
        processed_list = []
        CHOICE_KEYS = ["A", "B", "C", "D", "E"]

        template = self.base_prompt_template

        for item in hf_dataset:
            # The main text from the 'question' column is used as the 'paragraph'.
            question = item.get("question", "")
            
            # Build the choices string and a list of the option texts.
            choices_parts = []
            option_texts = []
            for key in CHOICE_KEYS:
                if key in item and item[key] is not None:
                    choice_text = str(item[key]).strip()
                    choices_parts.append(f"({key}) {choice_text}")
                    option_texts.append(choice_text)
            
            choices_str = "\n".join(choices_parts)

            # Format the final input prompt.
            final_input = template.format(
                question=question.strip(),
                choices=choices_str
            )

            # The 'reference' answer is the letter from the 'gt' column.
            reference = str(item.get("gt", "")).strip()
            
            processed_list.append({
                "input": final_input,
                "reference": reference,
                "options": option_texts,  # List of choice texts for potential use in evaluators
                "_subset_name": subset_name,
                "metadata": item.get("meta")
            })
            
        return processed_list

    def get_raw_samples(self) -> Any:
        """Returns the raw HuggingFace Dataset object for debugging purposes."""
        if self.subset is None:
            # If no subset is specified, load the first one as an example.
            try:
                first_subset = get_dataset_config_names(self.dataset_name)[0]
                return load_dataset(self.dataset_name, name=first_subset, split=self.split, **self.kwargs)
            except Exception as e:
                logger.error(f"Failed to load raw samples for '{self.dataset_name}': {e}")
                return None
        elif isinstance(self.subset, list):
            return [load_dataset(self.dataset_name, name=s, split=self.split, **self.kwargs) for s in self.subset]
        else: # str
            return load_dataset(self.dataset_name, name=self.subset, split=self.split, **self.kwargs)

    def info(self) -> Dict[str, Any]:
        """Returns metadata about the dataset."""
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": "KBL: Korean Benchmark for Legal Language Understanding.",
            "paper_url": "https://arxiv.org/abs/2410.08731",
            "evaluation_only": None
        }