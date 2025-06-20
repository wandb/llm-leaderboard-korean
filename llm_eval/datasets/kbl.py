import logging
from typing import List, Dict, Any, Optional, Union

from datasets import load_dataset
# kept for get_raw_samples or other utilities.
from datasets import get_dataset_config_names

from .base import BaseDataset
from . import register_dataset
from ..utils.logging import get_logger

logger = get_logger(name="kbl_dataset", level=logging.INFO)

DEFAULT_KBL_PROMPT_TEMPLATE = "Question: {question}\nChoices: {choices}"

# When no specific subset is requested, only these will be loaded to prevent format errors.
DEFAULT_KBL_SUBSETS = [
    "bar_exam_civil_2012", "bar_exam_civil_2013", "bar_exam_civil_2014", "bar_exam_civil_2015",
    "bar_exam_civil_2016", "bar_exam_civil_2017", "bar_exam_civil_2018", "bar_exam_civil_2019",
    "bar_exam_civil_2020", "bar_exam_civil_2021", "bar_exam_civil_2022", "bar_exam_civil_2023",
    "bar_exam_civil_2024", "bar_exam_civil_2025",
    "bar_exam_criminal_2012", "bar_exam_criminal_2013", "bar_exam_criminal_2014", "bar_exam_criminal_2015",
    "bar_exam_criminal_2016", "bar_exam_criminal_2017", "bar_exam_criminal_2018", "bar_exam_criminal_2019",
    "bar_exam_criminal_2020", "bar_exam_criminal_2021", "bar_exam_criminal_2022", "bar_exam_criminal_2023",
    "bar_exam_criminal_2024", "bar_exam_criminal_2025",
    "bar_exam_public_2012", "bar_exam_public_2013", "bar_exam_public_2014", "bar_exam_public_2015",
    "bar_exam_public_2016", "bar_exam_public_2017", "bar_exam_public_2018", "bar_exam_public_2019",
    "bar_exam_public_2020", "bar_exam_public_2021", "bar_exam_public_2022", "bar_exam_public_2023",
    "bar_exam_public_2024", "bar_exam_public_2025",
    "bar_exam_responsibility_2010", "bar_exam_responsibility_2011", "bar_exam_responsibility_2012",
    "bar_exam_responsibility_2013", "bar_exam_responsibility_2014", "bar_exam_responsibility_2015",
    "bar_exam_responsibility_2016", "bar_exam_responsibility_2017", "bar_exam_responsibility_2018",
    "bar_exam_responsibility_2019", "bar_exam_responsibility_2020", "bar_exam_responsibility_2021",
    "bar_exam_responsibility_2022", "bar_exam_responsibility_2023",
]

@register_dataset("kbl")
class KBLDataset(BaseDataset):
    """
    KBL(Korean Benchmark for Legal Language Understanding) Dataset Class.

    - Loads the 'lbox/kbl' dataset from the Hugging Face Hub.
    - It can load a specific subset, a list of subsets, or a predefined default list.
        - None: Loads a default list of subsets known to have a consistent format.
        - str: Loads a single specified subset.
        - list: Loads and combines multiple specified subsets.
    - Formats the prompt using the dataset's 'question' column and choice columns.
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
            subset (Optional[Union[str, List[str]]]): The subset(s) to load.
                                                      If None, a predefined list of subsets is loaded.
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
        # If no subset is specified, default to the predefined list of compatible subsets
        # to avoid format errors from other subsets.
        if subsets_to_load is None:
            subsets_to_load = DEFAULT_KBL_SUBSETS
            logger.info(f"Subset is None. Loading the predefined default set of {len(subsets_to_load)} KBL subsets.")

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
                "options": option_texts,
                "_subset_name": subset_name,
                "metadata": item.get("meta")
            })
            
        return processed_list

    def get_raw_samples(self) -> Any:
        """Returns the raw HuggingFace Dataset object for debugging purposes."""
        subsets_to_load = self.subset or DEFAULT_KBL_SUBSETS
        if isinstance(subsets_to_load, list):
            return [load_dataset(self.dataset_name, name=s, split=self.split, **self.kwargs) for s in subsets_to_load]
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