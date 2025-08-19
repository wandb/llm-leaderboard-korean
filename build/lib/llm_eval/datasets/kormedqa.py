import logging
from typing import List, Dict, Any, Optional, Union

from datasets import load_dataset

from .base import BaseDataset
from . import register_dataset
from ..utils.logging import get_logger

logger = get_logger(name="kormedmcqa_dataset", level=logging.INFO)

# A default prompt template suitable for a professional medical exam questions.
DEFAULT_KORMED_PROMPT_TEMPLATE = (
    "The following is a multiple-choice question about specialized medical knowledge. "
    "Choose the most appropriate answer from the options provided.\n\n"
    "Question: {question}\n\n{choices_str}"
)

@register_dataset("kormedmcqa")
class KorMedMCQADataset(BaseDataset):
    """
    Dataset loader for KorMedMCQA: A Multi-Choice Question Answering Benchmark
    for Korean Healthcare Professional Licensing Examinations.

    - Loads data from the 'sean0042/KorMedMCQA' repository on the Hugging Face Hub.
    - Supports loading specific subsets ('dentist', 'doctor', 'nurse', 'pharm') or all of them.
        - None: Loads all four subsets.
        - str: Loads a single specified subset.
        - list: Loads multiple specified subsets.
    - Formats the prompt using the 'question' and choice columns (A, B, C, D, E).
    - The 'answer' column (a 1-based integer) is mapped to the corresponding choice text
      to be used as the 'reference'.
    """
    def __init__(
        self,
        dataset_name: str = "sean0042/KorMedMCQA",
        subset: Optional[Union[str, List[str]]] = None,
        split: str = "train",
        base_prompt_template: str = DEFAULT_KORMED_PROMPT_TEMPLATE,
        **kwargs,
    ):
        """
        Initializes the KorMedMCQADataset instance.

        Args:
            dataset_name (str): The name of the dataset on the HuggingFace Hub.
            subset (Optional[Union[str, List[str]]]): The subset(s) to load. Defaults to all subsets.
            split (str): The data split to load (e.g., 'train', 'test').
            base_prompt_template (str): A prompt template string for formatting the input.
            **kwargs: Additional arguments to be passed to `datasets.load_dataset`.
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
        Loads the data and returns it in the HRET standard list format.
        """
        subsets_to_load = self.subset
        # If no subset is specified, default to all available subsets.
        if subsets_to_load is None:
            subsets_to_load = ['dentist', 'doctor', 'nurse', 'pharm']
            logger.info(f"Subset is None. Loading all default subsets: {subsets_to_load}")

        if isinstance(subsets_to_load, str):
            subsets_to_load = [subsets_to_load]

        all_items = []
        for sub in subsets_to_load:
            try:
                # Load each subset using its name.
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
            question = item.get("question", "").strip()
            
            # Create a list of choice texts and a formatted string for the prompt.
            option_texts = []
            choices_str_parts = []
            for key in CHOICE_KEYS:
                choice_text = item.get(key)
                if choice_text is not None:
                    choice_text = str(choice_text).strip()
                    option_texts.append(choice_text)
                    choices_str_parts.append(f"({key}) {choice_text}")
            
            choices_str = "\n".join(choices_str_parts)

            # Format the final input prompt.
            final_input = template.format(
                question=question,
                choices_str=choices_str
            )

            # Convert the 1-based answer index to the corresponding choice text.
            answer_idx_val = item.get("answer")
            reference_text = ""
            if answer_idx_val is not None:
                try:
                    # Convert 1-based index to 0-based
                    idx = int(answer_idx_val) - 1
                    if 0 <= idx < len(option_texts):
                        reference_text = option_texts[idx]
                    else:
                        logger.warning(f"Answer index {answer_idx_val} is out of bounds for options in sample with question: '{question[:50]}...'")
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse answer index: {answer_idx_val}")
            
            processed_list.append({
                "input": final_input,
                "reference": reference_text,
                "options": option_texts,
                "_subset_name": subset_name,
                "metadata": {
                    "cot": item.get("cot")
                }
            })
            
        return processed_list

    def get_raw_samples(self) -> Any:
        """Returns the raw HuggingFace Dataset object for debugging purposes."""
        subsets_to_load = self.subset or ['dentist', 'doctor', 'nurse', 'pharm']
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
            "description": "KorMedMCQA: Multi-Choice Question Answering Benchmark for Korean Healthcare Professional Licensing Examinations.",
            "paper_url": "https://arxiv.org/abs/2403.01469",
            "evaluation_only": None
        }