# llm_eval/datasets/kodarkbench.py

from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset, Dataset
import logging
from .base import BaseDataset
from . import register_dataset

logger = logging.getLogger(__name__)


@register_dataset("kodarkbench")
class KodarkbenchDataset(BaseDataset):
    """
    KoDarkBench Dataset Loader.

    This class loads the KoDarkBench dataset from HuggingFace and prepares
    it for the evaluation pipeline. It handles different subsets and formats
    the data into the required input/reference structure.

    Dataset Structure:
      - Input columns: id, input, ko-input, target, metadata
    """

    REQUIRED_COLUMNS = ["input", "target"]
    OPTIONAL_COLUMNS = ["ko-input", "metadata", "id"]

    def __init__(
        self,
        dataset_name: str = "shongdr/horangi_kodarkbench",
        subset: Optional[Union[str, List[str]]] = "default",
        split: str = "train",
        use_korean: bool = True,
        base_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """Initializes the KodarkbenchDataset."""
        super().__init__(
            dataset_name,
            split=split,
            subset=subset,
            base_prompt_template=base_prompt_template,
            **kwargs,
        )
        self.use_korean = use_korean
        logger.info(
            f"Initializing KodarkbenchDataset: "
            f"subset={self.subset}, split={self.split}, use_korean={self.use_korean}"
        )

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset from HuggingFace and converts it to a list of dicts.
        """
        try:
            if isinstance(self.subset, list):
                logger.info(f"Loading multiple subsets: {self.subset}")
                all_items = []
                for sub in self.subset:
                    partial_data = load_dataset(
                        self.dataset_name, sub, split=self.split, **self.kwargs
                    )
                    self._validate_columns(partial_data, sub)
                    converted = self._convert_to_list(partial_data, subset_name=sub)
                    all_items.extend(converted)
                processed_data = all_items
            else:
                logger.info(f"Loading single subset: {self.subset}")
                raw_data = load_dataset(
                    self.dataset_name, self.subset, split=self.split, **self.kwargs
                )
                self._validate_columns(raw_data, self.subset)
                processed_data = self._convert_to_list(raw_data, subset_name=self.subset)

            logger.info(f"✅ Loaded {len(processed_data)} samples for KoDarkBench.")
            return processed_data

        except Exception as e:
            logger.error(f"Failed to load dataset '{self.dataset_name}': {e}")
            raise ValueError(f"Error loading dataset: {e}")

    def _validate_columns(self, dataset: Dataset, subset_name: str) -> None:
        """Validates that the required columns exist in the dataset."""
        missing = [
            col for col in self.REQUIRED_COLUMNS if col not in dataset.column_names
        ]
        if missing:
            error_msg = (
                f"Subset '{subset_name}' is missing required columns: {missing}. "
                f"Available columns: {dataset.column_names}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _convert_to_list(
        self, hf_dataset: Dataset, subset_name: str
    ) -> List[Dict[str, Any]]:
        """Converts a HuggingFace Dataset object to our standard list format."""
        processed_list = []
        for idx, item in enumerate(hf_dataset):
            try:
                # Determine which input text to use
                if self.use_korean and "ko-input" in item and item.get("ko-input"):
                    instruction = str(item["ko-input"]).strip()
                else:
                    instruction = str(item.get("input", "")).strip()

                if not instruction:
                    logger.warning(f"Skipping sample {idx} due to empty instruction.")
                    continue

                target = str(item.get("target", "")).strip()
                metadata = item.get("metadata", {})
                subject = metadata.get("dark_pattern", target)

                # Apply prompt template if provided
                prompt = (
                    self.base_prompt_template.format(instruction=instruction)
                    if self.base_prompt_template
                    else instruction
                )

                processed_item = {
                    "input": prompt,
                    "reference": target,
                    "subject": subject,
                    "_subset_name": subset_name,
                    "_id": item.get("id", f"sample_{idx}"),
                }
                processed_list.append(processed_item)

            except Exception as e:
                logger.error(f"Error processing item {idx} in subset {subset_name}: {e}")
                continue
        return processed_list

    def info(self) -> Dict[str, Any]:
        """Returns metadata about the dataset."""
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "use_korean": self.use_korean,
            "description": "KoDarkBench: A dataset for evaluating dark patterns in LLMs.",
        }