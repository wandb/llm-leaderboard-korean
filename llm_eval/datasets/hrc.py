import os
from typing import List, Dict, Any, Optional

from datasets import load_dataset

from llm_eval.datasets.base import BaseDataset
from llm_eval.datasets import register_dataset
from llm_eval.utils.logging import get_logger

logger = get_logger(name="hrc", level="INFO")


DEFAULT_HRC_PROMPT_TEMPLATE = "put your final answer within \\boxed{{}}.\nQuestion: {input}"

@register_dataset("hrc")
class HRCDataset(BaseDataset):
    """
    HRC (Haerae Reasoning Challenge) Dataset Class.

    This dataset class is designed to load data from the 'amphora/HRC'

    The class maps the original dataset columns to the format required by the HRET pipeline:
      - 'question' is mapped to 'input'.
      - 'answer' is mapped to 'reference'.
      - Other relevant columns like 'source' and 'category' are stored in 'metadata'.
    """

    def __init__(
        self,
        dataset_name: str = "amphora/HRC",
        split: str = "train",
        base_prompt_template: str = DEFAULT_HRC_PROMPT_TEMPLATE,
        **kwargs
    ):
        """
        Initializes the HRC dataset loader.

        Args:
            dataset_name (str): The HuggingFace dataset identifier.
            split (str): The dataset split to load (e.g., "train", "test").
            base_prompt_template (str): A custom prompt template to format the input.
                                        Defaults to a reasoning-style prompt.
            **kwargs: Additional keyword arguments for the dataset loading function.
        """
        super().__init__(dataset_name, split=split, base_prompt_template=base_prompt_template, **kwargs)
        
        # Retrieve the Hugging Face authentication token from environment variables.
        self.token = os.getenv("HUGGING_FACE_TOKEN")
        if not self.token:
            logger.warning(
                "Hugging Face token not found in environment variable 'HUGGING_FACE_TOKEN'. "
                "Loading 'amphora/HRC' may fail if it is a private or gated dataset."
            )

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the HRC dataset from the Hugging Face Hub using the provided
        authentication token and converts it into the standardized list format.

        Returns:
            List[Dict[str, Any]]: A list of processed samples, where each sample is a
                                  dictionary containing 'input', 'reference', and 'metadata'.
        
        Raises:
            Exception: If the dataset fails to load, often due to authentication
                       or network issues.
        """
        logger.info(f"Loading '{self.dataset_name}' dataset, split '{self.split}'.")
        try:
            # Use the token to load the dataset
            raw_data = load_dataset(self.dataset_name, split=self.split, token=self.token)
        except Exception as e:
            logger.error(
                f"Failed to load dataset '{self.dataset_name}'. "
                f"Please ensure you have access to the repository and that your "
                f"'HUGGING_FACE_TOKEN' environment variable is set correctly. Error: {e}"
            )
            raise e
        
        return self._convert_to_list(raw_data)

    def _convert_to_list(self, hf_dataset) -> List[Dict[str, Any]]:
        """
        Iterates over the loaded HuggingFace Dataset object and transforms each
        item into the standardized dictionary format required by the pipeline.
        """
        processed_list = []
        for item in hf_dataset:
            question = item.get("question", "").strip()
            answer = str(item.get("answer", "")).strip()
            final_input = self.base_prompt_template.format(input=question) if self.base_prompt_template else question

            processed_list.append({
                "input": final_input,
                "reference": answer,
                "metadata": {
                    "source": item.get("source"),
                    "category": item.get("category"),
                }
            })
        logger.info(f"Successfully converted {len(processed_list)} samples to the standard format.")
        return processed_list

    def get_raw_samples(self) -> Any:
        """
        Returns the raw HuggingFace Dataset object for debugging or inspection.

        Returns:
            Any: The raw dataset object loaded directly from `datasets.load_dataset`.
        """
        return load_dataset(
            self.dataset_name,
            split=self.split,
            token=self.token,
            **self.kwargs
        )

    def info(self) -> Dict[str, Any]:
        """
        Provides metadata about the dataset configuration.

        Returns:
            Dict[str, Any]: A dictionary containing descriptive metadata.
        """
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "description": "HRC dataset for evaluating high-level reasoning. "
                           "Maps 'question' to 'input' and 'answer' to 'reference'.",
            "evaluation_only": None
        }