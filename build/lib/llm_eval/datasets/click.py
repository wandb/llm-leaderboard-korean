from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("click")
class ClickDataset(BaseDataset):
    """
    Click Dataset Class

    Logic:
      - input: The paragraph, question, and choices are formatted into a prompt.
      - reference: The expected answer.
      - options: The list of choices.

    This dataset applies a prompt template to format the input. By default, the template is:
        "Paragraph: {paragraph}\nQuestion: {question}\nChoices: {choices}"
    Users may override this template by providing a custom base_prompt_template.

    Example usage:
        ds = ClickDataset(
            dataset_name="EunsuKim/CLIcK",
            split="train"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "Paragraph: ...\nQuestion: ...\nChoices: choice1, choice2, ...",
        #     "reference": "answer",
        #     "options": ["choice1", "choice2", ...]
        #   },
        #   ...
        # ]
    """

    def __init__(
        self, 
        dataset_name: str = "EunsuKim/CLIcK", 
        split: str = "train", 
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        """
        Initializes the Click dataset.

        Args:
            dataset_name (str): HuggingFace dataset identifier.
            split (str): Dataset split (e.g., "train", "test", "validation").
            base_prompt_template (str, optional): A custom prompt template for formatting the input.
                If not provided, a default template is used.
            **kwargs: Additional parameters for dataset loading.
        """
        super().__init__(dataset_name, split=split, base_prompt_template=base_prompt_template, **kwargs)

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and returns it in the following format:
        [{"input": "...", "reference": "...", "options": [...]}, ...].

        Returns:
            List[Dict[str, Any]]: Processed dataset samples.
        """
        raw_data = load_dataset(self.dataset_name, split=self.split)
        return self._convert_to_list(raw_data)
        
    def _convert_to_list(self, hf_dataset) -> List[Dict[str, Any]]:
        """
        Iterates over the HuggingFace Dataset object (hf_dataset) and converts each item
        into a standardized format:
            {"input": formatted_prompt, "reference": answer, "options": choices}

        English Comments:
            - This method constructs a formatted prompt using a base prompt template.
            - If self.base_prompt_template is provided, it is used; otherwise, a default template is applied.
            - The default template is: 
                  "Paragraph: {paragraph}\nQuestion: {question}\nChoices: {choices}"
              where {choices} is a comma-separated string of the available choices.

        Returns:
            List[Dict[str, Any]]: A list of processed samples.
        """
        processed_list = []
        # Default prompt template if none is provided
        default_template = "Paragraph: {paragraph}\nQuestion: {question}\nChoices: {choices}"
        template = self.base_prompt_template if self.base_prompt_template is not None else default_template

        for item in hf_dataset:
            # Extract and clean fields
            paragraph = item.get("paragraph", "").strip()
            question = item.get("question", "").strip()
            choices = item.get("choices", [])
            choices_str = ", ".join(choices)
            # Format the prompt using the template
            formatted_query = template.format(
                paragraph=paragraph,
                question=question,
                choices=choices_str
            )
            processed_list.append({
                "input": formatted_query,
                "reference": item.get("answer", "").strip(),
                "options": choices
            })
        return processed_list

    def get_raw_samples(self) -> Any:
        """
        Returns the raw dataset.

        Returns:
            Any: The raw dataset loaded from HuggingFace.
        """
        return load_dataset(
            self.dataset_name,
            split=self.split,
            **self.kwargs
        )

    def info(self) -> Dict[str, Any]:
        """
        Returns metadata information about the dataset.

        Returns:
            Dict[str, Any]: Metadata including dataset name, split, and a description.
        """
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "description": (
                "CLIcK Benchmark. https://arxiv.org/abs/2403.06412 "
                "Columns: paragraph, question, choices, answer."
            ),
            "evaluation_only": None
        }
