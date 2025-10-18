from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("hrm8k")
class HRM8KDataset(BaseDataset):
    """
    HRM8K Dataset Class.

    - subset: 
         None -> Load the entire dataset
         list[str] -> Load and combine only the specified subsets
         str -> Load only the specified subset
    - Adds a '_subset_name' field to indicate from which subtask the data is loaded,
      which is used to differentiate scores by subset during evaluation.

    Example usage:
        ds = HRM8KDataset(
            dataset_name="hrm8k",
            subset=["GSM8K", "KSM"],  # multiple subsets
            split="test"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "...",
        #     "reference": "...",
        #     "_subset_name": "GSM8K"
        #   }, ...
        # ]
    """
    def __init__(
        self, 
        dataset_name: str = "HAERAE-HUB/HRM8K",
        subset: Optional[Union[str, list]] = None,
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        """
        Initializes the HRM8K dataset.

        Args:
            dataset_name (str): Unique identifier for the dataset.
            subset (str or list[str] or None): Subset(s) to load.
            split (str): The dataset split, e.g., "train", "test", or "validation".
            base_prompt_template (str, optional): A prompt template to format the input. 
                If not provided, a default template is used.
            **kwargs: Additional parameters for dataset loading (e.g., HF config, revision, use_auth_token).
        """
        super().__init__(dataset_name, split=split, subset=subset, base_prompt_template=base_prompt_template, **kwargs)
        
    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset and converts it into a standardized format.

        Returns:
            List[Dict[str, Any]]: A list of processed samples, where each sample is a dictionary with:
                - "input": The formatted prompt.
                - "reference": The expected answer.
                - "_subset_name": The name of the subset from which the sample was loaded.
        """
        # If no subset is specified, use the default list of subsets.
        if self.subset is None:
            self.subset = ['GSM8K', 'KSM', 'MATH', 'MMMLU', 'OMNI_MATH']

        if isinstance(self.subset, list):
            # Load and combine multiple subsets.
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
            # Load a single subset.
            raw_data = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **self.kwargs
            )
            return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        Converts the HuggingFace Dataset object into a standardized format.

        English Comments:
        - Iterates over each sample in the HuggingFace dataset and constructs a formatted input using a prompt template.
        - If `self.base_prompt_template` is provided, it is used to format the prompt; otherwise, a default template is applied.
        - The default template instructs the model to "put your final answer within \\boxed{}" and then shows the question.
        - Each processed sample includes the "input", "reference", and "_subset_name" fields.

        Returns:
            List[Dict[str, Any]]: A list of processed samples.
        """
        processed_list = []
        # Define a default prompt template if none is provided.
        default_template = "put your final answer within \\boxed{{}}.\nQuestion: {question}"
        
        for item in hf_dataset:
            # Extract and clean the question text.
            raw_question = item.get("question", "").strip()
            # Use the provided base_prompt_template or fall back to the default template.
            template = self.base_prompt_template if self.base_prompt_template else default_template
            formatted_query = template.format(question=raw_question)
            
            # Extract the answer and ensure it is a string.
            answer = str(item.get("answer", ""))
            processed_list.append({
                "input": formatted_query.strip(),
                "reference": answer.strip(),
                "_subset_name": subset_name,
            })
            if getattr(self, "dev_mode", False) and len(processed_list) >= 10:
                break
        return processed_list

    def get_raw_samples(self) -> Any:
        """
        Returns the raw dataset.

        Returns:
            Any: If multiple subsets are specified, returns a list of Dataset objects;
                 if a single subset or None is specified, returns a single Dataset object.
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

        Returns:
            Dict[str, Any]: A dictionary containing the dataset name, subset, split, and a description.
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "HRM8K dataset. "
                "If subset is a list -> loads multiple subsets, "
                "if subset is a string -> loads a single subset."
            ),
            "evaluation_only": None
        }
