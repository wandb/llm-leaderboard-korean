from typing import List, Dict, Any, Optional

class BaseDataset:
    """
    Abstract base class that all dataset classes should inherit from.
    
    Purpose:
      1) To provide a consistent interface expected by the evaluation pipeline (especially for 'input' and 'reference'):
         1-1) input: The complete text that the LLM will receive, including prompt, context, and instructions.
         1-2) reference: The expected output, such as the correct answer, label, or gold output.
      2) To allow easy customization of dataset-specific loading/preprocessing logic.
      
    Required Method:
      - load(): Loads the data and finally returns a list of dictionaries in the format [{"input":..., "reference":...}, ...].
    
    Optional Methods:
      - get_raw_samples(): Provides access to the raw data.
      - info(): Provides metadata about the dataset.
    """

    def __init__(self, dataset_name: str, split: str = "test", subset: str = None, base_prompt_template : str = None, **kwargs):
        """
        Args:
            dataset_name (str):
                Unique identifier for the dataset.
            split (str):
                String to distinguish between train/validation/test splits (used during loading).
            subset (str):
                Subtask or configuration (e.g., "abstract_algebra").
            kwargs:
                Additional parameters needed for loading the dataset (e.g., HF config, version, authentication token, etc.).
        """
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.base_prompt_template = base_prompt_template
        self.num_samples = kwargs.pop("num_samples", None)
        self.limit = kwargs.pop("limit", None)
        self.dev = kwargs.pop("dev", False)
        self.kwargs = kwargs  # Store additional parameters for extensibility

    def load(self) -> List[Dict[str, Any]]:
        """
        (Required) Returns a list of data samples for the evaluation pipeline.
        Each element is a dictionary in the format {"input": str, "reference": str, (optional) ...}.
        """
        raise NotImplementedError("Subclasses must implement load().")

    def get_raw_samples(self) -> Any:
        """
        (Optional) Returns the raw data, or, if needed, provides access to the cached raw data object.
        """
        raise NotImplementedError("This is optional. Override if needed.")

    def info(self) -> Dict[str, Any]:
        """
        (Optional) Returns metadata about the dataset as a dictionary.
        """
        return {"name": self.dataset_name, "split": self.split}
