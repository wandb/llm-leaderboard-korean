from typing import List, Dict, Any
from llm_eval.models.base import BaseModel

class BaseScalingMethod:
    """
    Scaling Method Base class.

    * Key Idea:
      1) Accept a data list (samples) and a model (BaseModel) as input,
      2) Update the "prediction" field using various candidate generation/search strategies,
      3) Return the same data list (or its copy).

    * Example data format:
      [
        {"input": str, "reference": str, ...},
        {"input": str, "reference": str, ...},
        ...
      ]

    * Return format:
      [
        {"input": ..., "reference": ..., "prediction": "...", ...},
        ...
      ]
    """

    def __init__(self, model: BaseModel = None, use_cot: bool = False, **kwargs):
        self.model = model
        self.use_cot = use_cot
        self.kwargs = kwargs  # e.g., n, beam_width, etc.

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for the scaling logic.
        Args:
            data: A list of dictionaries, e.g., [{"input": ..., "reference": ...}, ...]

        Returns:
            A list with the same structure as the input, but with the "prediction" field updated.
        """
        raise NotImplementedError("Subclasses must implement apply().")
    
    def set_params(self, **kwargs):
        """Update parameters (e.g., n, beam_width, etc.)."""
        self.kwargs.update(kwargs)
