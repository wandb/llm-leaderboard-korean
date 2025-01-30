from typing import List, Dict, Any, Union
from llm_eval.models.base import BaseModel
from llm_eval.models.multi import MultiModel # for llm as a judge

class BaseEvaluator:
    """
    Base class that all Evaluator classes must inherit from.

    Key concepts:
    - prepare_prompt: (Optional) Modifies the input prompt, inserts CoT, or performs other logic.
    - parse_prediction: (Optional) Parses the raw model output for easier comparison with the reference.
    - evaluate_predictions: Implements the metric calculation logic for evaluating predictions.
    - evaluate: (High-level logic) Iterates over data and follows the full flow of prompt preparation -> model call -> prediction parsing -> score calculation.

    If necessary, attributes like requires_logits and requires_chain_of_thought can be added
    to handle conditions where logits or CoT need to be requested from the model.
    """

    name: str = "base"
    requires_logits: bool = False
    requires_chain_of_thought: bool = False

    def prepare_prompt(self, input_text: str) -> str:
        """
        (Optional) Takes the input text and generates the final prompt to be fed into the model
        by attaching CoT instructions or additional system messages.

        The default implementation returns the input text unchanged.
        """
        return input_text

    def parse_prediction(self, raw_output: str) -> Any:
        """
        (Optional) Takes the raw output from the model (e.g., a string) and preprocesses/parses it
        to make comparison with the reference easier.

        Examples:
            - If the response is in JSON format, parse it.
            - If the response is in the format 'Answer: ...', extract only '...'.

        The default implementation returns the raw output unchanged.
        """
        return raw_output

    def evaluate_predictions(
        self, 
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        (Must be implemented)
        Computes the actual metrics on samples that already contain 'prediction' fields
        (i.e., after model inference).

        Args:
            samples: A list where each element has the format:
                {
                    "input": str,
                    "reference": str, 
                    "prompt": str,               # The actual prompt sent to the model after prepare_prompt
                    "prediction": Any,           # The result after parse_prediction
                    "logits" (optional): Any,    # If requires_logits=True, stores logits returned by the model
                    ...
                } 

        Returns:
            Dict[str, float]: {"metric_name": metric_value, ...} 
            Example: {"accuracy": 0.85, "f1": 0.76}
        """
        raise NotImplementedError("Subclasses must implement evaluate_predictions().")


    def evaluate(
        self,
        data: List[Dict[str, Any]],
        model: Union[BaseModel, MultiModel, None] = None
    ) -> Dict[str, Any]:
        """
        Args:
            data: A list of dictionaries in the format [{"input":..., "reference":..., "prediction":...}, ...].
                  - The 'prediction' field is already finalized (processed by Runner, ScalingMethod, etc.).
            model: If it is a MultiModel, judge_batch is called to add fields like judge_score.
                   If it is a BaseModel or None, this step is skipped.

        Returns:
            {
              "metrics": {...},
              "samples": [...]
            }
        """

        # 1) parse_prediction
        for sample in data:
            raw_pred = sample.get("prediction", "")
            sample["prediction"] = self.parse_prediction(raw_pred)

        # 2) (Optional) If MultiModel + Judge exists, call judge_batch
        if isinstance(model, MultiModel):
            # If multi_model is present and has a Judge model as a sub-item,
            # judge_batch() is called → each sample may be updated with {"judge_score": float, ...}.
            data = model.judge_batch(data)

        # 3) compute_metrics
        metrics = self.evaluate_predictions(data)
        return {
            "metrics": metrics,
            "samples": data
        }
