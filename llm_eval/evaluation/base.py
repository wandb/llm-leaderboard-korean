from typing import List, Dict, Any, Union
from llm_eval.models.base import BaseModel
from llm_eval.models.multi import MultiModel # for llm as a judge
import logging

logger = logging.getLogger(__name__)

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
        # 자체 judge 로직을 갖고 있는 evaluator인 경우 건너뛰기 (LLMJudgeEvaluator 등)
        has_custom_judge = hasattr(self, 'has_custom_judge') and getattr(self, 'has_custom_judge', False)
        
        if isinstance(model, MultiModel) and model.judge_model is not None and not has_custom_judge:
            # judge_batch() is called → each sample may be updated with {"judge_score": float, ...}.
            logger.info(f"BaseEvaluator: Calling judge_batch via MultiModel for {len(data)} samples")
            data = model.judge_batch(data)
            
            # 판단이 완료되었음을 표시하는 플래그 추가
            judged_count = 0
            for sample in data:
                # 판단 결과가 있는지 확인하고 플래그 추가
                has_judge_result = False
                if "judge_score" in sample:
                    has_judge_result = True
                    # None 값 검사 및 경고 로그 추가
                    if sample["judge_score"] is None:
                        logger.warning(f"Received None judge_score from judge_batch for sample: {sample.get('id', 'unknown')}")
                elif "evaluation" in sample:
                    has_judge_result = True
                
                if has_judge_result:
                    # _judged_by_evaluator 플래그는 내부적으로만 사용하도록 설정
                    # 이후 프로세스에서 참조할 수 있도록 유지하나, 삭제 가능하게 '_'로 시작하는 이름 사용
                    sample["_judged_by_evaluator"] = True
                    judged_count += 1
                    
            logger.info(f"BaseEvaluator: Marked {judged_count}/{len(data)} samples as judged")

        # 3) compute_metrics
        metrics = self.evaluate_predictions(data)
        return {
            "metrics": metrics,
            "samples": data
        }
