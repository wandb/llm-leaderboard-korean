"""
Improved Weave integration with proper evaluation tracking.

This module provides two approaches for Weave evaluation:
1. Enhanced EvaluationLogger with scorer integration
2. Standard Weave Evaluation framework implementation
"""

import wandb
import weave
from weave import EvaluationLogger, Model, Evaluation
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EnhancedWeaveController:
    """
    Enhanced Weave controller that properly logs evaluation scores for each prediction.
    This improves upon the existing EvaluationLogger approach by integrating scoring.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: Any,
        split: str,
        model_backend_name: str,
        model_name: str = None,
        evaluation_method_name: str = None,
        evaluator: Optional[Any] = None,  # BaseEvaluator instance
        batch_size: int = 32,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.model_backend_name = model_backend_name
        self.model_name = model_name or model_backend_name
        self.evaluation_method_name = evaluation_method_name
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.evaluation_logger = None

    def initialize(self) -> EvaluationLogger:
        """Initialize Weave EvaluationLogger."""
        # Build subset representation
        if isinstance(self.subset, list):
            subset_repr = "+".join(map(str, self.subset)) if self.subset else "all"
        else:
            subset_repr = str(self.subset) if self.subset is not None else "all"

        # Get model name for labeling
        model_label = str(self.model_name).replace("-", "_").replace(" ", "_").replace(".", "_")

        # Build metadata
        metadata: Dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "model_name": self.model_name,
            "evaluation_method_name": self.evaluation_method_name,
        }

        # Initialize EvaluationLogger
        self.evaluation_logger = EvaluationLogger(
            dataset=str(self.dataset_name),
            model=model_label,
            eval_attributes=metadata,
        )

        logger.info("Initialized Enhanced Weave Controller with evaluation scoring")
        return self.evaluation_logger

    def run_inference_and_evaluation(
        self,
        samples: List[Dict[str, Any]],
        inference_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
        scoring_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run inference and evaluation with proper Weave logging.

        Args:
            samples: Input samples
            inference_fn: Model inference function
            scoring_fn: Optional custom scoring function. If not provided, uses self.evaluator

        Returns:
            List of samples with predictions and evaluation scores
        """
        if self.evaluation_logger is None:
            raise RuntimeError("Must call initialize() before run_inference_and_evaluation()")

        def process_single_sample(sample_with_idx: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
            """Process a single sample with inference and evaluation."""
            idx, sample = sample_with_idx
            try:
                # Prepare inputs for logging
                input_text = sample.get("input", "")
                reference_text = sample.get("reference", None)

                inputs_payload = {"input": input_text}
                if reference_text is not None:
                    inputs_payload["reference"] = reference_text

                # Start prediction logging
                pred_logger = self.evaluation_logger.log_prediction(
                    inputs=inputs_payload,
                    output=""  # Will be updated after inference
                )

                # Run inference
                result_list = inference_fn([sample])
                result = result_list[0] if result_list else sample
                prediction_text = result.get("prediction", "")

                # Update the output in the logger
                try:
                    pred_logger.output = prediction_text
                except (AttributeError, TypeError):
                    if hasattr(pred_logger, '_output'):
                        pred_logger._output = prediction_text

                # Calculate evaluation scores
                if scoring_fn:
                    scores = scoring_fn(result)
                elif self.evaluator:
                    # Use the evaluator to calculate scores for this single prediction
                    eval_result = self._evaluate_single_sample(result)
                    scores = eval_result
                else:
                    # Default scoring: exact match if reference exists
                    scores = {}
                    if reference_text is not None:
                        scores["exact_match"] = float(prediction_text.strip() == reference_text.strip())

                # Log all scores to Weave
                for scorer_name, score_value in scores.items():
                    pred_logger.log_score(scorer=str(scorer_name), score=score_value)

                # Finish this prediction's logging
                pred_logger.finish()

                # Store evaluation scores in the result
                result["evaluation"] = scores
                result["_sample_idx"] = idx

                return result

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}", exc_info=True)
                sample["prediction"] = ""
                sample["evaluation"] = {}
                sample["_sample_idx"] = idx
                return sample

        logger.info(f"Running inference and evaluation for {len(samples)} samples")

        # Process samples concurrently
        processed_samples = []
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_sample = {
                executor.submit(process_single_sample, (idx, sample)): idx
                for idx, sample in enumerate(samples)
            }

            with tqdm(total=len(samples), desc="Inference + Evaluation + Weave logging") as pbar:
                for future in as_completed(future_to_sample):
                    result = future.result()
                    processed_samples.append(result)
                    pbar.update(1)

        # Sort by original index
        processed_samples.sort(key=lambda x: x.get("_sample_idx", 0))

        # Remove temporary index field
        for sample in processed_samples:
            sample.pop("_sample_idx", None)

        logger.info(f"Completed inference and evaluation for {len(processed_samples)} samples")
        return processed_samples

    def _evaluate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample using the configured evaluator."""
        if not self.evaluator:
            return {}

        try:
            # Call evaluator's evaluate method with a single sample
            eval_result = self.evaluator.evaluate([sample])
            if eval_result and "samples" in eval_result and eval_result["samples"]:
                # Extract evaluation scores from the first sample
                return eval_result["samples"][0].get("evaluation", {})
            return {}
        except Exception as e:
            logger.warning(f"Failed to evaluate sample: {e}")
            return {}

    def finalize(self, metrics: Dict[str, Any]) -> None:
        """Finalize Weave logging with summary metrics."""
        if self.evaluation_logger is None:
            return

        try:
            self.evaluation_logger.log_summary(summary=metrics or {})
            self.evaluation_logger.finish()
            logger.info("Finalized Enhanced Weave logging with summary metrics")
        except Exception as e:
            logger.warning(f"Failed to finalize Weave logging: {e}")


@dataclass
class WeaveDatasetExample:
    """Represents a dataset example for Weave Evaluation."""
    input: str
    reference: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {"input": self.input, "reference": self.reference}
        if self.metadata:
            result.update(self.metadata)
        return result


class WeaveModelWrapper(Model):
    """
    Wrapper for existing models to work with Weave Evaluation framework.
    """

    def __init__(
        self,
        model_name: str,
        inference_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.inference_fn = inference_fn
        self.metadata = metadata or {}

    @weave.op()
    def predict(self, input: str, **kwargs) -> Dict[str, Any]:
        """
        Predict method for Weave Evaluation framework.

        Args:
            input: Input text
            **kwargs: Additional arguments from dataset

        Returns:
            Dictionary with prediction and metadata
        """
        sample = {"input": input}
        sample.update(kwargs)

        # Run inference
        results = self.inference_fn([sample])
        if results:
            result = results[0]
            return {
                "prediction": result.get("prediction", ""),
                "metadata": result.get("metadata", {})
            }
        return {"prediction": "", "metadata": {}}


class StandardWeaveEvaluator:
    """
    Standard Weave Evaluation framework implementation.
    Uses Weave's native Evaluation class with proper scorers.
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        scorers: Optional[List[Callable]] = None
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.scorers = scorers or []

    def add_scorer(self, scorer_fn: Callable, name: str = None) -> None:
        """
        Add a scorer function to the evaluator.

        Args:
            scorer_fn: Scorer function with signature (output, reference, **kwargs) -> dict
            name: Optional name for the scorer
        """
        if name:
            scorer_fn.__name__ = name

        # Wrap scorer with @weave.op() if not already decorated
        if not hasattr(scorer_fn, '_weave_op'):
            scorer_fn = weave.op()(scorer_fn)

        self.scorers.append(scorer_fn)

    def create_default_scorers(self) -> List[Callable]:
        """Create default scorers for common evaluation metrics."""
        scorers = []

        @weave.op()
        def exact_match_scorer(reference: str, output: Dict[str, Any]) -> Dict[str, Any]:
            """Exact match scorer."""
            prediction = output.get("prediction", "")
            return {"exact_match": float(prediction.strip() == reference.strip())}

        @weave.op()
        def contains_scorer(reference: str, output: Dict[str, Any]) -> Dict[str, Any]:
            """Check if prediction contains reference."""
            prediction = output.get("prediction", "")
            return {"contains_reference": float(reference.lower() in prediction.lower())}

        @weave.op()
        def length_scorer(output: Dict[str, Any]) -> Dict[str, Any]:
            """Score based on response length."""
            prediction = output.get("prediction", "")
            return {"response_length": len(prediction)}

        scorers.extend([exact_match_scorer, contains_scorer, length_scorer])
        return scorers

    async def evaluate(
        self,
        model: Union[Model, Callable],
        dataset: List[Dict[str, Any]],
        custom_scorers: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation using Weave's standard framework.

        Args:
            model: Weave Model or inference function
            dataset: List of dataset examples
            custom_scorers: Optional list of custom scorer functions

        Returns:
            Evaluation results dictionary
        """
        # Prepare scorers
        all_scorers = self.scorers.copy()
        if custom_scorers:
            all_scorers.extend(custom_scorers)
        if not all_scorers:
            all_scorers = self.create_default_scorers()

        # Wrap model if it's a function
        if not isinstance(model, Model):
            model = WeaveModelWrapper(
                model_name=self.model_name,
                inference_fn=model,
                metadata={"dataset": self.dataset_name}
            )

        # Create evaluation
        evaluation = Evaluation(
            dataset=dataset,
            scorers=all_scorers,
            name=f"{self.dataset_name}_evaluation"
        )

        # Run evaluation
        logger.info(f"Running Weave evaluation with {len(all_scorers)} scorers")
        results = await evaluation.evaluate(model)

        logger.info("Weave evaluation completed")
        return results

    def evaluate_sync(
        self,
        model: Union[Model, Callable],
        dataset: List[Dict[str, Any]],
        custom_scorers: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for evaluate."""
        return asyncio.run(self.evaluate(model, dataset, custom_scorers))


# Helper functions for creating common scorers

@weave.op()
def create_regex_scorer(pattern: str, scorer_name: str = "regex_match") -> Callable:
    """Create a scorer that checks if output matches a regex pattern."""
    import re

    @weave.op(name=scorer_name)
    def regex_scorer(output: Dict[str, Any]) -> Dict[str, Any]:
        prediction = output.get("prediction", "")
        matches = bool(re.search(pattern, prediction))
        return {scorer_name: float(matches)}

    return regex_scorer


@weave.op()
def create_llm_judge_scorer(
    judge_model_fn: Callable,
    criteria: str,
    scorer_name: str = "llm_judge"
) -> Callable:
    """Create a scorer that uses an LLM as a judge."""

    @weave.op(name=scorer_name)
    def llm_judge_scorer(
        input: str,
        reference: str,
        output: Dict[str, Any]
    ) -> Dict[str, Any]:
        prediction = output.get("prediction", "")

        # Construct judge prompt
        judge_prompt = f"""
        Evaluate the following response based on this criteria: {criteria}

        Input: {input}
        Expected: {reference}
        Actual: {prediction}

        Score (0-1):
        """

        # Get judge's score
        judge_result = judge_model_fn([{"input": judge_prompt}])
        if judge_result:
            try:
                score_text = judge_result[0].get("prediction", "0")
                score = float(score_text.strip())
                score = max(0, min(1, score))  # Clamp to [0, 1]
            except (ValueError, AttributeError):
                score = 0.0
        else:
            score = 0.0

        return {scorer_name: score}

    return llm_judge_scorer