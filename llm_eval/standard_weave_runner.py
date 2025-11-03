"""
Standard Weave Evaluation Framework Runner

This module provides integration with Weave's standard evaluation framework,
ensuring that evaluation scores are properly logged for each prediction/trace.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
import weave
from weave import Model, Evaluation

from llm_eval.datasets import load_datasets
from llm_eval.models import load_model
from llm_eval.evaluation import get_evaluator
from llm_eval.utils.logging import get_logger
from llm_eval.utils.util import EvaluationResult
from llm_eval.utils.metrics import language_penalizer
from llm_eval.wandb_singleton import WandbConfigSingleton
import wandb
import pandas as pd

logger = get_logger(name="standard_weave_runner", level=logging.INFO)


class WeaveModelAdapter(Model):
    """Adapter to use existing models with Weave Evaluation framework."""

    # Define fields as class attributes for Weave Model
    model_backend: str
    model_name: str
    model_params: Dict[str, Any]

    def __init__(self, model_backend: str, model_name: str, model_params: Dict[str, Any] = None):
        # Pass fields to parent __init__
        super().__init__(
            model_backend=model_backend,
            model_name=model_name,
            model_params=model_params or {}
        )
        # Use private attribute for cached model (not a Weave field)
        self._cached_model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._cached_model is None:
            self._cached_model = load_model(self.model_backend, **self.model_params)
        return self._cached_model

    @weave.op()
    def predict(self, input: str, **kwargs) -> Dict[str, Any]:
        """
        Predict method compatible with Weave Evaluation.

        Args:
            input: Input text
            **kwargs: Additional dataset fields

        Returns:
            Dictionary with prediction and metadata
        """
        model = self._load_model()

        # Prepare sample for model
        sample = {"input": input}
        sample.update(kwargs)

        # Run inference
        try:
            results = model.generate_batch([sample])
            if results:
                result = results[0]
                return {
                    "prediction": result.get("prediction", ""),
                    "metadata": {
                        "model_name": self.model_name,
                        "sample_id": sample.get("id", None),
                    }
                }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"prediction": "", "metadata": {"error": str(e)}}

        return {"prediction": "", "metadata": {}}


def create_dataset_scorers(dataset_name: str, evaluation_method: str, eval_params: Dict[str, Any] = None) -> List[Callable]:
    """
    Create scorers based on existing evaluation methods.

    Args:
        dataset_name: Name of the dataset
        evaluation_method: Evaluation method to use
        eval_params: Additional evaluation parameters

    Returns:
        List of scorer functions
    """
    scorers = []
    eval_params = eval_params or {}

    # Get the evaluator instance for the specified method
    try:
        from llm_eval.evaluation import get_evaluator
        evaluator = get_evaluator(evaluation_method, **eval_params)

        # Create a scorer that wraps the evaluator
        @weave.op(name=f"{evaluation_method}_scorer")
        def evaluator_scorer(reference: str, output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Scorer that uses the existing evaluator."""
            prediction = output.get("prediction", "")

            # Create a single sample for evaluation
            sample = {
                "input": kwargs.get("input", ""),
                "prediction": prediction,
                "reference": reference,
                "id": kwargs.get("id", "sample"),
            }

            # Add any additional fields from kwargs
            for key, value in kwargs.items():
                if key not in sample:
                    sample[key] = value

            # Run evaluation on single sample
            try:
                # evaluate_predictions expects 'samples' parameter
                # It modifies samples in-place and returns metrics
                eval_result = evaluator.evaluate_predictions(
                    subsets=None,
                    samples=[sample]
                )

                # Check if evaluator added evaluation info to sample
                if "evaluation" in sample and isinstance(sample["evaluation"], dict):
                    # Return the evaluation details from the sample
                    return sample["evaluation"]
                elif eval_result:
                    # Fallback to metrics if no sample evaluation
                    return eval_result
                else:
                    # Final fallback to simple scoring
                    return {evaluation_method: float(prediction.strip() == reference.strip())}

            except Exception as e:
                logger.warning(f"Evaluator {evaluation_method} failed: {e}")
                return {evaluation_method: 0.0}

        scorers.append(evaluator_scorer)

    except ValueError as e:
        logger.warning(f"Evaluator '{evaluation_method}' not found, using fallback scorers: {e}")
        # Provide basic fallback scorer
        @weave.op(name="fallback_exact_match")
        def fallback_scorer(reference: str, output: Dict[str, Any]) -> Dict[str, Any]:
            """Fallback exact match scorer."""
            prediction = output.get("prediction", "")
            return {"exact_match": float(prediction.strip() == reference.strip())}
        scorers.append(fallback_scorer)


    # Add response length scorer
    @weave.op(name="response_length")
    def response_length_scorer(output: Dict[str, Any]) -> Dict[str, Any]:
        """Response length scorer."""
        prediction = output.get("prediction", "")
        return {"response_length": len(prediction)}
    scorers.append(response_length_scorer)

    return scorers


def run_with_standard_weave(
    dataset_key: str,
    dataset_config: Dict[str, Any],
    model_name: str,
    model_params: Dict[str, Any],
    wandb_params: Dict[str, Any],
    evaluation_method: str = "string_match",
    evaluator_params: Optional[Dict[str, Any]] = None,
    language_penalize: bool = True,
    target_lang: str = "ko",
) -> EvaluationResult:
    """
    Run evaluation using Standard Weave Evaluation Framework.

    Args:
        dataset_key: Dataset identifier
        dataset_config: Dataset configuration
        model_name: Model backend name
        model_params: Model parameters
        wandb_params: W&B parameters
        evaluation_method: Evaluation method
        evaluator_params: Additional evaluator parameters
        language_penalize: Whether to apply language penalization
        target_lang: Target language for penalization

    Returns:
        EvaluationResult object
    """
    logger.info(f"Starting Standard Weave Evaluation for dataset '{dataset_key}'")

    # Extract dataset parameters
    subset = dataset_config.get("subset")
    split = dataset_config.get("split", "test")
    limit = dataset_config.get("limit")
    dataset_params = dataset_config.get("params", {})

    # Apply testmode limit if dev is set
    if dataset_params.get("dev", False):
        # In testmode, limit samples more aggressively
        if dataset_key == "aime2025" and isinstance(subset, list) and len(subset) > 1:
            # For aime2025 with multiple subsets, divide limit by number of subsets
            limit = min(limit or 10, 10) // len(subset)  # Divide among subsets
            limit = max(limit, 2)  # At least 2 per subset
        elif dataset_key == "aime2025":
            limit = min(limit or 10, 10)  # Max 10 for aime2025 single subset
        else:
            limit = min(limit or 5, 5)  # Default to 5 for other datasets
        logger.info(f"Testmode enabled, limiting to {limit} samples per subset")

    if "limit" not in dataset_params:
        dataset_params["limit"] = limit

    try:
        # Load dataset
        logger.info(f"Loading dataset: {dataset_key}, subset={subset}, split={split}")
        dataset_loader = load_datasets(
            name=dataset_key,
            subset=subset,
            split=split,
            **dataset_params
        )
        raw_samples = dataset_loader.load()

        if not raw_samples:
            logger.warning(f"No samples loaded for dataset '{dataset_key}'")
            return EvaluationResult(
                metrics={"error": "No samples loaded"},
                samples=[],
                info={"dataset": dataset_key, "status": "no_data"}
            )

        logger.info(f"Loaded {len(raw_samples)} samples")

        # Prepare dataset for Weave Evaluation
        weave_dataset = []
        for sample in raw_samples:
            # Start with all original fields to preserve dataset-specific information
            weave_sample = sample.copy()

            # Ensure required fields exist
            if "input" not in weave_sample:
                weave_sample["input"] = ""
            if "reference" not in weave_sample:
                weave_sample["reference"] = ""

            weave_dataset.append(weave_sample)

        # Create model adapter
        model_adapter = WeaveModelAdapter(
            model_backend=model_name,
            model_name=model_params.get("model_name", model_name),
            model_params=model_params
        )

        # Create scorers
        scorers = create_dataset_scorers(dataset_key, evaluation_method, evaluator_params)
        logger.info(f"Created {len(scorers)} scorers for evaluation")

        # Create Weave evaluation
        evaluation = Evaluation(
            dataset=weave_dataset,
            scorers=scorers,
            name=f"{dataset_key}_evaluation"
        )

        # Run evaluation asynchronously
        logger.info("Running Weave evaluation...")
        results = asyncio.run(evaluation.evaluate(model_adapter))

        logger.info("Evaluation completed, processing results...")

        # Process results
        metrics = {}
        samples_with_scores = []

        # Get detailed results using get_eval_results if available
        try:
            from weave import get_eval_results
            detailed_results = get_eval_results(results)

            for row in detailed_results:
                sample_scores = {}
                for scorer_name in [s.__name__ for s in scorers]:
                    if hasattr(row, scorer_name):
                        sample_scores[scorer_name] = getattr(row, scorer_name)

                samples_with_scores.append({
                    "input": row.input if hasattr(row, 'input') else "",
                    "prediction": row.output.get("prediction", "") if hasattr(row, 'output') else "",
                    "reference": row.reference if hasattr(row, 'reference') else "",
                    "evaluation": sample_scores
                })
        except Exception as e:
            logger.warning(f"Could not get detailed results: {e}")
            # Fallback to basic results
            if hasattr(results, 'summary'):
                metrics = results.summary

        # Calculate aggregate metrics
        if samples_with_scores:
            score_totals = {}
            score_counts = {}

            for sample in samples_with_scores:
                if "evaluation" in sample:
                    for score_name, score_value in sample["evaluation"].items():
                        if isinstance(score_value, (int, float)):
                            if score_name not in score_totals:
                                score_totals[score_name] = 0
                                score_counts[score_name] = 0
                            score_totals[score_name] += score_value
                            score_counts[score_name] += 1

            # Calculate averages
            for score_name in score_totals:
                metrics[f"{score_name}_avg"] = score_totals[score_name] / score_counts[score_name]
                metrics[f"{score_name}_total"] = score_totals[score_name]
                metrics[f"{score_name}_count"] = score_counts[score_name]

        # Add overall metrics
        metrics["num_samples"] = len(samples_with_scores) if samples_with_scores else len(weave_dataset)

        # Calculate AVG score for leaderboard
        if "exact_match_avg" in metrics:
            metrics["AVG"] = metrics["exact_match_avg"]
        elif "choice_match_avg" in metrics:
            metrics["AVG"] = metrics["choice_match_avg"]
        else:
            # Use first available average score
            for key in metrics:
                if key.endswith("_avg") and not key.startswith("response_length"):
                    metrics["AVG"] = metrics[key]
                    break

        # Log to W&B if configured
        if wandb_params:
            table_name = f"{dataset_key}_weave_leaderboard"
            data = {
                "model_name": model_params.get("model_name", model_name),
                "AVG": metrics.get("AVG", 0.0),
                **{k: v for k, v in metrics.items() if k.endswith("_avg")}
            }
            df = pd.DataFrame([data])

            # Log to W&B table
            WandbConfigSingleton.collect_leaderboard_table(dataset_key, df)
            run = WandbConfigSingleton.get_instance().run
            if run:
                run.log({table_name: wandb.Table(dataframe=df)})

        logger.info(f"Standard Weave Evaluation completed for '{dataset_key}'. Metrics: {metrics}")

        return EvaluationResult(
            metrics=metrics,
            samples=samples_with_scores[:100],  # Limit samples for memory
            info={
                "dataset_name": dataset_key,
                "subset": subset,
                "split": split,
                "model_backend_name": model_name,
                "evaluation_method": evaluation_method,
                "weave_evaluation": True,
                "num_scorers": len(scorers),
                "status": "completed"
            }
        )

    except Exception as e:
        logger.error(f"Standard Weave Evaluation failed for '{dataset_key}': {e}", exc_info=True)
        return EvaluationResult(
            metrics={"error": str(e)},
            samples=[],
            info={
                "dataset_name": dataset_key,
                "model_backend_name": model_name,
                "error": str(e),
                "status": "failed"
            }
        )