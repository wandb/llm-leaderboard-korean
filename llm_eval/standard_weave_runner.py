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
    model_params: Dict[str, Any]

    def __init__(self, model_backend: str, model_params: Dict[str, Any] = None):
        # Pass fields to parent __init__
        super().__init__(
            model_backend=model_backend,
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
                        "sample_id": sample.get("id", None),
                    }
                }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"prediction": "", "metadata": {"error": str(e)}}

        return {"prediction": "", "metadata": {}}


def create_dataset_scorers(dataset_name: str, subset: List[str], evaluation_method: str, eval_params: Dict[str, Any] = None) -> List[Callable]:
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
                    subsets=subset,
                    samples=[sample]
                )
                return eval_result

                # # Check if evaluator added evaluation info to sample
                # if "evaluation" in sample and isinstance(sample["evaluation"], dict):
                #     # Return the evaluation details from the sample
                #     return sample["evaluation"]
                # elif eval_result:
                #     # Fallback to metrics if no sample evaluation
                #     return eval_result
                # else:
                #     # Final fallback to simple scoring
                #     return {evaluation_method: float(prediction.strip() == reference.strip())}

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
    dataset_params = dataset_config.get("params", {})
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
        import types
        NewModelAdapter = types.new_class(model_params.get("model_name", model_name), (WeaveModelAdapter,))
        model_adapter = NewModelAdapter(
            model_backend=model_name,
            model_params=model_params
        )

        # Create scorers
        scorers = create_dataset_scorers(dataset_key, subset, evaluation_method, evaluator_params)
        logger.info(f"Created {len(scorers)} scorers for evaluation")

        # Create named Weave dataset (exclude subset from name; fallback to raw list if not supported)
        dataset_name_str = str(dataset_key)
        try:
            # Prefer explicit constructor signature if available
            from weave import Dataset as WeaveDataset  # type: ignore
            try:
                named_dataset = WeaveDataset(name=dataset_name_str, rows=weave_dataset)  # type: ignore
            except TypeError:
                named_dataset = WeaveDataset(weave_dataset, name=dataset_name_str)  # type: ignore
        except Exception:
            named_dataset = weave_dataset

        # Create Weave evaluation
        evaluation = Evaluation(
            dataset=named_dataset,
            scorers=scorers,
            name=f"{dataset_key}_evaluation"
        )

        # Run evaluation asynchronously
        logger.info("Running Weave evaluation...")
        results = asyncio.run(evaluation.evaluate(model_adapter))

        logger.info("Evaluation completed, processing results...")

        # Process results
        metrics = results
        samples_with_scores = []

        # # Calculate aggregate metrics
        # if samples_with_scores:
        #     score_totals = {}
        #     score_counts = {}

        #     for sample in samples_with_scores:
        #         if "evaluation" in sample:
        #             for score_name, score_value in sample["evaluation"].items():
        #                 if isinstance(score_value, (int, float)):
        #                     if score_name not in score_totals:
        #                         score_totals[score_name] = 0
        #                         score_counts[score_name] = 0
        #                     score_totals[score_name] += score_value
        #                     score_counts[score_name] += 1

        #     # Calculate averages
        #     for score_name in score_totals:
        #         metrics[f"{score_name}_avg"] = score_totals[score_name] / score_counts[score_name]
        #         metrics[f"{score_name}_total"] = score_totals[score_name]
        #         metrics[f"{score_name}_count"] = score_counts[score_name]

        # Add overall metrics
        metrics["num_samples"] = len(samples_with_scores) if samples_with_scores else len(weave_dataset)
        scores = {}

        if "string_match_scorer" in metrics:
            metric = metrics["string_match_scorer"]
            scores['score'] = metric["AVG"]["mean"]
            # for haerae_bench_v1/kobalt subsets
            for key, value in metric.items():
                if key != "AVG":
                    scores[key] = value["mean"]
        elif "ifeval_strict_scorer" in metrics:
            metric = metrics["ifeval_strict_scorer"]
            scores['score'] = metric["AVG"]["mean"]
        elif "sequence_match_scorer" in metrics:
            metric = metrics["sequence_match_scorer"]
            scores['score'] = metric["AVG"]["mean"]
        elif "mt_bench_judge_scorer" in metrics:
            metric = metrics["mt_bench_judge_scorer"]
            scores['score'] = metric["AVG"]["mean"]
            scores["roleplay"] = metric["roleplay/average_judge_score"]["mean"]
            scores["humanities"] = metric["humanities/average_judge_score"]["mean"]
            scores["writing"] = metric["writing/average_judge_score"]["mean"]
            scores["reasoning"] = metric["reasoning/average_judge_score"]["mean"]
            scores["coding"] = metric["coding/average_judge_score"]["mean"]
        elif "char_f1_scorer" in metrics:
            metric = metrics["char_f1_scorer"]
            scores['score'] = metric["AVG"]["mean"]
        elif "math_match_scorer" in metrics:
            metric = metrics["math_match_scorer"]
            scores['score'] = metric["AVG"]["mean"]
        elif "grid_match_scorer" in metrics:
            metric = metrics["grid_match_scorer"]
            scores['score'] = metric["AVG"]["mean"]
        elif "hallulens_scorer" in metrics:
            metric = metrics["hallulens_scorer"]
            scores['score'] = metric["is_hallucinated"]["true_fraction"]
        elif "bfcl_scorer" in metrics:
            metric = metrics["bfcl_scorer"]
            scores['score'] = metric["is_correct"]["true_fraction"]
        elif "swebench_scorer" in metrics:
            metric = metrics["swebench_scorer"]
            scores['score'] = metric["resolved_rate"]["mean"]
        elif "comet_score_scorer" in metrics:
            metric = metrics["comet_score_scorer"]
            scores['score'] = metric["AVG"]["mean"]

        # Log to W&B if configured
        if wandb_params:
            table_name = f"{dataset_key}_leaderboard"
            data = {
                "model_name": model_params.get("model_name", model_name),
                "score": scores['score'],
                **{k: v for k, v in scores.items()}
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