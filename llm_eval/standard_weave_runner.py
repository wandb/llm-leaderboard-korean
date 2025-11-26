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
                    # _subset_name과 subset 모두 문자열로 변환
                    if key in ["_subset_name", "subset"] and value is not None:
                        sample[key] = str(value) if not isinstance(value, str) else value
                    else:
                        sample[key] = value

            # Run evaluation on single sample
            try:
                # 샘플의 실제 subset 정보를 사용 (전체 subset 리스트 대신)
                sample_subset = sample.get("_subset_name")

                # evaluate_predictions expects 'samples' parameter
                # It modifies samples in-place and returns metrics
                eval_result = evaluator.evaluate_predictions(
                    subsets=[sample_subset] if sample_subset else None,
                    samples=[sample]
                )

                # subset 정보는 이제 input에 있으므로 output에서는 제거
                if "subset" in eval_result:
                    del eval_result["subset"]

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

            # Move _subset_name to a clear 'subset' field for Weave display
            if "_subset_name" in weave_sample:
                subset_value = weave_sample["_subset_name"]
                # Ensure subset is a string
                weave_sample["subset"] = str(subset_value) if not isinstance(subset_value, str) else subset_value
                # Keep _subset_name for backward compatibility but ensure it's also a string
                weave_sample["_subset_name"] = weave_sample["subset"]

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

        # Debug: Weave 결과 구조 확인
        if "string_match_scorer" in results:
            logger.debug(f"string_match_scorer keys: {list(results['string_match_scorer'].keys())}")
            if "subset" in results["string_match_scorer"]:
                subset_info = results["string_match_scorer"]["subset"]
                logger.info(f"Found subset information in results: {type(subset_info)} - {subset_info}")
            if "is_correct" in results["string_match_scorer"]:
                logger.info(f"is_correct field found: {results['string_match_scorer']['is_correct']}")

        # Process results
        metrics = results
        samples_with_scores = []

        # Add overall metrics
        metrics["num_samples"] = len(samples_with_scores) if samples_with_scores else len(weave_dataset)
        scores = {}

        if "string_match_scorer" in metrics:
            metric = metrics["string_match_scorer"]

            # 전체 accuracy 계산
            if "is_correct" in metric and isinstance(metric["is_correct"], dict):
                # boolean 값이 집계된 경우 (true_fraction이 accuracy)
                scores['accuracy'] = metric["is_correct"].get("true_fraction", 0.0)
            elif "accuracy" in metric and isinstance(metric["accuracy"], dict):
                scores['accuracy'] = metric["accuracy"]["mean"]
            # 기존 형식 호환성 유지
            elif "AVG" in metric:
                scores['accuracy'] = metric["AVG"]["mean"]
            elif "score" in metric:
                scores['accuracy'] = metric["score"]["mean"]

            # subset 정보 처리
            # 데이터셋에서 실제 subset 정보 추출
            unique_subsets = set()
            for sample in weave_dataset:
                if "_subset_name" in sample and sample["_subset_name"]:
                    unique_subsets.add(sample["_subset_name"])

            if unique_subsets:
                scores['subset'] = sorted(list(unique_subsets))
            else:
                # Weave metric에서 subset 정보 확인
                if "subset" in metric:
                    subset_value = metric["subset"]
                    # subset이 딕셔너리 형태로 집계된 경우
                    if isinstance(subset_value, dict):
                        scores['subset'] = list(subset_value.keys())
                    # subset이 문자열인 경우
                    elif isinstance(subset_value, str):
                        scores['subset'] = [subset_value]
                    else:
                        scores['subset'] = 'N/A'
                else:
                    scores['subset'] = 'N/A'

            logger.info(f"Evaluation includes subsets: {scores.get('subset', 'N/A')}")
        elif "ifeval_strict_scorer" in metrics:
            metric = metrics["ifeval_strict_scorer"]
            scores['score'] = metric["AVG"]["mean"]
        elif "sequence_match_scorer" in metrics:
            metric = metrics["sequence_match_scorer"]
            # sequence_match_scorer returns 'sequence_match_score' not 'AVG'
            if "sequence_match_score" in metric:
                scores['score'] = metric["sequence_match_score"]["mean"]
            elif "AVG" in metric:
                scores['score'] = metric["AVG"]["mean"]
            else:
                logger.warning(f"Neither 'sequence_match_score' nor 'AVG' found in sequence_match_scorer metrics: {metric.keys()}")
                scores['score'] = 0.0
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
            scores['score'] = metric["char_f1"]["mean"]
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

            # subset 정보 처리 - 리스트인 경우 문자열로 변환
            subset_value = scores.get('subset', 'N/A')
            if isinstance(subset_value, list):
                subset_value = ', '.join(subset_value) if subset_value else 'N/A'
            elif not isinstance(subset_value, str):
                subset_value = str(subset_value)
            print(model_params)
            print(model_name)
            data = {
                "model_name": model_params.get("model_name", model_name.split('/')[ -1]),
                "score": scores.get('accuracy', scores.get('score', 0.0)),  # accuracy 우선, 없으면 score
                "subset": subset_value,  # subset 정보 추가 (문자열로 변환됨)
                **{k: v for k, v in scores.items() if k not in ['accuracy', 'score', 'subset']}
            }
            df = pd.DataFrame([data])
            df.to_csv(f"{dataset_key}_leaderboard.csv", index=False)

            # Log to W&B table
            WandbConfigSingleton.collect_leaderboard_table(dataset_key, df)
            run = WandbConfigSingleton.get_instance().run
            if run:
                run.log({table_name: wandb.Table(dataframe=df)})

        logger.info(f"Standard Weave Evaluation completed for '{dataset_key}'. Metrics: {metrics}")

        # 최종 메트릭 구성에 scores 정보 추가
        if scores:
            # string_match_scorer의 결과를 더 명확하게 구성
            if "string_match_scorer" in metrics:
                final_scorer_metrics = {
                    "accuracy": scores.get('accuracy', 0.0),
                }
                # subset 정보가 있으면 추가
                if 'subset' in scores:
                    final_scorer_metrics["subset"] = scores['subset']

                # 기존 메트릭 구조 업데이트
                metrics["string_match_scorer_summary"] = final_scorer_metrics

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
                "status": "completed",
                "scores": scores  # scores 정보도 info에 포함
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
