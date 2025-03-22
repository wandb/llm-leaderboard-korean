#!/usr/bin/env python
"""
PipelineRunner 

This module encapsulates the entire pipeline:
  1. Load a dataset (with optional subset and split)
  2. Load a model backend
  3. Optionally apply a scaling method for model inference
  4. Evaluate the generated predictions using an evaluator
  5. Optionally apply a language penalizer to each prediction, using a parameterized target language
  6. Aggregate metrics and sample outputs into an EvaluationResult

This runner can be used via CLI or integrated into an API.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from llm_eval.datasets import load_datasets, BaseDataset
from llm_eval.models import load_model, BaseModel
from llm_eval.scaling_methods import load_scaling_method, BaseScalingMethod
from llm_eval.evaluation import get_evaluator, BaseEvaluator
from llm_eval.utils.logging import get_logger
from llm_eval.utils.util import EvaluationResult

logger = get_logger(name="runner", level=logging.INFO)


class PipelineRunner:
    """
    PipelineRunner encapsulates the entire LLM evaluation pipeline:
      - Dataset loading (using dataset_name, subset, split)
      - Model loading (using model_backend_name and its parameters)
      - Optional scaling method application for advanced decoding
      - Evaluation using a specified evaluator
      - Optional language penalizer applied to predictions (target language parameterized)
      - Aggregation of metrics and pipeline info into an EvaluationResult

    The pipeline steps are:
      1. Load dataset via load_datasets().
      2. Load model via load_model().
      3. If specified, load scaling method via load_scaling_method().
      4. Run inference: if predictions already exist, skip; otherwise, generate predictions.
      5. Evaluate predictions using the evaluator.
      6. Optionally apply the language penalizer to each prediction based on target_lang.
      7. Aggregate and return the results.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: Union[str, List[str], None] = None,
        split: str = "test",
        model_backend_name: str = "huggingface",
        scaling_method_name: Optional[str] = None,
        evaluation_method_name: str = "string_match",
        dataset_params: Optional[Dict[str, Any]] = None,
        model_backend_params: Optional[Dict[str, Any]] = None,
        scaling_params: Optional[Dict[str, Any]] = None,
        evaluator_params: Optional[Dict[str, Any]] = None,
        language_penalize: bool = True,
        target_lang: str = "ko",  # parameterized target language for penalizer
    ):
        """
        Initialize the PipelineRunner with identifiers and parameters for each component.

        Args:
            dataset_name (str): Dataset identifier (registered in the dataset registry).
            subset (str | list[str] | None): Optional sub-task or configuration (e.g., "csat_geo").
            split (str): Dataset split to load ("train", "valid", "test", etc.).
            model_backend_name (str): Model backend identifier (e.g., "huggingface", "openai", "multi").
            scaling_method_name (str | None): Scaling (decoding) method identifier (e.g., "beam_search", "best_of_n").
            evaluation_method_name (str): Evaluator identifier (e.g., "string_match", "llm_judge").
            dataset_params (dict): Additional parameters for the dataset loader.
            model_backend_params (dict): Additional parameters for the model backend.
            scaling_params (dict): Parameters for the scaling method.
            evaluator_params (dict): Parameters for the evaluator.
            language_penalize (bool): If True, apply language penalizer to predictions.
            target_lang (str): Target language code for language penalization (e.g., "ko" for Korean).
        """
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.model_backend_name = model_backend_name
        self.scaling_method_name = scaling_method_name
        self.evaluation_method_name = evaluation_method_name

        self.dataset_params = dataset_params or {}
        self.model_backend_params = model_backend_params or {}
        self.scaling_params = scaling_params or {}
        self.evaluator_params = evaluator_params or {}

        self.language_penalize = language_penalize
        self.target_lang = target_lang  # 저장된 target language

        self.dataset: Optional[BaseDataset] = None
        self.model: Optional[BaseModel] = None
        self.scaler: Optional[BaseScalingMethod] = None
        self.evaluator: Optional[BaseEvaluator] = None

        # Pre-load components; lazy loading is also possible.
        self._load_components()

    def _load_components(self) -> None:
        """
        Internal method to load all pipeline components:
          1) Loads dataset using load_datasets() with dataset_name, subset, split, and dataset_params.
          2) Loads model backend using load_model().
          3) Loads scaling method (if specified) using load_scaling_method(), verifying against dataset restrictions.
          4) Loads evaluator using get_evaluator(), verifying against dataset restrictions.
        """
        logger.info(
            f"[Pipeline] Loading dataset: {self.dataset_name}, subset={self.subset}, split={self.split}"
        )
        self.dataset = load_datasets(
            name=self.dataset_name,
            subset=self.subset,
            split=self.split,
            **self.dataset_params
        )
        ds_info = self.dataset.info()
        scaling_only = ds_info.get("scaling_only", None)
        evaluation_only = ds_info.get("evaluation_only", None)

        logger.info(
            f"[Pipeline] Loading model backend: {self.model_backend_name} with params={self.model_backend_params}"
        )
        self.model = load_model(self.model_backend_name, **self.model_backend_params)

        if self.scaling_method_name:
            if scaling_only is not None:
                if self.scaling_method_name not in scaling_only:
                    raise ValueError(
                        f"Dataset '{self.dataset_name}' only allows scaling methods {scaling_only}, "
                        f"but got '{self.scaling_method_name}'."
                    )
            logger.info(
                f"[Pipeline] Loading scaling method: {self.scaling_method_name} with params={self.scaling_params}"
            )
            self.scaler = load_scaling_method(
                self.scaling_method_name,
                model=self.model,
                **self.scaling_params
            )
        else:
            if scaling_only is not None:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' requires a scaling method from {scaling_only}, "
                    f"but scaling_method_name is None."
                )

        if evaluation_only is not None:
            if isinstance(evaluation_only, bool):
                if evaluation_only:
                    raise ValueError(
                        f"Dataset '{self.dataset_name}' requires a specific evaluation method, "
                        f"but no specific methods were provided."
                    )
            elif self.evaluation_method_name not in evaluation_only:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' only allows evaluation methods {evaluation_only}, "
                    f"but got '{self.evaluation_method_name}'."
                )
        logger.info(
            f"[Pipeline] Loading evaluator: {self.evaluation_method_name} with params={self.evaluator_params}"
        )
        if self.evaluation_method_name == "llm_judge":
            self.evaluator = get_evaluator(self.evaluation_method_name, model=self.model, **self.evaluator_params)
        else:
            self.evaluator = get_evaluator(self.evaluation_method_name, **self.evaluator_params)

    def run(self) -> EvaluationResult:
        """
        Executes the entire pipeline:
          1) Dataset loading: Reads data from the dataset loader.
          2) Inference: Either skips if predictions exist or performs model generation/scaling.
          3) Evaluation: Evaluates predictions against references using the evaluator.
          4) Optionally applies the language penalizer to each prediction based on target_lang.
          5) Aggregates metrics, sample outputs, and pipeline info into an EvaluationResult.

        Returns:
            EvaluationResult: An object encapsulating final metrics, samples, and additional pipeline info.
        """
        if not self.dataset or not self.model or not self.evaluator:
            raise RuntimeError("Pipeline components are not fully loaded.")

        start_time = time.time()

        # Step 1: Load data
        data = self.dataset.load()  # e.g., [{"input": "...", "reference": "...", ...}, ...]
        logger.info(f"[Pipeline] Dataset loaded. Number of samples: {len(data)}")

        # Step 2: Inference
        already_has_prediction = all("prediction" in item for item in data)
        if already_has_prediction:
            logger.info(
                "[Pipeline] Existing predictions found in dataset items; skipping model inference."
            )
            predictions = data
        else:
            if self.scaler:
                logger.info(f"[Pipeline] Applying scaling method: {self.scaling_method_name}")
                predictions = self.scaler.apply(data)
            else:
                logger.info("[Pipeline] No scaling method provided; using direct model inference.")
                predictions = self.model.generate_batch(data)
            logger.info(f"[Pipeline] Inference completed for {len(predictions)} items.")

        # Step 3: Evaluation
        logger.info(f"[Pipeline] Evaluating with {self.evaluation_method_name}")
        eval_dict = self.evaluator.evaluate(predictions, model=self.model)
        # Expected eval_dict format: {"metrics": {...}, "samples": [...], ...}

        # Step 4: Optionally apply language penalizer if enabled
        if self.language_penalize:
            from llm_eval.utils.metrics import language_penalizer
            # Use the parameterized target language instead of a hardcoded value
            target_lang = self.target_lang
            language_scores = []
            for sample in eval_dict.get("samples", []):
                # original_prediction이 있으면 그것을 사용하고, 없으면 prediction 사용
                pred_text = sample.get("original_prediction", sample.get("prediction", ""))
                lp_score = language_penalizer(pred_text, target_lang=target_lang)
                sample["language_penalizer"] = lp_score
                language_scores.append(lp_score)
            if language_scores:
                avg_lp = sum(language_scores) / len(language_scores)
            else:
                avg_lp = 0.0
            eval_dict.setdefault("metrics", {})["language_penalizer_average"] = avg_lp

        # Step 5: Aggregate results into an EvaluationResult
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[Pipeline] Pipeline completed. Elapsed time: {elapsed:.2f} seconds.")

        pipeline_info = {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "model_backend_name": self.model_backend_name,
            "scaling_method_name": self.scaling_method_name,
            "evaluation_method_name": self.evaluation_method_name,
            "elapsed_time_sec": elapsed,
        }

        metrics = eval_dict.get("metrics", {})
        samples = eval_dict.get("samples", [])
        
        # 내부 상태 필드 제거
        for sample in samples:
            if "_judged_by_evaluator" in sample:
                del sample["_judged_by_evaluator"]
                
        existing_info = eval_dict.get("info", {})
        merged_info = {**existing_info, **pipeline_info}

        result = EvaluationResult(
            metrics=metrics,
            samples=samples,
            info=merged_info
        )

        return result
