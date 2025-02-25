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
    A PipelineRunner class that encapsulates the entire LLM evaluation pipeline:
      - Dataset loading
      - (Optional) Scaling method for model inference
      - Evaluation
      - Final results (as an EvaluationResult object)
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
    ):
        """
        Args:
            dataset_name (str):
                Identifier of the dataset (registered in a dataset registry).
            subset (str | List[str] | None):
                Sub-task or configuration name (e.g., "csat_geo"). Can be None or multiple.
            split (str):
                Which split of the dataset to load ("train", "valid", "test", etc.).
            model_backend_name (str):
                Name of the model backend (e.g., "huggingface", "openai", "multi").
            scaling_method_name (str | None):
                Name of the scaling (decoding) method (e.g., "beam_search", "best_of_n").
                If None, direct inference is used.
            evaluation_method_name (str):
                Name of the evaluation method (e.g., "string_match", "llm_judge").
            dataset_params (dict):
                Additional parameters for the dataset loader (e.g., HF config).
            model_backend_params (dict):
                Additional parameters for the model backend (e.g., endpoint, API key).
            scaling_params (dict):
                Parameters for the scaling method (e.g., beam_size, n).
            evaluator_params (dict):
                Parameters for the evaluator (e.g., CoT prompt injection).
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

        self.dataset: Optional[BaseDataset] = None
        self.model: Optional[BaseModel] = None
        self.scaler: Optional[BaseScalingMethod] = None
        self.evaluator: Optional[BaseEvaluator] = None

        # Pre-load components (lazy loading is also possible)
        self._load_components()

    def _load_components(self) -> None:
        """
        Internal method that:
          1) Loads the dataset
          2) Loads the model
          3) Loads the scaling method (if specified)
          4) Loads the evaluator
          5) Checks dataset info (e.g., scaling_only, evaluation_only)
        """
        # 1) Load dataset
        logger.info(
            f"[Pipeline] Loading dataset: {self.dataset_name}, "
            f"subset={self.subset}, split={self.split}"
        )
        self.dataset = load_datasets(
            name=self.dataset_name,
            subset=self.subset,
            split=self.split,
            **self.dataset_params
        )
        ds_info = self.dataset.info()

        # Example checks for restricted scaling/evaluation
        scaling_only = ds_info.get("scaling_only", None)
        evaluation_only = ds_info.get("evaluation_only", None)

        # 2) Load model
        logger.info(
            f"[Pipeline] Loading model backend: {self.model_backend_name} "
            f"with params={self.model_backend_params}"
        )
        self.model = load_model(self.model_backend_name, **self.model_backend_params)

        # 3) Load scaling method (if specified)
        if self.scaling_method_name:
            if scaling_only is not None:
                if self.scaling_method_name not in scaling_only:
                    raise ValueError(
                        f"Dataset '{self.dataset_name}' only allows "
                        f"scaling methods {scaling_only}, but got '{self.scaling_method_name}'."
                    )
            logger.info(
                f"[Pipeline] Loading scaling method: {self.scaling_method_name} "
                f"with params={self.scaling_params}"
            )
            self.scaler = load_scaling_method(
                self.scaling_method_name,
                model=self.model,
                **self.scaling_params
            )
        else:
            # If the dataset enforces a scaling method but none was provided
            if scaling_only is not None:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' requires a scaling method from {scaling_only}, "
                    f"but scaling_method_name=None."
                )

        # 4) Load evaluator
        if evaluation_only is not None:
            if self.evaluation_method_name not in evaluation_only:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' only allows evaluation methods {evaluation_only}, "
                    f"but got '{self.evaluation_method_name}'."
                )
        logger.info(
            f"[Pipeline] Loading evaluator: {self.evaluation_method_name} "
            f"with params={self.evaluator_params}"
        )
        if self.evaluation_method_name == "llm_judge":
            self.evaluator = get_evaluator(self.evaluation_method_name, model=self.model, **self.evaluator_params)
        else:
            self.evaluator = get_evaluator(self.evaluation_method_name, **self.evaluator_params)

    def run(self) -> EvaluationResult:
        """
        Executes the pipeline:
          1) Dataset load
          2) Skips or applies scaling method (model inference)
          3) Evaluation
          4) Returns final results in an EvaluationResult object

        Returns:
            EvaluationResult: Encapsulates metrics, samples, and pipeline info.
        """
        if not self.dataset or not self.model or not self.evaluator:
            raise RuntimeError("Pipeline components are not fully loaded.")

        start_time = time.time()

        # 1) Load data
        data = self.dataset.load()  # e.g., [{"input":"...", "reference":"...", ...}, ...]
        logger.info(f"[Pipeline] Dataset loaded. Number of samples: {len(data)}")

        # 2) Skip or apply inference/scaling
        already_has_prediction = all("prediction" in item for item in data)
        if already_has_prediction:
            logger.info(
                "[Pipeline] Found existing 'prediction' in dataset items. "
                "Skipping model generation/scaling."
            )
            predictions = data
        else:
            if self.scaler:
                logger.info(f"[Pipeline] Applying scaling method: {self.scaling_method_name}")
                predictions = self.scaler.apply(data)
            else:
                logger.info("[Pipeline] No scaling method, using direct model inference.")
                predictions = self.model.generate_batch(data)

            logger.info(f"[Pipeline] Inference done for {len(predictions)} items.")

        # 3) Evaluation
        logger.info(f"[Pipeline] Evaluating with {self.evaluation_method_name}")
        eval_dict = self.evaluator.evaluate(predictions, model=self.model)
        # eval_dict 예시: {"metrics": {...}, "samples": [...], ...}

        # 4) Wrap results into an EvaluationResult
        end_time = time.time()
        elapsed = end_time - start_time

        pipeline_info = {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "model_backend_name": self.model_backend_name,
            "scaling_method_name": self.scaling_method_name,
            "evaluation_method_name": self.evaluation_method_name,
            "elapsed_time_sec": elapsed,
        }

        # Merge any existing info from eval_dict
        metrics = eval_dict.get("metrics", {})
        samples = eval_dict.get("samples", [])
        existing_info = eval_dict.get("info", {})
        merged_info = {**existing_info, **pipeline_info}

        # Construct the EvaluationResult object
        result = EvaluationResult(
            metrics=metrics,
            samples=samples,
            info=merged_info
        )

        return result
