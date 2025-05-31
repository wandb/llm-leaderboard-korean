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
from typing import Any, Dict, List, Optional, Union, Callable, Tuple

from llm_eval.datasets import load_datasets, BaseDataset
from llm_eval.models import load_model, BaseModel
from llm_eval.scaling_methods import load_scaling_method, BaseScalingMethod
from llm_eval.evaluation import get_evaluator, BaseEvaluator
from llm_eval.utils.logging import get_logger
from llm_eval.utils.util import EvaluationResult
from llm_eval.internal.benchhub_info import DATASETS as BENCHHUB_INFO_ENTRIES
from llm_eval.utils.prompt_template import (
    format_few_shot_prompt_prefix, 
    DEFAULT_FEW_SHOT_INSTRUCTION, 
    DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE,
)

from llm_eval.utils.metrics import language_penalizer

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
        custom_cot_parser: Optional[Callable[[str], Tuple[str, str]]] = None,
        num_few_shot: Optional[int] = 0,
        few_shot_split: Optional[str] = None,
        few_shot_instruction: Optional[str] = DEFAULT_FEW_SHOT_INSTRUCTION,
        few_shot_example_template: str = DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE,
    ):
        """
        Initialize the PipelineRunner with identifiers and parameters for each component.

        Args:
            dataset_name (str): Dataset identifier (registered in the dataset registry).
            subset (str | list[str] | None): Optional sub-task or configuration (e.g., "csat_geo").
            split (str): Dataset split to load ("train", "valid", "test", etc.).
            model_backend_name (str): Model backend identifier (e.g., "huggingface", "openai", "multi").
            scaling_method_name (str | None): Scaling (decoding) method identifier.
            evaluation_method_name (str): Evaluator identifier.
            dataset_params (dict): Additional parameters for the dataset loader.
            model_backend_params (dict): Additional parameters for the model backend.
            scaling_params (dict): Parameters for the scaling method.
            evaluator_params (dict): Parameters for the evaluator.
            language_penalize (bool): If True, apply language penalizer to predictions.
            target_lang (str): Target language code for penalization.
            custom_cot_parser (callable | None): Optional custom chain-of-thought parser function.
                If provided, this function will override the default cot_parser in the model backend.
                (For example, a string path can be dynamically loaded to use a custom parser.)
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

        # --- Custom CoT parser handling ---
        # If a custom cot_parser is provided, we inject it into model_backend_params so that
        # the model backend (e.g., HuggingFaceModel) can use it during initialization.
        self.custom_cot_parser = custom_cot_parser
        if self.custom_cot_parser is not None:
            self.model_backend_params["cot_parser"] = self.custom_cot_parser
        # ------------------------------------

        self.num_few_shot = num_few_shot if num_few_shot is not None else 0
        if not isinstance(self.num_few_shot, int) or self.num_few_shot < 0:
            logger.warning(f"Invalid num_few_shot value: {self.num_few_shot}. Setting to 0.")
            self.num_few_shot = 0
            
        self.few_shot_split = few_shot_split
        self.few_shot_instruction = few_shot_instruction
        self.few_shot_example_template = few_shot_example_template

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
        logger.info(f"[Pipeline] Loading dataset: {self.dataset_name}, subset={self.subset}, split={self.split}")
        self.dataset = load_datasets(
            name=self.dataset_name,
            subset=self.subset,
            split=self.split,
            **self.dataset_params
        )
        ds_info = self.dataset.info()
        scaling_only = ds_info.get("scaling_only", None)
        evaluation_only = ds_info.get("evaluation_only", None)

        logger.info(f"[Pipeline] Loading model backend: {self.model_backend_name} with params={self.model_backend_params}")
        self.model = load_model(self.model_backend_name, **self.model_backend_params)

        if self.scaling_method_name:
            if scaling_only is not None:
                if self.scaling_method_name not in scaling_only:
                    raise ValueError(
                        f"Dataset '{self.dataset_name}' only allows scaling methods {scaling_only}, but got '{self.scaling_method_name}'."
                    )
            logger.info(f"[Pipeline] Loading scaling method: {self.scaling_method_name} with params={self.scaling_params}")
            self.scaler = load_scaling_method(
                self.scaling_method_name,
                model=self.model,
                **self.scaling_params
            )
        else:
            if scaling_only is not None:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' requires a scaling method from {scaling_only}, but scaling_method_name is None."
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
                    f"Dataset '{self.dataset_name}' only allows evaluation methods {evaluation_only}, but got '{self.evaluation_method_name}'."
                )
        logger.info(f"[Pipeline] Loading evaluator: {self.evaluation_method_name} with params={self.evaluator_params}")
        if self.evaluation_method_name == "llm_judge":
            self.evaluator = get_evaluator(self.evaluation_method_name, model=self.model, **self.evaluator_params)
        else:
            self.evaluator = get_evaluator(self.evaluation_method_name, **self.evaluator_params)

    def _prepare_few_shot_prefix(self) -> str:
        if self.num_few_shot <= 0:
            logger.info("[FewShot] num_few_shot is 0 or less. No few-shot examples will be prepared.")
            return ""

        few_shot_source_split_actual = self.few_shot_split if self.few_shot_split else self.split
        logger.info(f"[FewShot] Attempting to prepare {self.num_few_shot} few-shot examples "
                    f"from dataset '{self.dataset_name}', subset '{self.subset}', split '{few_shot_source_split_actual}'.")

        try:
            # dataset_params에서 few-shot 관련 파라미터는 제외하고 전달 (무한 재귀 방지)
            fs_dataset_params = self.dataset_params.copy()
            for key in ['num_few_shot', 'few_shot_split', 'few_shot_instruction', 'few_shot_example_template']:
                fs_dataset_params.pop(key, None)
            
            logger.debug(f"[FewShot] Loading dataset for few-shot examples with params: {fs_dataset_params}")
            few_shot_ds_loader = load_datasets(
                name=self.dataset_name,
                subset=self.subset, # 동일한 subset 사용
                split=few_shot_source_split_actual,
                **fs_dataset_params 
            )
            few_shot_examples_raw = few_shot_ds_loader.load() # This returns List[Dict[str, Any]]
            
            if not few_shot_examples_raw:
                logger.warning(f"[FewShot] No data found for few-shot examples from split '{few_shot_source_split_actual}'. Returning empty prefix.")
                return ""
            
            num_available = len(few_shot_examples_raw)
            num_to_take = min(self.num_few_shot, num_available)

            if num_to_take == 0:
                 logger.warning(f"[FewShot] Not enough samples available ({num_available}) in '{few_shot_source_split_actual}' to create {self.num_few_shot} few-shot examples. No prefix generated.")
                 return ""
            
            if num_to_take < self.num_few_shot:
                logger.warning(f"[FewShot] Requested {self.num_few_shot} examples, but only {num_to_take} are available in '{few_shot_source_split_actual}'. Using {num_to_take} examples.")

            selected_few_shot_samples = few_shot_examples_raw[:num_to_take]
            logger.info(f"[FewShot] Selected {len(selected_few_shot_samples)} samples for few-shot prefix construction.")
            
            prefix_str = format_few_shot_prompt_prefix(
                selected_few_shot_samples,
                instruction=self.few_shot_instruction,
                example_template=self.few_shot_example_template
            )
            if not prefix_str:
                 logger.warning("[FewShot] Constructed few-shot prefix is empty. This might happen if all selected samples were invalid for formatting.")
            else:
                 logger.info(f"[FewShot] Successfully constructed few-shot prefix with {len(selected_few_shot_samples)} examples.")
            return prefix_str

        except Exception as e:
            logger.error(f"[FewShot] Critical error during loading or processing of few-shot examples from split '{few_shot_source_split_actual}': {e}", exc_info=True)
            return "" # Return empty string on error to allow main pipeline to proceed without few-shot
        
    def run(self) -> EvaluationResult:
        """
        Executes the entire pipeline:
          1. Dataset loading: Reads data from the dataset loader.
          2. Inference: Either skips if predictions exist or performs model generation/scaling.
          3. Evaluation: Evaluates predictions against references using the evaluator.
          4. Optionally applies the language penalizer to each prediction based on target_lang.
          5. Aggregates metrics, sample outputs, and pipeline info into an EvaluationResult.

        Returns:
            EvaluationResult: An object encapsulating final metrics, samples, and additional pipeline info.
        """
        if not self.dataset or not self.model or not self.evaluator:
            raise RuntimeError("Pipeline components are not fully loaded.")

        start_time = time.time()
        logger.info(f"Pipeline run started for dataset: {self.dataset_name}, split: {self.split}, model: {self.model_backend_name}")

        main_evaluation_data_raw = self.dataset.load()
        logger.info(f"Loaded {len(main_evaluation_data_raw)} samples for main evaluation from split '{self.split}'.")

        few_shot_prompt_prefix = self._prepare_few_shot_prefix()

        samples_for_processing = main_evaluation_data_raw
        if self.num_few_shot > 0 and (self.few_shot_split is None or self.few_shot_split == self.split):
            if len(main_evaluation_data_raw) <= self.num_few_shot:
                logger.warning(
                    f"Number of samples in '{self.split}' ({len(main_evaluation_data_raw)}) is less than or equal to "
                    f"num_few_shot ({self.num_few_shot}). No samples left for evaluation."
                )
                samples_for_processing = []
            else:
                logger.info(
                    f"Using first {self.num_few_shot} samples from '{self.split}' for few-shot examples. "
                    f"Evaluating on the remaining {len(main_evaluation_data_raw) - self.num_few_shot} samples."
                )
                samples_for_processing = main_evaluation_data_raw[self.num_few_shot:]
        
        if not samples_for_processing:
            logger.warning("No samples available for evaluation after processing few-shot examples. Returning empty result.")
            return EvaluationResult(
                metrics={}, 
                samples=[], 
                info={
                    "dataset_name": self.dataset_name, "subset": self.subset, "split": self.split,
                    "model_backend_name": self.model_backend_name,
                    "num_few_shot_applied": self.num_few_shot if few_shot_prompt_prefix else 0,
                    "error": "No samples for evaluation after few-shot example processing."
                }
            )

        # 각 샘플의 'input'에 few_shot_prompt_prefix 추가 
        samples_for_inference = []
        for sample in samples_for_processing:
            new_sample = sample.copy() 
            original_input = new_sample.get("input", "")
            new_sample["input"] = few_shot_prompt_prefix + original_input
            if self.num_few_shot > 0 and few_shot_prompt_prefix: # few-shot이 실제로 적용되었는지 추적
                new_sample["few_shot_applied"] = True 
            samples_for_inference.append(new_sample)
        
        logger.info(f"Prepared {len(samples_for_inference)} samples for model inference with few-shot prefixes (if any).")

        try:
            if self.scaler:
                logger.info(f"Applying scaling method: {self.scaling_method_name} to {len(samples_for_inference)} samples.")
                predictions_with_metadata = self.scaler.apply(samples_for_inference)
            else:
                logger.info(f"Performing direct model inference for {len(samples_for_inference)} samples.")
                predictions_with_metadata = self.model.generate_batch(samples_for_inference)
            logger.info(f"Inference completed for {len(predictions_with_metadata)} items.")
        except Exception as e:
            logger.error(f"Error during model inference or scaling: {e}", exc_info=True)
            return EvaluationResult(
                metrics={"error_on_inference": str(e)}, 
                samples=samples_for_inference, # Prefix가 적용된 input을 포함한 샘플 반환 (디버깅용)
                info={
                    "dataset_name": self.dataset_name, "subset": self.subset, "split": self.split,
                    "model_backend_name": self.model_backend_name, "error": f"Inference failed: {e}"
                }
            )

        logger.info(f"Starting evaluation with '{self.evaluation_method_name}'.")
        try:
            eval_dict = self.evaluator.evaluate(predictions_with_metadata, model=self.model)
        except Exception as e:
            logger.error(f"Error during evaluation with '{self.evaluation_method_name}': {e}", exc_info=True)
            return EvaluationResult(
                metrics={"error_on_evaluation": str(e)}, 
                samples=predictions_with_metadata, # 추론 결과까지는 포함하여 반환
                info={
                    "dataset_name": self.dataset_name, "subset": self.subset, "split": self.split,
                    "model_backend_name": self.model_backend_name, "error": f"Evaluation failed: {e}"
                }
            )


        if self.language_penalize:
            logger.info(f"Applying language penalizer with target language '{self.target_lang}'.")
            target_lang_for_penalty = self.target_lang
            language_scores = []
            evaluated_samples = eval_dict.get("samples", [])
            for sample_in_eval_result in evaluated_samples:
                pred_text = sample_in_eval_result.get("original_prediction", sample_in_eval_result.get("prediction", ""))
                if not isinstance(pred_text, str): # Ensure pred_text is a string
                    logger.debug(f"Prediction text is not a string, cannot apply language penalizer. Sample ID: {sample_in_eval_result.get('id', 'N/A')}, Type: {type(pred_text)}")
                    lp_score = 0.0 # Or some other default or skip
                else:
                    lp_score = language_penalizer(pred_text, target_lang=target_lang_for_penalty)
                sample_in_eval_result["language_penalizer"] = lp_score
                language_scores.append(lp_score)
            
            if language_scores: 
                avg_lp = sum(language_scores) / len(language_scores)
                eval_dict.setdefault("metrics", {})["language_penalizer_average"] = avg_lp
                logger.info(f"Average language penalizer score: {avg_lp:.4f}")
            else:
                eval_dict.setdefault("metrics", {})["language_penalizer_average"] = 0.0
                logger.info("No samples to calculate language penalizer average or all predictions were non-string.")


        end_time = time.time()
        elapsed_seconds = end_time - start_time
        logger.info(f"Pipeline run completed in {elapsed_seconds:.2f} seconds.")

        pipeline_info = {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "model_backend_name": self.model_backend_name,
            "scaling_method_name": self.scaling_method_name,
            "evaluation_method_name": self.evaluation_method_name,
            "elapsed_time_sec": elapsed_seconds,
            "num_few_shot_configured": self.num_few_shot,
            "num_few_shot_applied": self.num_few_shot if few_shot_prompt_prefix else 0, #실제 적용된 수
            "few_shot_source_split": self.few_shot_split if self.num_few_shot > 0 and few_shot_prompt_prefix else None,
            "language_penalize_enabled": self.language_penalize,
            "target_lang_for_penalty": self.target_lang if self.language_penalize else None,
        }
        
        # If the dataset is BenchHub, add benchmark details
        if self.dataset_name == "benchhub": #
            processed_samples = eval_dict.get("samples", [])
            if processed_samples:
                unique_benchmark_names = set()
                for sample in processed_samples:
                    if "metadata" in sample and "benchmark_name" in sample["metadata"]: #
                        unique_benchmark_names.add(sample["metadata"]["benchmark_name"]) #
                
                if unique_benchmark_names:
                    benchmark_details = {}
                    for bn_name in unique_benchmark_names:
                        for entry in BENCHHUB_INFO_ENTRIES: #
                            if entry.dataset == bn_name: # DatasetEntry.dataset과 비교
                                benchmark_details[bn_name] = {
                                    "citation_key": entry.citation_key, #
                                    "citation": entry.citation, #
                                    "license": entry.license, #
                                    "anthology": entry.anthology, #
                                    "languages": entry.languages, #
                                }
                                break
                    if benchmark_details:
                        pipeline_info['benchmark_details'] = benchmark_details
                        logger.info(f"BenchHub 벤치마크에 대한 인용 정보 추가 완료: {list(benchmark_details.keys())}")

        metrics = eval_dict.get("metrics", {})
        final_samples_output = eval_dict.get("samples", [])
        
        existing_info = eval_dict.get("info", {}) # evaluator가 info를 반환했을 경우
        merged_info = {**existing_info, **pipeline_info}

        return EvaluationResult(
            metrics=metrics,
            samples=final_samples_output,
            info=merged_info
        )