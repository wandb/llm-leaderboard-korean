import wandb
import pandas as pd
from typing import Any, Dict, List, Tuple, Callable
from llm_eval.utils.util import EvaluationResult
import weave
from weave import EvaluationLogger
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

class WandbController:
    def __init__(self, wandb_params: dict, dataset_name: str, model_name: str):
        self.wandb_params = wandb_params
        self.dataset_name = dataset_name
        self.model_name = model_name

    def log_to_wandb(self, result: EvaluationResult) -> None:
        """Log evaluation summary to Weights & Biases if configured."""
        table_name = self.dataset_name + "_leaderboard_table"
        data = {k: result.metrics.get(k) for k in {"model_name", "AVG", *result.metrics.keys()}}
        data["model_name"] = self.model_name
        df = pd.DataFrame([data])
        
        cols = ["model_name", "AVG"] + sorted([c for c in df.columns if c not in ["model_name", "AVG"]])
        df = df[cols]
        leaderboard_table = wandb.Table(dataframe=df)
        with wandb.init(
            entity=self.wandb_params.get("entity"),
            project=self.wandb_params.get("project"),
            name=self.model_name
            ) as run:
            run.log({table_name: leaderboard_table})


class WeaveInferenceController:
    """
    Handles Weave logging with automatic token/latency tracking during inference.

    This controller wraps LLM inference calls in Weave's log_prediction context manager
    to automatically capture token usage and latency metadata.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: Any,
        split: str,
        model_backend_name: str,
        model_name: str = None,
        scaling_method_name: str = None,
        evaluation_method_name: str = None,
        language_penalize: bool = False,
        target_lang: str = "ko",
        batch_size: int = 32,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.model_backend_name = model_backend_name
        self.model_name = model_name
        self.scaling_method_name = scaling_method_name
        self.evaluation_method_name = evaluation_method_name
        self.language_penalize = language_penalize
        self.target_lang = target_lang
        self.batch_size = batch_size
        self.evaluation_logger = None

    def initialize(self) -> EvaluationLogger:
        """Initialize Weave EvaluationLogger before inference."""
        # Build subset representation
        if isinstance(self.subset, list):
            subset_repr = "+".join(map(str, self.subset)) if self.subset else "all"
        else:
            subset_repr = str(self.subset) if self.subset is not None else "all"

        # Get model name for labeling
        model_name = self.model_name or self.model_backend_name
        model_label = str(model_name).replace("-", "_").replace(" ", "_").replace(".", "_")

        # Build metadata
        metadata: Dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "model_name": model_name,
            "scaling_method_name": self.scaling_method_name,
            "evaluation_method_name": self.evaluation_method_name,
            "language_penalize": self.language_penalize,
            "target_lang": self.target_lang,
        }

        # Initialize EvaluationLogger
        self.evaluation_logger = EvaluationLogger(
            dataset=str(self.dataset_name),
            model=model_label,
            eval_attributes=metadata,
        )

        logger.info("Initialized Weave EvaluationLogger for automatic token/latency tracking")
        return self.evaluation_logger

    def run_inference_with_logging(
        self,
        samples: List[Dict[str, Any]],
        inference_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Run inference with each LLM call wrapped in Weave log_prediction context.

        Args:
            samples: Input samples for inference
            inference_fn: Function that takes a single-element list and returns inference results
                         (e.g., model.generate_batch)

        Returns:
            List of samples with predictions
        """
        if self.evaluation_logger is None:
            raise RuntimeError("Must call initialize() before run_inference_with_logging()")

        def process_single_sample(sample_with_idx: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
            """Process a single sample with its own Weave context."""
            idx, sample = sample_with_idx
            try:
                # Prepare inputs for logging
                input_text = sample.get("input", "")
                reference_text = sample.get("reference", None)

                inputs_payload = {"input": input_text}
                if reference_text is not None:
                    inputs_payload["reference"] = reference_text

                # Use log_prediction as context manager to ensure LLM calls are tracked
                # This way litellm acompletion calls will be nested under the prediction
                with self.evaluation_logger.log_prediction(
                    inputs=inputs_payload,
                    output=""  # Will be populated automatically
                ):
                    # Run model inference INSIDE the prediction context
                    # Weave will automatically track token usage and latency
                    result_list = inference_fn([sample])
                    result = result_list[0] if result_list else sample

                # Store index for sorting results
                result["_sample_idx"] = idx

                return result

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}", exc_info=True)
                sample["prediction"] = ""
                sample["_sample_idx"] = idx
                return sample

        logger.info(f"Running inference with Weave logging for {len(samples)} samples (batch_size={self.batch_size})")

        # Use ThreadPoolExecutor for concurrent processing
        processed_samples = []
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            # Submit all samples for processing
            future_to_sample = {
                executor.submit(process_single_sample, (idx, sample)): idx
                for idx, sample in enumerate(samples)
            }

            # Collect results with progress bar
            with tqdm(total=len(samples), desc="Inference + Weave logging") as pbar:
                for future in as_completed(future_to_sample):
                    result = future.result()
                    processed_samples.append(result)
                    pbar.update(1)

        # Sort by original index to maintain order
        processed_samples.sort(key=lambda x: x.get("_sample_idx", 0))

        # Remove the temporary index field
        for sample in processed_samples:
            sample.pop("_sample_idx", None)

        logger.info(f"Completed inference with Weave logging for {len(processed_samples)} samples")
        return processed_samples

    def finalize(self, metrics: Dict[str, Any]) -> None:
        """Finalize Weave logging with summary metrics."""
        if self.evaluation_logger is None:
            logger.warning("EvaluationLogger not initialized, skipping finalization")
            return

        try:
            self.evaluation_logger.log_summary(summary=metrics or {})
            self.evaluation_logger.finish()
            logger.info("Finalized Weave logging with summary metrics")
        except Exception as e:
            logger.warning(f"Failed to finalize Weave logging: {e}")


class WeaveEvalsController:
    """Helper to log evaluations to Weave Evals using EvaluationLogger.

    Best-effort: Any exception is swallowed after a debug log in the caller.
    """

    @staticmethod
    def log(
        *,
        dataset_name: str,
        subset: Any,
        split: str,
        model_backend_name: str,
        model_name: str,
        scaling_method_name: Any,
        evaluation_method_name: str,
        language_penalize: bool,
        target_lang: str,
        samples: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        wandb_params: Dict[str, Any],
    ) -> None:
        from weave import EvaluationLogger 
        # Build evaluation name
        if isinstance(subset, list):
            subset_repr = "+".join(map(str, subset)) if subset else "all"
        else:
            subset_repr = str(subset) if subset is not None else "all"

        evaluation_name = str(model_name or model_backend_name)
        # Weave의 Model 표시는 클래스명을 사용하므로 파이썬 식별자 형태로 정규화 필요
        model_label = (model_name or model_backend_name) or "Model"
        model_label = str(model_label).replace("-", "_").replace(" ", "_").replace(".", "_")

        metadata: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "subset": subset,
            "split": split,
            "model_name": model_name or model_backend_name,
            "scaling_method_name": scaling_method_name,
            "evaluation_method_name": evaluation_method_name,
            "language_penalize": language_penalize,
            "target_lang": target_lang,
        }

        # Use current Weave API field names
        elog = EvaluationLogger(
            # name=evaluation_name,
            dataset=str(dataset_name),
            model=model_label,
            eval_attributes=metadata,
        )

        for item in samples:
            input_text = item.get("input", "")
            prediction_text = item.get("prediction", item.get("original_prediction", ""))
            reference_text = item.get("reference", None)

            per_item_metadata = {
                "_subset_name": item.get("_subset_name", None),
                "id": item.get("id", None),
            }

            # Build inputs payload (dict required by EvaluationLogger)
            inputs_payload: Dict[str, Any] = {"input": input_text}
            if reference_text is not None:
                inputs_payload["reference"] = reference_text
            # Carry helpful metadata in the inputs
            inputs_payload.update({k: v for k, v in per_item_metadata.items() if v is not None})

            subset_name = item.get("_subset_name", None)
            evaluation_fields = item.get("evaluation", {}) or {}
            # Optionally include normalized texts into inputs for easier inspection
            inputs_payload["dataset"] = str(dataset_name)
            if subset_name:
                inputs_payload["subset"] = str(subset_name)

            pred_logger = elog.log_prediction(inputs=inputs_payload, output=prediction_text)

            # 1) 평가 모듈이 만든 모든 메트릭은 이름 그대로 로깅
            for key, value in evaluation_fields.items():
                pred_logger.log_score(scorer=str(key), score=value)

            # 2) 러너 레벨에서 추가한 보조 스코어도 함께 기록(있을 때)
            if "language_penalizer" in item:
                pred_logger.log_score(scorer="language_penalizer", score=float(item["language_penalizer"]))
            

        elog.log_summary(summary=metrics or {})
        elog.finish()
