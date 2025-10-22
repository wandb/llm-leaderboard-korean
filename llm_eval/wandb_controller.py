import wandb
import pandas as pd
from typing import Any, Dict, List, Tuple
from llm_eval.utils.util import EvaluationResult
import weave

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


class WeaveSampleLogger:
    """Minimal helper to keep Weave Inputs and Outputs clean for per-sample traces.

    - Inputs: dataset_name, subset_name, input_text
    - Outputs: every field except 'input' (e.g., prediction, reference, evaluation columns)
    """

    _cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    @staticmethod
    def cache_sample(dataset_name: Any, item: Dict[str, Any]) -> None:
        input_text = str(item.get("input", ""))
        key = (str(dataset_name), input_text)
        WeaveSampleLogger._cache[key] = {
            'normalized_pred': item['evaluation']['normalized_pred'],
            'normalized_ref': item['evaluation']['normalized_ref'],
            'is_correct': item['evaluation']['is_correct'],
            }

    @staticmethod
    def make_op(op_name: str):
        @weave.op(name=op_name)
        def _op(dataset_name: str, subset_name: Any, input_text: str) -> Dict[str, Any]:
            key = (str(dataset_name), str(input_text))
            item = WeaveSampleLogger._cache.get(key, {})
            return {k: v for k, v in item.items() if k != "input"}

        return _op

    @staticmethod
    def log_samples(op_name: str, dataset_name: Any, samples: List[Dict[str, Any]]) -> None:
        op = WeaveSampleLogger.make_op(op_name=str(op_name))
        for s in samples or []:
            subset_name = s.get("_subset_name", None)
            WeaveSampleLogger.cache_sample(dataset_name, s)
            op(dataset_name, subset_name, str(s.get("input", "")))


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

            # 1) 평가 모듈이 만든 모든 스칼라 메트릭(숫자/불리언)은 이름 그대로 로깅
            for key, value in evaluation_fields.items():
                if isinstance(value, (int, float, bool)):
                    pred_logger.log_score(scorer=str(key), score=value)

            # 2) 러너 레벨에서 추가한 보조 스코어도 함께 기록(있을 때)
            if "language_penalizer" in item:
                pred_logger.log_score(scorer="language_penalizer", score=float(item["language_penalizer"]))
            

        elog.log_summary(summary=metrics or {})
        elog.finish()
