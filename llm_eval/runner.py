import logging
import time
from typing import Any, Dict, List, Optional, Union

from llm_eval.datasets import load_dataset, BaseDataset
from llm_eval.models import load_model, BaseModel
from llm_eval.scaling_methods import load_scaling_method, BaseScalingMethod
from llm_eval.evaluation import get_evaluator, BaseEvaluator
from llm_eval.utils.logging import get_logger

logger = get_logger(name="runner", level=logging.INFO)

class PipelineRunner:
    """
    전체 LLM 평가 파이프라인을 캡슐화하는 Runner 클래스.
    'dataset_name/subset/split' -> 데이터셋 로딩
    'model_name' -> 모델(백엔드) 로딩
    'scaling_method_name' -> 여러 추론 후보 중 최종 답을 고르는 스케일링(디코딩) 전략
    'evaluation_method_name' -> 최종적으로 모델 출력을 평가하는 방법

    사용 예시:
        runner = PipelineRunner(
            dataset_name="haerae_bench",
            subset=None,
            split="test",
            model_name="vllm",
            scaling_method_name="beam_search",
            evaluation_method_name="string_match",
            model_params={"endpoint": "http://localhost:8000"},
            scaling_params={"beam_size": 3, "num_iterations": 5},
        )
        results = runner.run()
        # results -> {"metrics": {...}, "samples": [...]} 식의 평가 결과
    """

    def __init__(
        self,
        dataset_name: str,
        subset: Union[str, List[str], None] = None,
        split: str = "test",
        model_name: str = "vllm",
        scaling_method_name: Optional[str] = None,
        evaluation_method_name: str = "string_match",
        dataset_params:  Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        scaling_params: Optional[Dict[str, Any]] = None,
        evaluator_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            dataset_name (str): 데이터셋 이름(레지스트리 등록된 것)
            subset (str | list[str] | None): 하위 태스크(예: "csat_geo" 등)
            split (str): "train"/"valid"/"test" 등
            model_name (str): 모델 식별자(레지스트리 등록된 것)
            scaling_method_name (str | None): 스케일링(디코딩) 방법 식별자
            evaluation_method_name (str): 평가 방법 식별자
            model_params (dict): 모델 초기화 시 필요한 추가 파라미터 (endpoint, API key, 등)
            scaling_params (dict): 스케일링 메서드 초기화 시 필요한 추가 파라미터 (beam_size, n, 등)
            evaluator_params (dict): evaluator 초기화 시 필요한 추가 파라미터
        """
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.model_name = model_name
        self.scaling_method_name = scaling_method_name
        self.evaluation_method_name = evaluation_method_name

        self.dataset_params = dataset_params or {}
        self.model_params = model_params or {}
        self.scaling_params = scaling_params or {}
        self.evaluator_params = evaluator_params or {}

        self.dataset: Optional[BaseDataset] = None
        self.model: Optional[BaseModel] = None
        self.scaler: Optional[BaseScalingMethod] = None
        self.evaluator: Optional[BaseEvaluator] = None

        # 사전 로드/초기화 (필요하다면 lazy-load 형태로 바꿀 수도 있음)
        self._load_components()

    def _load_components(self) -> None:
        """
        내부적으로 dataset, model, scaler, evaluator를 로드하거나 초기화한다.
        """
        # 1) Dataset
        logger.info(f"[Pipeline] Loading dataset: {self.dataset_name}, subset={self.subset}, split={self.split}")
        self.dataset = load_dataset(
            name=self.dataset_name,
            subset=self.subset,
            split=self.split,
            **self.dataset_params  # => 여기서는 dataset kwargs와 model_params 구분이 필요할 수도 있음
        )

        # 2) Model
        logger.info(f"[Pipeline] Loading model: {self.model_name} with {self.model_params}")
        self.model = load_model(self.model_name, **self.model_params)

        # 3) Scaling Method (optional)
        if self.scaling_method_name:
            logger.info(f"[Pipeline] Loading scaling method: {self.scaling_method_name} with {self.scaling_params}")
            self.scaler = load_scaling_method(
                self.scaling_method_name,
                model=self.model,  # 모델 인스턴스 주입
                **self.scaling_params
            )

        # 4) Evaluator
        logger.info(f"[Pipeline] Loading evaluator: {self.evaluation_method_name} with {self.evaluator_params}")
        self.evaluator = get_evaluator(self.evaluation_method_name)
        # evaluator_params를 내부적으로 쓸 수 있도록 세팅
        # (BaseEvaluator에 set_params 같은 메서드가 있다면 호출)
        # ex) self.evaluator.set_params(**self.evaluator_params)

    def run(self) -> Dict[str, Any]:
        """
        실제 파이프라인 실행: Dataset 로드 -> (Scaling) Inference -> Evaluation -> 결과 반환
        Returns:
            {
                "metrics": {...},
                "samples": [...],
                "info": {...}  # (선택) 부가적인 실행 정보
            }
        """
        if not self.dataset or not self.model or not self.evaluator:
            raise RuntimeError("Pipeline components are not fully loaded.")

        # 1) Dataset load
        start_time = time.time()
        data = self.dataset.load()  # [{"input":..., "reference":..., ...}, ...]
        logger.info(f"[Pipeline] Dataset loaded. # of samples: {len(data)}")

        # 2) Scaling / Inference
        if self.scaler:
            logger.info(f"[Pipeline] Applying scaling method: {self.scaling_method_name}")
            predictions = self.scaler.apply(data)  
        else:
            logger.info("[Pipeline] Direct model inference (no scaling).")
            predictions = self.model.generate_batch(data)
        logger.info(f"[Pipeline] Inference done for {len(predictions)} items.")

        # 3) Evaluation
        logger.info(f"[Pipeline] Evaluating with {self.evaluation_method_name}")
        eval_results = self.evaluator.evaluate(predictions, model=None)
        # eval_results 예시: {"metrics": {...}, "samples": [...], ...}

        # 4) 결과 요약
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[Pipeline] Done. Elapsed: {elapsed:.2f} sec")

        # 필요하다면 추가 정보 기록
        eval_results["info"] = {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "model_name": self.model_name,
            "scaling_method_name": self.scaling_method_name,
            "evaluation_method_name": self.evaluation_method_name,
            "elapsed_time_sec": elapsed,
            # etc.
        }

        return eval_results


if __name__ == "__main__":
    """
    이 부분은 '프로덕션 레벨'에서 별도의 CLI 스크립트를 두는 대신,
    필요하다면 여기서 간단히 테스트하거나
    python runner.py --some_args 로 쓸 수 있도록 구성한 용도
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="LLM Eval PipelineRunner (optional CLI)")
    parser.add_argument("--dataset", required=True, help="Name of dataset in registry")
    parser.add_argument("--subset", default=None, help="Subset or config name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--model", required=True, help="Model name in registry")
    parser.add_argument("--scaling_method", default=None, help="Scaling method name in registry")
    parser.add_argument("--evaluation_method", default="string_match", help="Evaluator name in registry")
    parser.add_argument("--output_file", default=None, help="Where to save JSON results (optional)")

    args = parser.parse_args()

    runner = PipelineRunner(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        model_name=args.model,
        scaling_method_name=args.scaling_method,
        evaluation_method_name=args.evaluation_method,
        dataset_params={},
        model_params={},       # 여기에 endpoint, api_key 등...
        scaling_params={},     # 예: {"beam_size": 3, "num_iterations": 5}
        evaluator_params={}    # 평가 메서드가 필요로 하는 파라미터들
    )
    results = runner.run()

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))
