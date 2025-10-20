from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.logging import get_logger


logger = get_logger(name="comet_score")


def comet_score(comet_data: List[Dict[str, str]], batch_size: int = 8) -> List[float]:
    """
    COMET 점수를 계산해 각 문장 쌍의 품질을 반환합니다.

    Args:
        comet_data: {"src": str, "mt": str, "ref": str} 항목의 리스트
        batch_size: 배치 크기 (기본 8)

    Returns:
        각 항목에 대한 COMET 점수 리스트 (float)
    """
    try:
        from comet import download_model, load_from_checkpoint
    except Exception as e:
        raise RuntimeError(
            "unbabel-comet 패키지가 필요합니다. requirements.txt에 추가 후 설치하세요."
        ) from e

    # GPU 사용 가능 여부 자동 감지
    try:
        import torch

        gpus = 1 if torch.cuda.is_available() else 0
    except Exception:
        gpus = 0

    logger.info("Downloading/loading COMET model: Unbabel/wmt22-comet-da")
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    logger.info("Scoring with COMET ...")
    predictions = model.predict(
        comet_data, batch_size=batch_size, gpus=gpus, progress_bar=False, num_workers=1
    )
    return predictions.scores


@register_evaluator("comet_score")
class COMETEvaluator(BaseEvaluator):
    """Machine Translation 품질을 COMET으로 평가하는 평가기.

    입력 샘플은 다음 키를 포함해야 합니다.
      - input: 원문(Source)
      - prediction: 모델 번역(MT)
      - reference: 기준 번역(Reference)

    메트릭:
      - AVG: 전체 평균 COMET 점수
      - comet: AVG와 동일한 전체 평균 점수
      - <subset>/comet: 서브셋별 평균 점수 (subsets가 요청된 경우)
    각 샘플에는 evaluation.comet 필드로 개별 점수가 기록됩니다.
    """

    name = "comet_score"

    def __init__(self, batch_size: int = 8, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def parse_prediction(self, raw_output: Any) -> str:
        if raw_output is None:
            return ""
        return str(raw_output)

    def _build_comet_inputs(self, samples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        data: List[Dict[str, str]] = []
        for s in samples:
            src = s.get("input", "")
            mt = s.get("prediction", "")
            ref = s.get("reference", None)

            # ref가 없으면 참조 없는 품질 추정은 지원하지 않으므로 빈 문자열로 대체
            ref = ref if ref is not None else ""

            data.append({"src": str(src), "mt": str(mt), "ref": str(ref)})
        return data

    def evaluate_predictions(
        self, subsets: Optional[List[str]], samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        if not samples:
            return {"AVG": 0.0, "comet": 0.0}

        # COMET 입력 구성
        comet_inputs = self._build_comet_inputs(samples)

        # 점수 계산
        try:
            scores = comet_score(comet_inputs, batch_size=self.batch_size)
        except Exception as e:
            logger.error(f"COMET 평가 중 오류 발생: {e}")
            # 실패 시 0.0 점수로 대체
            scores = [0.0] * len(samples)

        # 샘플별 기록 및 집계
        total = 0.0
        subset_stats: Dict[str, Dict[str, float]] = {}
        if subsets:
            for s in subsets:
                subset_stats[s] = {"sum": 0.0, "count": 0.0}

        for idx, sample in enumerate(samples):
            sc = float(scores[idx]) if idx < len(scores) else 0.0
            total += sc

            subset_name = sample.get("_subset_name")
            if subset_name and subset_name not in subset_stats:
                subset_stats[subset_name] = {"sum": 0.0, "count": 0.0}
            if subset_name:
                subset_stats[subset_name]["sum"] += sc
                subset_stats[subset_name]["count"] += 1

            sample.setdefault("evaluation", {})
            sample["evaluation"]["comet"] = sc

        avg = total / len(samples)
        metrics: Dict[str, float] = {"final_score": avg, "comet": avg}

        if subsets:
            for sname, st in subset_stats.items():
                cnt = st["count"]
                s_avg = (st["sum"] / cnt) if cnt > 0 else 0.0
                metrics[f"{sname}/comet"] = s_avg

        return metrics


