import logging
import re
from typing import Optional, Dict, Any, List, Union

from llm_eval.models.multi import MultiModel
from llm_eval.models import load_model, BaseModel as _BaseGen
from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.prompt_template import JudgeType
from llm_eval.utils.logging import get_logger

logger = get_logger(name="mt_bench_judge", level=logging.INFO)


@register_evaluator("mt_bench_judge")
class MTBenchJudgeEvaluator(BaseEvaluator):
    """
    MT Bench 전용 LLM-as-a-Judge 평가자.
    - 데이터셋에서 제공하는 judge_prompt_template, ref_answer_1 등을 활용해 프롬프트 구성
    - 한국어 점수 포맷(평가: [[N]] / [[N]]) 파싱 지원
    """

    name = "mt_bench_judge"
    has_custom_judge = True

    def __init__(
        self,
        model: Optional[Union[MultiModel, _BaseGen]] = None,
        **kwargs
    ):
        super().__init__()
        # Prefer judge from MultiModel if provided
        if isinstance(model, MultiModel) and getattr(model, "judge_model", None) is not None:
            # Use MultiModel routing so downstream calls stay consistent
            self._judge = model
            return

        # Otherwise, build internal judge backend from kwargs (evaluation.params)
        # Accept 'judge_backend_name' (preferred) or legacy 'name'
        judge_backend_name = kwargs.get("judge_backend_name") or kwargs.get("name")
        judge_params = {k: v for k, v in kwargs.items() if k not in ("name", "judge_backend_name")}
        if not judge_backend_name:
            raise TypeError("mt_bench_judge requires a judge backend via 'name' or 'judge_backend_name' in evaluation params when no MultiModel.judge_model is available")
        self._judge = load_model(judge_backend_name, **judge_params)

    def _build_prompt(self, sample: Dict[str, Any]) -> str:
        # sample-provided judge template 우선 (한국어 MT-Bench)
        custom_tpl = sample.get("judge_prompt_template")
        if isinstance(custom_tpl, str) and custom_tpl.strip():
            return custom_tpl.format(
                question=sample.get("question_1", sample.get("input", "")).strip(),
                question_1=sample.get("question_1", ""),
                answer=sample.get("prediction", "").strip(),
                answer_1=sample.get("prediction", "").strip(),
                ref_answer_1=sample.get("ref_answer_1", ""),
                input=sample.get("input", "").strip(),
            )

        # 폴백: 간단한 한국어 루브릭
        rubric = sample.get("rubric", "다음 응답을 1~10점으로 평가하고 형식을 준수하세요.")
        return (
            f"[지시]\n{rubric}\n\n[질문]\n{sample.get('question_1', sample.get('input', '')).strip()}\n\n"
            f"[도움말 답변 시작]\n{sample.get('prediction', '').strip()}\n[도움말 답변 종료]\n"
            "평가: [[N]]"
        )

    def evaluate_predictions(self, subsets: Optional[List[str]], samples: List[Dict[str, Any]]) -> Dict[str, float]:
        if not samples:
            return {}

        total_score = 0.0
        score_count = 0

        batch_inputs: List[Dict[str, Any]] = []
        batch_indices: List[int] = []

        for i, sample in enumerate(samples):
            try:
                prompt = self._build_prompt(sample)
                batch_inputs.append({"input": prompt})
                batch_indices.append(i)
            except Exception as e:
                logger.error(f"Prompt build failed: {e}")

        if batch_inputs:
            logger.info(f"MTBenchJudgeEvaluator: Calling judge_batch for {len(batch_inputs)} samples")
            judge_responses = self._judge.judge_batch(batch_inputs)
            for response_idx, sample_idx in enumerate(batch_indices):
                sample = samples[sample_idx]
                judge_response = judge_responses[response_idx].get("prediction", "")
                sample["judge_evaluation"] = judge_response

                # 한국어 점수 포맷 파싱: 평가: [[N]] 또는 [[N]]
                try:
                    m = re.search(r"평가\s*:\s*\[\[(\d+(?:\.\d+)?)\]\]", judge_response)
                    if not m:
                        m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", judge_response)
                    if m:
                        score = float(m.group(1))
                        sample["judge_score"] = score
                        total_score += score
                        score_count += 1
                except Exception:
                    pass

        metrics: Dict[str, float] = {}
        if score_count > 0:
            metrics["average_judge_score"] = total_score / score_count
            metrics["AVG"] = total_score / score_count
        else:
            metrics["AVG"] = 0.0

        # Subset-wise aggregation (by sample['_subset_name'])
        subset_stats: Dict[str, Dict[str, float]] = {}
        for sample in samples:
            sname = sample.get("_subset_name")
            if not sname:
                continue
            if sname not in subset_stats:
                subset_stats[sname] = {"total": 0.0, "sum_score": 0.0}
            subset_stats[sname]["total"] += 1
            if isinstance(sample.get("judge_score"), (int, float)):
                subset_stats[sname]["sum_score"] += float(sample["judge_score"])

        for sname, st in subset_stats.items():
            denom = st["total"] if st["total"] > 0 else 1.0
            avg_score = st["sum_score"] / denom if st["sum_score"] > 0 else 0.0
            metrics[f"{sname}/average_judge_score"] = avg_score

        return metrics


