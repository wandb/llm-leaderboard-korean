import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from tqdm import tqdm

from llm_eval.models.multi import MultiModel
from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.prompt_template import JUDGE_PROMPTS, JudgeType
from llm_eval.utils.logging import get_logger

logger = get_logger(name="llm_judge", level=logging.INFO)


@dataclass
class JudgeInput:
    """
    Data structure representing the input necessary for the judge system.
    """
    input: str
    judge_type: JudgeType
    model_response: str
    rubric: Optional[str] = None
    gold_response: Optional[str] = None
    model_response_b: Optional[str] = None


class ResponseParser:
    """Base class for parsing the LLM's returned response."""

    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        raise NotImplementedError


class RubricScoreParser(ResponseParser):
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        score_pattern = r"\[\[score:\s*(\d+\.?\d*)\]\]"
        match = re.search(score_pattern, response)
        if not match:
            raise ValueError(f"No valid score found in response: {response}")
        return {
            "score": float(match.group(1)),
            "model_name": model_name or "unknown"
        }


class PairwiseComparisonParser(ResponseParser):
    def parse(self, response: str, model_a: str = None, model_b: str = None) -> Dict[str, Any]:
        winner_pattern = r"\[\[([AB])\]\]"
        match = re.search(winner_pattern, response)
        if not match:
            raise ValueError(f"No valid winner found in response: {response}")
        winner = match.group(1)
        model_name = model_a if winner == "A" else model_b
        return {
            "winner": winner,
            "model_name": model_name or "unknown"
        }


class K2EvalResponseParser(ResponseParser):
    """
    Extracts a numeric score from a response in K2-Eval format.
    Expected format: a line like "Score: 3.5" somewhere in the text.
    """

    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        pattern = r"Score:\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, response)
        if not match:
            raise ValueError(f"No valid score found in K2-Eval response: {response}")
        score = float(match.group(1))
        return {
            "score": score,
            "model_name": model_name or "unknown"
        }


class GoldComparisonParser(ResponseParser):
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        resp_lower = response.lower()
        if "[[true]]" in resp_lower:
            return {"correct": True, "model_name": model_name or "unknown"}
        elif "[[false]]" in resp_lower:
            step_pattern = r"step:\s*\[(\d+)\]"
            match = re.search(step_pattern, response, flags=re.IGNORECASE)
            step = int(match.group(1)) if match else None
            return {"correct": False, "step": step, "model_name": model_name or "unknown"}
        raise ValueError(f"No valid verdict found in response: {response}")


class ResponseComparisonParser(ResponseParser):
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        match = re.search(r'\[\[([AB])\]\]', response)
        if not match:
            raise ValueError(f"No valid verdict [[A]] or [[B]] found in response: {response}")
        winner = match.group(1)
        return {"winner": winner}


@register_evaluator("llm_judge")
class LLMJudgeEvaluator(BaseEvaluator):
    """
    Evaluator that uses an LLM-as-a-Judge approach to assess model responses.
    Judge evaluation results (raw LLM output) are stored separately so that
    the original model generation result in sample["prediction"] is preserved.
    """
    name = "llm_judge"
    has_custom_judge = True

    def __init__(
            self,
            model: MultiModel,
            default_judge_type: Union[str, JudgeType] = "rubric_and_response",
            **kwargs
    ):
        super().__init__()
        self.default_judge_type = (
            JudgeType(default_judge_type)
            if isinstance(default_judge_type, str)
            else default_judge_type
        )
        self.multi_judge_model = model
        self.parsers = {
            JudgeType.RUBRIC_AND_RESPONSE: RubricScoreParser(),
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: GoldComparisonParser(),
            JudgeType.RESPONSE_COMPARISON: PairwiseComparisonParser(),
            JudgeType.K2_EVAL: K2EvalResponseParser()
        }
        self.prompt_templates = JUDGE_PROMPTS

    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        judge_type = sample.get("judge_type", self.default_judge_type.value)
        template = self.prompt_templates.get(JudgeType(judge_type))
        if template is None:
            raise ValueError(f"No template found for judge_type: {judge_type}")
        return template.format(**sample)

    def evaluate_predictions(self, subsets: Optional[List[str]], samples: List[Dict[str, Any]]) -> Dict[str, float]:
        if not samples:
            return {}

        total_score = 0.0
        score_count = 0
        total_correct = 0
        total_items = len(samples)

        # TODO: dataclass로 변경 고려
        ksat_result = defaultdict(lambda: defaultdict(dict))
        batch_inputs = []
        batch_indices = []

        for i, sample in enumerate(samples):
            try:
                judge_type_str = sample.get("judge_type", self.default_judge_type.value)
                j_type = JudgeType(judge_type_str)

                # Prepare prompt based on judge type.
                if j_type == JudgeType.RESPONSE_COMPARISON:
                    filled_prompt = (
                        f"Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants. "
                        f"Choose the assistant that follows the instructions and answers the question better.\n\n"
                        f"### Instruction:\n{sample.get('input', '').strip()}\n\n"
                        f"### Response A:\n{sample.get('prediction', '').strip()}\n\n"
                        f"### Response B:\n{sample.get('model_response_b', '').strip()}\n\n"
                        "Provide your verdict in this exact format: [[A]] or [[B]]."
                    )
                else:
                    template = self.prompt_templates.get(j_type)
                    if template is None:
                        raise ValueError(f"No template found for judge_type: {j_type}")
                    filled_prompt = template.format(
                        input=sample.get("input", "").strip(),
                        rubric=sample.get("rubric", "").strip(),
                        response=sample.get("prediction", "").strip(),
                        gold=sample.get("reference", "").strip(),
                        response_b=sample.get("model_response_b", "").strip()
                    )

                batch_inputs.append({"input": filled_prompt})
                batch_indices.append(i)
            except Exception as e:
                logger.error(f"Error preparing prompt: {e}")

        if batch_inputs:
            try:
                logger.info(f"LLMJudgeEvaluator: Calling judge_batch for {len(batch_inputs)} samples")
                judge_responses = self.multi_judge_model.judge_batch(batch_inputs)
                # Process each judge response
                for response_idx, sample_idx in enumerate(batch_indices):
                    sample = samples[sample_idx]
                    judge_response = judge_responses[response_idx]["prediction"]

                    # Instead of overwriting the original generation result,
                    # we store the judge output separately.
                    sample["judge_evaluation"] = judge_response

                    j_type = JudgeType(sample.get("judge_type", self.default_judge_type.value))

                    # For K2_EVAL judge type, use K2EvalResponseParser to extract score.
                    if j_type == JudgeType.K2_EVAL:
                        parser = self.parsers.get(JudgeType.K2_EVAL)
                        try:
                            parsed = parser.parse(judge_response)
                            sample["judge_score"] = parsed["score"]
                            total_score += parsed["score"]
                            score_count += 1
                        except Exception as e:
                            logger.error(f"Error parsing K2-Eval response: {e}")
                    elif j_type == JudgeType.RUBRIC_AND_RESPONSE:
                        if "judge_score" in judge_responses[response_idx]:
                            score = judge_responses[response_idx]["judge_score"]
                            if isinstance(score, (int, float)):
                                sample["judge_score"] = score
                                total_score += score
                                score_count += 1
                    elif j_type == JudgeType.KOREAN_SAT:
                        query_id = sample['metadata']['query_id']
                        year = query_id[:4]
                        question_num = sample['metadata']['question_num']
                        score_value = float(sample["metadata"]['score'])

                        # Initialize nested structure if not exists
                        if query_id not in ksat_result[year]:
                            ksat_result[year][query_id] = {
                                'question_num': question_num,
                                'score_value': score_value,
                                'is_correct': False
                            }

                        # Update correctness
                        if sample['judge_evaluation'] == sample['reference']:
                            total_score += score_value
                            ksat_result[year][query_id]['is_correct'] = True
                            print(f"정답!")
                        else:
                            ksat_result[year][query_id]['is_correct'] = False

                    elif j_type == JudgeType.RESPONSE_COMPARISON:
                        # Use a pairwise comparison parser.
                        parser = self.parsers.get(JudgeType.RESPONSE_COMPARISON)
                        try:
                            parsed = parser.parse(judge_response)
                            sample["evaluation"] = {
                                "raw_output": judge_response,
                                "parsed": parsed,
                            }
                            if "winner" in parsed:
                                winner = parsed["winner"]
                                sample["judge_winner"] = winner
                                sample["evaluation"]["winner"] = winner
                                if winner == "A":
                                    model_name = sample.get("model_a", "unknown")
                                else:
                                    model_name = sample.get("model_b", "unknown")
                                sample["evaluation"]["parsed"]["model_name"] = model_name
                                is_correct = sample.get("reference") == winner
                                sample["evaluation"]["is_correct"] = is_correct
                                sample["judge_correct"] = is_correct
                                if is_correct:
                                    total_correct += 1
                        except ValueError as e:
                            logger.error(f"Error parsing response comparison: {e}")
                    # For other judge types, similar processing can be added.

            except Exception as e:
                logger.error(f"Error in judge_batch: {e}")
                return {"error": str(e)}

        # 전체 메트릭 유지 (가능한 경우)
        metrics: Dict[str, float] = {}
        if score_count > 0:
            metrics["average_judge_score"] = total_score / score_count
        # Korean SAT specific metrics
        if (total_items > 0) and len(ksat_result) > 0:
            from llm_eval.datasets import load_datasets
            from llm_eval.utils.korean_sat_tools import ScoreCalculator

            loader = load_datasets(name="korean_sat_grading_info", split="test")
            grading_info = loader.load()
            ksat_calculator = ScoreCalculator(grading_info)
            metrics["average_score_overall_years"] = total_score / total_items

            for year, year_data in ksat_result.items():
                # Initialize metrics for this year
                metrics[f"{year}_total_score"] = 0.0
                metrics[f"{year}_common_score"] = 0.0
                metrics[f"{year}_choice_score"] = 0.0
                for qid, scoring_data in year_data.items():
                    if scoring_data["is_correct"]:
                        score_val = scoring_data['score_value']
                        metrics[f"{year}_total_score"] += score_val
                        # 2022년 이후 공통과목/선택과목 분리
                        question_num = int(scoring_data['question_num'])
                        if int(year) > 2021:
                            if question_num < 34:
                                metrics[f"{year}_common_score"] += score_val
                            else:
                                metrics[f"{year}_choice_score"] += score_val
                        else:
                            metrics[f"{year}_common_score"] += score_val
                grade_info = ksat_calculator.calculate_result(year=year, score_result=metrics)
                metrics[f"{year}_raw_score"] = grade_info.raw_score
                metrics[f"{year}_std_score"] = grade_info.std_score
                metrics[f"{year}_percentile"] = grade_info.percentile
                metrics[f"{year}_grade"] = grade_info.grade

        if total_items > 0 and any(
                sample.get("judge_type", self.default_judge_type.value) == JudgeType.RESPONSE_COMPARISON.value
                for sample in samples
        ):
            metrics["judge_accuracy"] = total_correct / total_items

        # subset 별 AVG만 추가 (subset 없는 샘플은 제외) + all 집계
        subset_stats: Dict[str, Dict[str, float]] = {}
        if subsets:
            for s in subsets:
                subset_stats[s] = {"total": 0.0, "sum_score": 0.0, "correct": 0.0}
        for sample in samples:
            sname = sample.get("_subset_name")
            if not sname:
                # subsets 지정이고 subset 미지정 샘플은 per-subset 계산에서 제외
                continue
            if sname not in subset_stats:
                subset_stats[sname] = {"total": 0.0, "sum_score": 0.0, "correct": 0.0}

            subset_stats[sname]["total"] += 1

            # 점수형
            if "judge_score" in sample and isinstance(sample.get("judge_score"), (int, float)):
                subset_stats[sname]["sum_score"] += float(sample["judge_score"])

            # 이진 정확도형 (pairwise 등)
            if sample.get("judge_correct") is True:
                subset_stats[sname]["correct"] += 1

        if isinstance(subsets, (list, tuple, str)):
            if isinstance(subsets, str):
                subsets = [subsets]
            for sname, st in subset_stats.items():
                denom = st["total"] if st["total"] > 0 else 1.0
                # 우선순위: score 기반 평균 -> 정확도 기반 -> 0.0
                avg_score = st["sum_score"] / denom if st["sum_score"] > 0 else None
                acc = st["correct"] / denom if st["correct"] > 0 else None
                if avg_score is not None:
                    metrics[f"{sname}/average_judge_score"] = avg_score
                elif acc is not None:
                    metrics[f"{sname}/judge_accuracy"] = acc
                else:
                    metrics[f"{sname}/judge_accuracy"] = 0.0

        # 전체 AVG는 항상 포함: 점수 평균 우선, 없으면 정확도, 없으면 0.0
        if score_count > 0:
            metrics["AVG"] = total_score / score_count
        elif total_items > 0 and any(
                sample.get("judge_type", self.default_judge_type.value) == JudgeType.RESPONSE_COMPARISON.value
                for sample in samples
        ):
            metrics["AVG"] = total_correct / total_items
        else:
            metrics["AVG"] = 0.0

        return metrics
