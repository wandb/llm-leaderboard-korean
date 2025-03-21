import logging
import time
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union

from llm_eval.models.multi import MultiModel
from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.prompt_template import JUDGE_PROMPTS, JudgeType
from llm_eval.utils.logging import get_logger
import logging
from tqdm import tqdm

logger = get_logger(name="llm_judge", level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# 1) JudgeType, JudgeInput, Parser classes (original code kept intact)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeInput:
    """
    Data structure representing the input necessary for the judge system.

    Attributes:
        judge_type: Type of evaluation to perform (rubric vs. comparison).
        model_response: The primary model's response to evaluate.
        rubric: Evaluation criteria (if any).
        gold_response: Gold standard answer for comparison-based evaluation.
        model_response_b: An additional model response for pairwise comparison.
    """
    judge_type: JudgeType
    model_response: str
    rubric: Optional[str] = None
    gold_response: Optional[str] = None
    model_response_b: Optional[str] = None


class ResponseParser:
    """Base class for parsing the LLM's returned response."""
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        """
        Parse a raw LLM response into a more structured form.

        Args:
            response: The raw LLM output string.
            model_name: (Optional) the name or identifier of the model generating the response.

        Returns:
            A dictionary containing the parsed information.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError


class RubricScoreParser(ResponseParser):
    """
    Parser for extracting a numeric score from the LLM response.

    Format example in the LLM response: [[score: 4.5]].
    """
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        if not response:
            raise ValueError("Response is None")
        score_pattern = r"\[\[score:\s*(\d+\.?\d*)\]\]"
        match = re.search(score_pattern, response)
        if not match:
            raise ValueError(f"No valid score found in response: {response}")
        return {
            "score": float(match.group(1)),
            "model_name": model_name or "unknown"
        }


class PairwiseComparisonParser(ResponseParser):
    """
    Parser for pairwise winner selection in a comparative evaluation.
    Format example: [[A]] or [[B]]
    """
    def parse(self, response: str, model_a: str = None, model_b: str = None) -> Dict[str, Any]:
        if not response:
            raise ValueError("Response is None")
        
        winner_pattern = r"\[\[([AB])\]\]"
        match = re.search(winner_pattern, response)
        if not match:
            raise ValueError(f"No valid winner found in response: {response}")
            
        winner = match.group(1)
        # Map winner (A/B) to corresponding model name
        model_name = model_a if winner == "A" else model_b
        
        return {
            "winner": winner,
            "model_name": model_name or "unknown"
        }


class GoldComparisonParser(ResponseParser):
    """
    Parser to determine whether a response is correct compared to the gold standard.

    Format example in the LLM response: [[true]] or [[false]] with optional step number.
    """
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        if not response:
            raise ValueError("Response is None")
        # lower() for case-insensitivity
        resp_lower = response.lower()
        if "[[true]]" in resp_lower:
            return {"correct": True, "model_name": model_name or "unknown"}
        elif "[[false]]" in resp_lower:
            step_pattern = r"step:\s*\[(\d+)\]"
            match = re.search(step_pattern, response, flags=re.IGNORECASE)
            if match:
                return {
                    "correct": False,
                    "step": int(match.group(1)),
                    "model_name": model_name or "unknown"
                }
            return {"correct": False, "step": None, "model_name": model_name or "unknown"}
        raise ValueError(f"No valid verdict found in response: {response}")


class ResponseComparisonParser(ResponseParser):
    """Parser for A/B comparison responses."""
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        # 정규식으로 [[A]] 또는 [[B]] 찾기
        match = re.search(r'\[\[([AB])\]\]', response)
        if not match:
            raise ValueError(f"No valid verdict [[A]] or [[B]] found in response: {response}")
        winner = match.group(1)
        return {
            "winner": winner,
            "raw_response": response
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2) MultiLLMJudge - updated to use the new "multi.py" structure
# ─────────────────────────────────────────────────────────────────────────────

class MultiLLMJudge:
    """
    Class responsible for performing evaluations using a single "judge model" via MultiModel.

    - We assume models_config is a list with exactly ONE dict 
      describing the Judge LLM (since the new MultiModel can store only one judge_model).
    - We use self.multi_model.judge_batch(...) to get the LLM's outputs, 
      then parse them with the appropriate parser.
    """

    def __init__(
        self,
        models_config: List[Dict[str, Any]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        aggregation_strategy: str = "majority",
    ):
        """
        Args:
            models_config: a list with exactly ONE item, describing the judge LLM.
               e.g., [{"name": "huggingface_judge", "params": {...}}]
            max_retries, retry_delay, aggregation_strategy: not used in this minimal example,
               but kept for potential expansions (like retrying on API failures).
        """
        if not models_config or len(models_config) > 1:
            raise ValueError(
                f"MultiLLMJudge currently supports exactly 1 judge model config, got {len(models_config)}."
            )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.aggregation_strategy = aggregation_strategy

        # Prepare the "judge_model" param for MultiModel
        judge_config = models_config[0]
        # We create a multi_model that only has "judge_model"
        self.multi_model = MultiModel(
            judge_model=judge_config
        )

        # Mappings from JudgeType -> Parser
        self.parsers = {
            JudgeType.RUBRIC_AND_RESPONSE: RubricScoreParser(),
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: GoldComparisonParser(),
            JudgeType.RESPONSE_COMPARISON: PairwiseComparisonParser()
        }
        # The prompt templates used for generating a judge prompt
        self.prompt_templates = JUDGE_PROMPTS  # e.g. {JudgeType.RUBRIC_AND_RESPONSE: "Rubric: {rubric}\n...", ...}

    def judge(self, judge_inputs: List[JudgeInput]) -> List[Dict[str, Any]]:
        """
        Executes LLM-based evaluations for each JudgeInput by constructing appropriate prompts,
        calling multi_model.judge_batch(...), and parsing the results.

        Args:
            judge_inputs: List of JudgeInput, each containing the type of evaluation and necessary data.

        Returns:
            A list of dictionaries, each containing:
                - "raw_output": The raw LLM output
                - "parsed": The parser's output (e.g., score, correctness)
                - "judge_type": The type of evaluation (string)
        """
        if not judge_inputs:
            return []

        # Build prompts
        prompts = []
        judge_types = []
        for j_input in judge_inputs:
            template = self.prompt_templates.get(j_input.judge_type, "")
            filled_prompt = template.format(
                rubric=j_input.rubric or "",
                response=j_input.model_response or "",
                gold=j_input.gold_response or "",
                response_b=j_input.model_response_b or ""
            )
            # Our MultiModel judge_batch expects a "input" field per sample
            prompts.append({"input": filled_prompt})
            judge_types.append(j_input.judge_type)

        # Call judge_batch to get raw outputs
        judged_outputs = self.multi_model.judge_batch(prompts)

        # Parse each output according to its JudgeType
        results = []
        for out, j_type in zip(judged_outputs, judge_types):
            raw_output = out.get("prediction", "")
            parser = self.parsers[j_type]

            try:
                parsed_res = parser.parse(raw_output, model_name="JudgeLLM")
            except ValueError as e:
                parsed_res = {"error": str(e)}

            results.append({
                "raw_output": raw_output,
                "parsed": parsed_res,
                "judge_type": j_type.value
            })

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 3) LLMJudgeEvaluator: BaseEvaluator → integrates with the standard pipeline
# ─────────────────────────────────────────────────────────────────────────────

@register_evaluator("llm_judge")
class LLMJudgeEvaluator(BaseEvaluator):
    """
    Evaluator that uses an LLM-as-a-Judge approach to assess the quality of model responses.
    """
    name = "llm_judge"

    def __init__(
        self,
        model: MultiModel,
        default_judge_type: Union[str, JudgeType] = "rubric_and_response",
        **kwargs
    ):
        super().__init__()
        self.default_judge_type = JudgeType(default_judge_type) if isinstance(default_judge_type, str) else default_judge_type
        self.multi_judge_model = model
        self.parsers = {
            JudgeType.RUBRIC_AND_RESPONSE: RubricScoreParser(),
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: GoldComparisonParser(),
            JudgeType.RESPONSE_COMPARISON: PairwiseComparisonParser(),
        }
        self.prompt_templates = JUDGE_PROMPTS

    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        template = self.prompt_templates.get(self.default_judge_type)
        if template is None:
            raise ValueError(f"No template found for judge_type: {self.default_judge_type}")
            
        return template.format(**sample)

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        if not samples:
            return {}

        total_score = 0.0
        score_count = 0
        total_correct = 0
        total_items = len(samples)

        # 모든 샘플의 프롬프트를 한번에 준비
        batch_inputs = []
        for sample in samples:
            try:
                judge_type_str = sample.get("judge_type", self.default_judge_type.value)
                j_type = JudgeType(judge_type_str)

                if j_type == JudgeType.RESPONSE_COMPARISON:
                    filled_prompt = f"""Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants. Choose the assistant that follows the instructions and answers the question better.

### Instruction:
{sample.get('input', '').strip()}

### Response A:
{sample.get('prediction', '').strip()}

### Response B:
{sample.get('model_response_b', '').strip()}

Compare these responses and provide your verdict in this exact format:
[[A]] if Response A is better, or [[B]] if Response B is better."""
                else:
                    template = self.prompt_templates.get(j_type)
                    if template is None:
                        raise ValueError(f"No template found for judge_type: {j_type}")
                    filled_prompt = template.format(
                        rubric=sample.get("rubric", "").strip(),
                        response=sample.get("prediction", "").strip(),
                        gold=sample.get("reference", "").strip(),
                        response_b=sample.get("model_response_b", "").strip()
                    )

                batch_inputs.append({
                    "system": "You are an expert evaluator. Always provide your verdict in the exact format specified in the prompt.",
                    "input": filled_prompt
                })
            except Exception as e:
                logger.error(f"Error preparing prompt: {e}")
                batch_inputs.append(None)

        # 한번에 judge_batch 호출
        try:
            judge_responses = self.multi_judge_model.judge_batch(
                [inp for inp in batch_inputs if inp is not None]
            )
        except Exception as e:
            logger.error(f"Error in judge_batch: {e}")
            return {"error": str(e)}

        # 결과 처리
        response_idx = 0
        for i, sample in enumerate(samples):
            if batch_inputs[i] is None:
                continue

            try:
                judge_response = judge_responses[response_idx]["prediction"]
                response_idx += 1

                j_type = JudgeType(sample.get("judge_type", self.default_judge_type.value))
                
                # Pointwise 평가(RUBRIC_AND_RESPONSE)인 경우 간소화
                if j_type == JudgeType.RUBRIC_AND_RESPONSE:
                    # 원본 prediction에서 직접 점수 파싱
                    original_prediction = sample.get("prediction", "")
                    score_pattern = r"\[RESULT\]\s*(\d+(?:\.\d+)?)"
                    score_match = re.search(score_pattern, original_prediction)
                    if score_match:
                        score = float(score_match.group(1))
                        sample["judge_score"] = score
                        total_score += score
                        score_count += 1
                    # 아무 필드도 추가하지 않음
                
                # Pairwise 평가(RESPONSE_COMPARISON)인 경우 기존 로직 유지
                elif j_type == JudgeType.RESPONSE_COMPARISON:
                    parser = ResponseComparisonParser()  # PairwiseComparisonParser 대신 사용
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
                            
                            # 여기서 model_name을 설정해야 함
                            if winner == "A":
                                model_name = sample.get("model_a", "unknown")
                            else:  # winner == "B"
                                model_name = sample.get("model_b", "unknown")
                                
                            sample["evaluation"]["parsed"]["model_name"] = model_name
                            
                            is_correct = sample.get("reference") == winner
                            sample["evaluation"]["is_correct"] = is_correct
                            sample["judge_correct"] = is_correct
                            if is_correct:
                                total_correct += 1
                    except ValueError as e:
                        sample["evaluation"] = {
                            "raw_output": judge_response,
                            "parsed": {"error": str(e)},
                        }
                
            except Exception as e:
                # Pointwise일 때는 에러 정보도 추가하지 않고 그냥 무시
                if j_type != JudgeType.RUBRIC_AND_RESPONSE:
                    logger.error(f"Error processing result for sample {i}: {e}")
                    sample["evaluation"] = {
                        "raw_output": str(e),
                        "parsed": {"error": str(e)},
                    }

        # 최종 메트릭 계산
        metrics = {}
        if total_items > 0:
            metrics["accuracy"] = total_correct / total_items
        if score_count > 0:
            metrics["average_score"] = total_score / score_count

        return metrics
