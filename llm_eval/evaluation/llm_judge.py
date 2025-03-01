import logging
import time
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union

from llm_eval.models.multi import MultiModel
from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.prompt_template import JUDGE_PROMPTS
from llm_eval.utils.logging import get_logger
import logging
from tqdm import tqdm

logger = get_logger(name="llm_judge", level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# 1) JudgeType, JudgeInput, Parser classes (original code kept intact)
# ─────────────────────────────────────────────────────────────────────────────

class JudgeType(Enum):
    """Evaluation types for LLM responses."""
    RUBRIC_AND_RESPONSE = "rubric_and_response"
    RUBRIC_RESPONSE_AND_GOLD = "rubric_response_and_gold"
    RESPONSE_COMPARISON = "response_comparison"


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

    It looks for tokens like [[A]] or [[B]] or [[C]](tie).
    """
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        if not response:
            raise ValueError("Response is None")
        if "[[A]]" in response:
            return {"winner": "A", "model_name": model_name or "unknown"}
        elif "[[B]]" in response:
            return {"winner": "B", "model_name": model_name or "unknown"}
        elif "[[C]]" in response:
            return {"winner": "tie", "model_name": model_name or "unknown"}
        raise ValueError(f"No valid verdict found in response: {response}")


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

    Requirements / Assumptions:
    - The 'multi_judge_model' argument is a MultiModel instance that has a valid 'judge_model' loaded.
    - Each sample in 'samples' should contain:
        * "input" (optional, not strictly used here)
        * "reference": Gold standard or correct answer
        * "prediction": The model's generated answer to evaluate
        * "judge_type": (optional) which type of judging logic to use (RUBRIC_AND_RESPONSE, etc.)
          If absent, uses 'default_judge_type'.
        * "rubric", "model_response_b" (optional, for more advanced judge tasks)
    - The code calls `multi_judge_model.judge_batch(...)` with N prompts, each derived from a template
      matching the 'judge_type'. The judge_batch method is expected to fill 'prediction' with
      the judge model's raw output (one per sample).
    - The raw outputs are then parsed to extract "score", "correct", or "winner", depending on the judge type.
    """

    name = "llm_judge"

    def __init__(
        self,
        model: MultiModel,
        default_judge_type: Union[str, JudgeType] = "rubric_and_response",
        **kwargs
    ):
        """
        Args:
            multi_judge_model (MultiModel):
                A MultiModel instance that has its 'judge_model' set.
            default_judge_type (str|JudgeType):
                The judge type to apply if a sample does not specify 'judge_type'.
                Valid values: "rubric_and_response", "rubric_response_and_gold", "response_comparison".
            kwargs: 
                Additional parameters if needed (not used in this minimal example).
        """
        super().__init__()
        # Convert string to JudgeType if needed
        if isinstance(default_judge_type, str):
            self.default_judge_type = JudgeType(default_judge_type)
        else:
            self.default_judge_type = default_judge_type

        self.multi_judge_model = model
        # Define parsers for each judge type
        self.parsers = {
            JudgeType.RUBRIC_AND_RESPONSE: RubricScoreParser(),
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: GoldComparisonParser(),
            JudgeType.RESPONSE_COMPARISON: PairwiseComparisonParser(),
        }

        # Prompt templates (dict: JudgeType -> str)
        self.prompt_templates = JUDGE_PROMPTS

    def evaluate_predictions(
        self,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Implementation of BaseEvaluator's evaluate_predictions() method:
        1) Build judge prompts from each sample (based on judge_type)
        2) Call `multi_judge_model.judge_batch(...)` to get LLM-based evaluation outputs
        3) Parse each raw output (prediction) using the appropriate parser
        4) Record "judge_score", "judge_correct", "judge_winner", etc. in samples
        5) Compute simple metrics (e.g., average_score, correct_rate) and return

        Returns:
            A dict with metric names (e.g. "average_score", "correct_rate") and values.
        """
        if not samples:
            return {}

        # 1) Build prompts
        prompts = []
        judge_types = []
        for s in samples:
            judge_type_str = s.get("judge_type", self.default_judge_type.value)
            try:
                j_type = JudgeType(judge_type_str)
            except Exception as e:
                logger.error(f"Invalid judge_type '{judge_type_str}' in sample {s}, using default '{self.default_judge_type.value}'.")
                j_type = self.default_judge_type

            template = self.prompt_templates.get(j_type.name, "")
            try:
                filled_prompt = template.format(
                    rubric=s.get("rubric", "").strip(),
                    response=s.get("prediction", "").strip(),
                    gold=s.get("reference", "").strip(),
                    response_b=s.get("model_response_b", "").strip()
                )
            except Exception as e:
                logger.error(f"Error formatting judge prompt for sample {s}: {e}")
                filled_prompt = "Invalid prompt."
            if not filled_prompt.strip():
                logger.warning("Filled judge prompt is empty; assigning default prompt.")
                filled_prompt = "No prompt provided for judge evaluation."
            prompts.append({"input": filled_prompt})
            judge_types.append(j_type)

        # 2) Call multi_judge_model.judge_batch to get raw outputs
        try:
            judged_results = self.multi_judge_model.judge_batch(prompts)
        except Exception as e:
            logger.error(f"Error during judge_batch call: {e}")
            judged_results = [{"prediction": ""} for _ in prompts]


        total_score = 0.0
        score_count = 0
        total_correct = 0
        total_items = len(samples)

        for sample, out, j_type in zip(samples, judged_results, judge_types):
            raw_output = out.get("prediction", "").strip()
            if not raw_output:
                logger.warning(f"Empty judge output for sample {sample}. Using default error output.")
                raw_output = "[[score: 0]]"  # 기본값; 필요에 따라 조정
            parser = self.parsers.get(j_type)
            try:
                parsed = parser.parse(raw_output, model_name="JudgeLLM")
            except ValueError as e:
                logger.error(f"Parser error for sample {sample} with raw_output '{raw_output}': {e}")
                parsed = {"error": str(e)}
            if "evaluation" not in sample or not isinstance(sample["evaluation"], dict):
                sample["evaluation"] = {}
            sample["judge_raw_output"] = raw_output
            sample["judge_parsed"] = parsed
            sample["judge_type"] = j_type.value

            # is_correct 계산 로직
            is_correct = None
            if "correct" in parsed:
                try:
                    is_correct = bool(parsed["correct"])
                except Exception as e:
                    logger.error(f"Error converting 'correct' value {parsed.get('correct')} to bool: {e}")
                    is_correct = False
                if is_correct:
                    total_correct += 1
            sample["judge_correct"] = is_correct

            # Create or update "evaluation" dict
            sample["evaluation"] = {
                "raw_output": raw_output,
                "parsed": parsed,
                "is_correct": is_correct,
            }

            # Check if there's a numeric score
            if "score" in parsed:
                try:
                    sc = float(parsed["score"])
                    score_count += 1
                    total_score += sc
                    sample["judge_score"] = sc
                    sample["evaluation"]["score"] = sc
                except Exception as e:
                    logger.error(f"Error converting score {parsed.get('score')} to float: {e}")
                    sample["judge_score"] = 0.0
                    sample["evaluation"]["score"] = 0.0
                    
            # If there's a winner field
            if "winner" in parsed:
                sample["judge_winner"] = parsed["winner"]
                sample["evaluation"]["winner"] = parsed["winner"]

        metrics = {}
        if score_count > 0:
            metrics["average_score"] = total_score / score_count
        if total_items > 0:
            metrics["correct_rate"] = total_correct / total_items

        return metrics
