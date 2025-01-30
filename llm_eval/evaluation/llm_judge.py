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
        if "[[true]]" in response.lower():
            return {"correct": True, "step": -1, "model_name": model_name or "unknown"}
        elif "[[false]]" in response.lower():
            step_pattern = r"step:\s*\[(\d+)\]"
            match = re.search(step_pattern, response)
            if match:
                return {
                    "correct": False,
                    "step": int(match.group(1)),
                    "model_name": model_name or "unknown"
                }
        raise ValueError(f"No valid verdict found in response: {response}")


# ─────────────────────────────────────────────────────────────────────────────
# 2) MultiLLMJudge (original code) - uses multiple LLMs internally for evaluation
# ─────────────────────────────────────────────────────────────────────────────

class MultiLLMJudge:
    """
    Class responsible for performing evaluations using multiple LLMs.

    It uses MultiModel to handle multiple sub-models, 
    and then uses various parser classes to interpret the results.

    Args:
        models_config: A list of configurations describing the sub-models (LLMs) to load.
        max_retries: Maximum number of retries for possible API issues or timeouts.
        retry_delay: Seconds to wait between retries.
        aggregation_strategy: Strategy for how to aggregate multiple results, e.g., "majority".
        logger: Optional logger instance.

    Attributes:
        multi_model: A MultiModel instance to handle parallel or sequential calls to sub-models.
        parsers: A dictionary mapping JudgeType to the appropriate ResponseParser.
        prompt_templates: A dictionary containing prompt templates indexed by JudgeType.
    """
    def __init__(
        self,
        models_config: List[Dict[str, Any]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        aggregation_strategy: str = "majority",
        logger: Optional[logging.Logger] = None
    ):
        self.multi_model = MultiModel(items_config=models_config)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.aggregation_strategy = aggregation_strategy
        self.logger = logger or logging.getLogger(__name__)
        
        self.parsers = {
            JudgeType.RUBRIC_AND_RESPONSE: RubricScoreParser(),
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: GoldComparisonParser(),
            JudgeType.RESPONSE_COMPARISON: PairwiseComparisonParser()
        }
        self.prompt_templates = JUDGE_PROMPTS  # e.g., {JudgeType.RUBRIC_AND_RESPONSE: "Rubric: {rubric}\n...", ...}

    def judge(self, judge_inputs: List[JudgeInput]) -> List[Dict[str, Any]]:
        """
        Executes LLM-based evaluations for each JudgeInput by constructing appropriate prompts,
        querying the sub-model(s), and parsing the results.

        Args:
            judge_inputs: List of JudgeInput, each containing the type of evaluation and necessary data.

        Returns:
            A list of dictionaries, each containing:
                - "raw_output": The raw LLM output
                - "parsed": The parser's output (e.g., score, correctness)
                - "judge_type": The type of evaluation (string)
        """
        results = []
        prompts = []
        types = []

        # Build prompts for each JudgeInput
        for j_input in judge_inputs:
            template = self.prompt_templates.get(j_input.judge_type, "")
            filled_prompt = template.format(
                rubric=j_input.rubric or "",
                response=j_input.model_response or "",
                gold=j_input.gold_response or "",
                response_b=j_input.model_response_b or ""
            )
            prompts.append({"input": filled_prompt})
            types.append(j_input.judge_type)

        # Use multi_model to generate responses in one batch (if desired)
        outputs = self.multi_model.generate_batch(
            prompts,
            return_logits=False
        )

        # Parse each output according to its JudgeType
        for out, j_type in zip(outputs, types):
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
    Evaluator that uses LLM-as-a-Judge logic to assess the quality of model responses.

    Workflow:
      - For each sample in `samples`, build a JudgeInput (including `prediction`, `reference`, or `rubric`).
      - Pass these to MultiLLMJudge, which calls specialized LLM(s) for scoring/comparison.
      - Parse the results into "judge_parsed" fields in `samples`.
      - Compute overall metrics (e.g., average score, correctness rate) to return.
    """

    name = "llm_judge"

    def __init__(
        self,
        judge_models_config: List[Dict[str, Any]],
        default_judge_type: Union[str, JudgeType] = "rubric_and_response",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        aggregation_strategy: str = "majority",
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Args:
            judge_models_config: Config list for the judge model(s), passed to MultiLLMJudge.
            default_judge_type: If a sample does not specify a judge_type, use this one.
            max_retries: Retries for LLM calls.
            retry_delay: Delay (in seconds) between retries.
            aggregation_strategy: How to handle multiple judge responses (if any).
            logger: Optional logger instance.
            kwargs: Additional parameters if needed.
        """
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.default_judge_type = (
            JudgeType(default_judge_type)
            if isinstance(default_judge_type, str)
            else default_judge_type
        )
        self.judge_engine = MultiLLMJudge(
            models_config=judge_models_config,
            max_retries=max_retries,
            retry_delay=retry_delay,
            aggregation_strategy=aggregation_strategy,
            logger=self.logger
        )

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Implementation of BaseEvaluator's mandatory method:
        1) Convert each sample to a JudgeInput (e.g., including `prediction`, `reference`).
        2) Call judge_engine.judge() to get results from the LLM-based evaluation.
        3) Store the parsed results in `samples` and calculate any desired metrics.

        Returns:
            A dict with metric names and values, e.g.:
                {"average_score": 4.2, "correct_rate": 0.78}
        """
        judge_inputs = []
        for s in samples:
            # If a sample doesn't specify judge_type, use default
            judge_type_str = s.get("judge_type", self.default_judge_type.value)
            judge_type = JudgeType(judge_type_str)

            j_input = JudgeInput(
                judge_type=judge_type,
                model_response=s.get("prediction", ""),
                rubric=s.get("rubric", ""),
                gold_response=s.get("reference", ""),
                model_response_b=s.get("model_response_b", None)
            )
            judge_inputs.append(j_input)

        judge_outputs = self.judge_engine.judge(judge_inputs)
        # judge_outputs: [{"raw_output": "...", "parsed": {...}, "judge_type": "..."}]

        total_score = 0.0
        score_count = 0
        total_correct = 0
        total_items = len(samples)

        for sample, jout in zip(samples, judge_outputs):
            sample["judge_raw_output"] = jout["raw_output"]
            sample["judge_parsed"] = jout["parsed"]
            sample["judge_type"] = jout["judge_type"]

            parsed = jout["parsed"]
            if "score" in parsed:
                score_count += 1
                total_score += float(parsed["score"])
                sample["judge_score"] = float(parsed["score"])

            if "correct" in parsed:
                if parsed["correct"] is True:
                    total_correct += 1
                sample["judge_correct"] = bool(parsed["correct"])

            if "winner" in parsed:
                sample["judge_winner"] = parsed["winner"]
                # e.g. "A", "B", or "tie". It's up to you to interpret that.

        metrics = {}
        if score_count > 0:
            avg_score = total_score / score_count
            metrics["average_score"] = avg_score

        if total_items > 0:
            correct_rate = total_correct / total_items
            metrics["correct_rate"] = correct_rate

        return metrics
